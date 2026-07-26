from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "seedvr2_upscale.py"
SPEC = importlib.util.spec_from_file_location("seedvr2_upscale", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ResolutionTests(unittest.TestCase):
    def test_preflight_only_requires_no_positional_arguments(self) -> None:
        args = MODULE.parse_arguments(["--preflight-only"])
        self.assertTrue(args.preflight_only)
        self.assertIsNone(args.input)

    def test_normal_run_keeps_three_positional_arguments(self) -> None:
        args = MODULE.parse_arguments(["in.png", "out.png", "2560x1440"])
        self.assertEqual(args.input, Path("in.png"))
        self.assertEqual(args.output_res.exact_size, (2560, 1440))

    def test_short_side(self) -> None:
        resolution = MODULE.parse_resolution("1440")
        self.assertEqual(resolution.short_side, 1440)
        self.assertIsNone(resolution.exact_size)

    def test_exact_size_accepts_x_and_multiplication_sign(self) -> None:
        for value in ("2560x1440", "2560×1440"):
            with self.subTest(value=value):
                resolution = MODULE.parse_resolution(value)
                self.assertEqual(resolution.short_side, 1440)
                self.assertEqual(resolution.exact_size, (2560, 1440))

    def test_rejects_invalid_resolution(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.parse_resolution("tiny")

    def test_rejects_negative_hardware_limit(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.nonnegative_float("-1")


class InputOutputTests(unittest.TestCase):
    def test_directory_batch_filters_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            (directory / "b.jpg").touch()
            (directory / "a.png").touch()
            (directory / "notes.txt").touch()
            self.assertEqual(
                [path.name for path in MODULE.input_images(directory)],
                ["a.png", "b.jpg"],
            )

    def test_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            source = directory / "image.png"
            output = directory / "result.png"
            source.touch()
            output.touch()
            with self.assertRaises(RuntimeError):
                MODULE.plan_outputs(source, output, [source])

    def test_publish_does_not_overwrite_a_racing_output(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            staged = directory / "staged.png"
            final = directory / "final.png"
            staged.write_bytes(b"generated")
            final.write_bytes(b"user data")
            with self.assertRaises(RuntimeError):
                MODULE.publish_outputs([staged], [final], "test-token")
            self.assertEqual(final.read_bytes(), b"user data")

    def test_publish_copies_validated_data_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            staged = directory / "staged.png"
            final = directory / "nested" / "final.png"
            staged.write_bytes(b"generated")
            MODULE.publish_outputs([staged], [final], "test-token")
            self.assertEqual(final.read_bytes(), b"generated")

    def test_publish_race_preserves_the_competing_file(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            staged = directory / "staged.png"
            final = directory / "final.png"
            staged.write_bytes(b"generated")

            operation_name = "rename" if os.name == "nt" else "link"
            real_operation = getattr(os, operation_name)

            def create_competing_file_then_publish(source, destination):
                Path(destination).write_bytes(b"user data")
                return real_operation(source, destination)

            with patch.object(
                MODULE.os,
                operation_name,
                side_effect=create_competing_file_then_publish,
            ):
                with self.assertRaises(OSError):
                    MODULE.publish_outputs([staged], [final], "test-token")
            self.assertEqual(final.read_bytes(), b"user data")


class CleanupTests(unittest.TestCase):
    def test_persistent_cache_is_not_a_cleanup_target(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            parent = Path(value)
            persistent = parent / "persistent"
            session = MODULE.create_temporary_environment(
                parent / "sessions",
                persistent,
            )
            protected = persistent / "models" / "keep.bin"
            protected.parent.mkdir(parents=True)
            protected.write_bytes(b"keep me")
            MODULE.remove_temporary_environment(session)
            self.assertEqual(protected.read_bytes(), b"keep me")
            self.assertFalse(session.root.exists())

    def test_cleanup_does_not_depend_on_path_is_mount(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            with patch.dict(os.environ, {"SEEDVR2_TEMP_ROOT": value}):
                session = MODULE.create_temporary_environment()
                temporary = session.root / "temporary.bin"
                temporary.write_bytes(b"temporary")
                with patch.object(
                    Path,
                    "is_mount",
                    side_effect=NotImplementedError("unsupported"),
                ):
                    MODULE.remove_temporary_environment(session)
                self.assertFalse(session.root.exists())

    def test_cleanup_fails_closed_at_filesystem_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            with patch.dict(os.environ, {"SEEDVR2_TEMP_ROOT": value}):
                session = MODULE.create_temporary_environment()
                protected = session.root / "protected.bin"
                protected.write_bytes(b"keep me")
                with patch.object(
                    MODULE,
                    "is_filesystem_boundary",
                    return_value=True,
                ):
                    with self.assertRaises(RuntimeError):
                        MODULE.remove_temporary_environment(session)

                quarantine = session.parent / (
                    f"{MODULE.QUARANTINE_PREFIX}{session.token}"
                )
                quarantined_file = quarantine / "session" / protected.name
                self.assertEqual(quarantined_file.read_bytes(), b"keep me")
                MODULE.remove_temporary_environment(session)
                self.assertFalse(quarantine.exists())

    def test_session_redirects_caches_and_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            with patch.dict(os.environ, {"SEEDVR2_TEMP_ROOT": value}):
                session = MODULE.create_temporary_environment()
                self.assertTrue(session.cache.is_dir())
                self.assertTrue(session.root.name.startswith("seedvr2-batch-"))
                self.assertTrue(
                    Path(os.environ["UV_CACHE_DIR"]).is_relative_to(session.root)
                )
                MODULE.remove_temporary_environment(session)
                self.assertFalse(session.root.exists())
                quarantine = session.parent / (
                    f"{MODULE.QUARANTINE_PREFIX}{session.token}"
                )
                self.assertFalse(quarantine.exists())

    def test_cleanup_rejects_unowned_directory(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            parent = Path(value)
            directory = Path(
                tempfile.mkdtemp(prefix="seedvr2-batch-", dir=parent)
            )
            session = MODULE.TemporaryEnvironment(
                cache=directory / "runtime",
                root=directory,
                parent=parent,
                token="not-a-real-token",
            )
            with self.assertRaises(RuntimeError):
                MODULE.remove_temporary_environment(session)
            self.assertTrue(directory.exists())

    def test_cleanup_rejects_tampered_ownership_marker(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            with patch.dict(os.environ, {"SEEDVR2_TEMP_ROOT": value}):
                session = MODULE.create_temporary_environment()
                (session.root / MODULE.SESSION_MARKER).write_text(
                    "tampered",
                    encoding="utf-8",
                )
                with self.assertRaises(RuntimeError):
                    MODULE.remove_temporary_environment(session)
                self.assertTrue(session.root.exists())

    def test_cleanup_rejects_quarantine_name_collision(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            with patch.dict(os.environ, {"SEEDVR2_TEMP_ROOT": value}):
                session = MODULE.create_temporary_environment()
                quarantine = session.parent / (
                    f"{MODULE.QUARANTINE_PREFIX}{session.token}"
                )
                quarantine.mkdir()
                protected = quarantine / "protected.txt"
                protected.write_text("keep me", encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    MODULE.remove_temporary_environment(session)
                self.assertTrue(session.root.exists())
                self.assertEqual(protected.read_text(encoding="utf-8"), "keep me")

    def test_identity_change_is_rejected_without_deleting_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "entry"
            path.write_text("first", encoding="utf-8")
            entry = MODULE.CleanupEntry(
                path=path,
                identity=os.lstat(path),
                kind="file",
            )
            path.unlink()
            path.write_text("replacement", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                MODULE.remove_manifest_entry(entry)
            self.assertEqual(path.read_text(encoding="utf-8"), "replacement")

    def test_read_only_file_inside_owned_session_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            with patch.dict(os.environ, {"SEEDVR2_TEMP_ROOT": value}):
                session = MODULE.create_temporary_environment()
                read_only = session.root / "read-only.bin"
                read_only.write_bytes(b"temporary")
                read_only.chmod(0o444)
                MODULE.remove_temporary_environment(session)
                self.assertFalse(session.root.exists())

    def test_cleanup_never_follows_a_directory_link(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            parent = Path(value)
            outside = parent / "outside"
            outside.mkdir()
            protected = outside / "protected.txt"
            protected.write_text("keep me", encoding="utf-8")
            with patch.dict(os.environ, {"SEEDVR2_TEMP_ROOT": value}):
                session = MODULE.create_temporary_environment()
                link = session.root / "external-link"
                if os.name == "nt":
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                        check=True,
                        capture_output=True,
                    )
                else:
                    link.symlink_to(outside, target_is_directory=True)
                MODULE.remove_temporary_environment(session)

            self.assertTrue(protected.is_file())
            self.assertEqual(protected.read_text(encoding="utf-8"), "keep me")

    def test_cleanup_rejects_a_linked_session_root(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            parent = Path(value)
            outside = parent / "outside"
            outside.mkdir()
            protected = outside / "protected.txt"
            protected.write_text("keep me", encoding="utf-8")
            linked_root = parent / "seedvr2-batch-linked-root"
            if os.name == "nt":
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(linked_root), str(outside)],
                    check=True,
                    capture_output=True,
                )
            else:
                linked_root.symlink_to(outside, target_is_directory=True)

            session = MODULE.TemporaryEnvironment(
                cache=linked_root / "runtime",
                root=linked_root,
                parent=parent,
                token="fake-token",
            )
            with self.assertRaises(RuntimeError):
                MODULE.remove_temporary_environment(session)
            self.assertEqual(protected.read_text(encoding="utf-8"), "keep me")


class PlatformTests(unittest.TestCase):
    def test_platform_cache_directories(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            base = Path(value)
            with patch.dict(os.environ, {"LOCALAPPDATA": str(base)}):
                self.assertEqual(
                    MODULE.platform_cache_directory("Windows"),
                    base / "clean-seedvr2-upscaler",
                )
            with patch.dict(os.environ, {"XDG_CACHE_HOME": str(base)}):
                self.assertEqual(
                    MODULE.platform_cache_directory("Linux"),
                    base / "clean-seedvr2-upscaler",
                )
            self.assertEqual(
                MODULE.platform_cache_directory("Darwin"),
                Path.home() / "Library" / "Caches" / "clean-seedvr2-upscaler",
            )

    def test_nvidia_accelerator_selects_requested_device(self) -> None:
        output = (
            "0, NVIDIA Test GPU Small, 12288, 999.1\n"
            "1, NVIDIA Test GPU Large, 24564, 999.1\n"
        )
        with patch.object(MODULE, "find_nvidia_smi", return_value=Path("nvidia-smi")):
            with patch.object(
                MODULE.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=output),
            ):
                name, memory = MODULE.nvidia_accelerator(1)
        self.assertIn("Test GPU Large", name)
        self.assertEqual(memory, 24564 * 1024**2)

    def test_apple_silicon_uses_unified_memory(self) -> None:
        with patch.object(MODULE.platform, "machine", return_value="arm64"):
            with patch.object(MODULE, "total_ram_bytes", return_value=32 * MODULE.GIB):
                with patch.object(MODULE.os, "cpu_count", return_value=10):
                    profile = MODULE.detect_hardware(0, "Darwin")
        self.assertEqual(profile.accelerator_memory_bytes, 32 * MODULE.GIB)
        self.assertIn("MPS", profile.accelerator)

    def test_intel_mac_is_rejected(self) -> None:
        with patch.object(MODULE.platform, "machine", return_value="x86_64"):
            with patch.object(MODULE, "total_ram_bytes", return_value=32 * MODULE.GIB):
                with self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
                    MODULE.detect_hardware(0, "Darwin")


class PreflightTests(unittest.TestCase):
    def test_preflight_refuses_multiple_insufficient_resources(self) -> None:
        profile = MODULE.HardwareProfile(
            system="Linux",
            machine="x86_64",
            cpu_cores=2,
            ram_bytes=8 * MODULE.GIB,
            accelerator="NVIDIA test",
            accelerator_memory_bytes=4 * MODULE.GIB,
        )
        with tempfile.TemporaryDirectory() as value:
            with patch.object(MODULE, "detect_hardware", return_value=profile):
                with patch.object(
                    MODULE.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=5 * MODULE.GIB),
                ):
                    with self.assertRaisesRegex(RuntimeError, "CPU has 2"):
                        MODULE.run_preflight([Path(value)], 0, 4, 16, 8, 20)

    def test_zero_limits_disable_resource_refusal(self) -> None:
        profile = MODULE.HardwareProfile(
            system="Linux",
            machine="x86_64",
            cpu_cores=1,
            ram_bytes=1,
            accelerator="NVIDIA test",
            accelerator_memory_bytes=1,
        )
        with tempfile.TemporaryDirectory() as value:
            with patch.object(MODULE, "detect_hardware", return_value=profile):
                with patch.object(
                    MODULE.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=1),
                ):
                    actual = MODULE.run_preflight([Path(value)], 0, 0, 0, 0, 0)
        self.assertEqual(actual, profile)


class RuntimeTests(unittest.TestCase):
    def test_valid_cached_source_is_reused_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            cache = Path(value)
            source = cache / "source"
            source.mkdir()
            (source / "inference_cli.py").touch()
            (source / ".clean-seedvr2-source-revision").write_text(
                MODULE.SEEDVR2_REVISION,
                encoding="utf-8",
            )
            with patch.object(
                MODULE.urllib.request,
                "urlopen",
                side_effect=AssertionError("download should not run"),
            ):
                self.assertEqual(MODULE.ensure_source(cache), source)

    def test_incomplete_cached_environment_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            cache = Path(value)
            source = cache / "source"
            source.mkdir()
            (source / "requirements.txt").write_text("", encoding="utf-8")
            fingerprint = MODULE.environment_fingerprint(source, "Linux")
            (cache / f"venv-{fingerprint[:16]}").mkdir()
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                MODULE.ensure_environment(cache, source, Path("uv"), "Linux")

    def test_upscale_uses_model_dir_and_cuda_device_on_linux(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            source = root / "input.png"
            target = root / "generated.png"
            source.touch()
            target.touch()
            commands: list[list[object]] = []
            with patch.object(MODULE, "run", side_effect=lambda command, cwd=None: commands.append(command)):
                MODULE.upscale(
                    source,
                    target,
                    [target],
                    MODULE.Resolution(1440, None),
                    root,
                    Path("python"),
                    root / "models",
                    "Linux",
                    2,
                )
            command = [str(part) for part in commands[0]]
            self.assertIn("--model_dir", command)
            self.assertEqual(command[command.index("--cuda_device") + 1], "2")

    def test_upscale_omits_cuda_device_on_macos(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            source = root / "input.png"
            target = root / "generated.png"
            source.touch()
            target.touch()
            commands: list[list[object]] = []
            with patch.object(MODULE, "run", side_effect=lambda command, cwd=None: commands.append(command)):
                MODULE.upscale(
                    source,
                    target,
                    [target],
                    MODULE.Resolution(1440, None),
                    root,
                    Path("python"),
                    root / "models",
                    "Darwin",
                    0,
                )
            command = [str(part) for part in commands[0]]
            self.assertNotIn("--cuda_device", command)


if __name__ == "__main__":
    unittest.main()
