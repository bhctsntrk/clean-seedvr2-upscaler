from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "seedvr2_upscale.py"
SPEC = importlib.util.spec_from_file_location("seedvr2_upscale", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ResolutionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
