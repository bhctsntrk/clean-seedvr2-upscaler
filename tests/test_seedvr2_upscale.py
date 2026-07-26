from __future__ import annotations

import argparse
import importlib.util
import os
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
                MODULE.prepare_output(source, output, [source])


class CleanupTests(unittest.TestCase):
    def test_session_redirects_caches_and_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            with patch.dict(os.environ, {"SEEDVR2_TEMP_ROOT": value}):
                cache, cleanup_root = MODULE.create_temporary_environment()
                self.assertTrue(cache.is_dir())
                self.assertTrue(cleanup_root.name.startswith("seedvr2-batch-"))
                self.assertTrue(
                    Path(os.environ["UV_CACHE_DIR"]).is_relative_to(cleanup_root)
                )
                MODULE.remove_temporary_environment(cleanup_root)
                self.assertFalse(cleanup_root.exists())

    def test_cleanup_rejects_unowned_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="not-owned-") as value:
            directory = Path(value)
            with self.assertRaises(RuntimeError):
                MODULE.remove_temporary_environment(directory)
            self.assertTrue(directory.exists())


if __name__ == "__main__":
    unittest.main()
