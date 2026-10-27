#!/usr/bin/env python3
"""Tests for validate_wheel.py using synthetic wheels. No CUDA toolkit."""
import io
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
import tempfile

import validate_wheel


WHEEL_NAME = "nvdiffrec_render-0.0.1+cu128torch29-cp310-cp310-linux_x86_64.whl"


def build_wheel(directory: Path, metadata: str, filename: str = WHEEL_NAME) -> Path:
    path = directory / filename
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nvdiffrec_render/renderutils/__init__.py", "")
        archive.writestr("nvdiffrec_render/renderutils/_C.cpython-310-x86_64-linux-gnu.so", b"\0")
        archive.writestr(
            "nvdiffrec_render-0.0.1+cu128torch29.dist-info/WHEEL",
            "Wheel-Version: 1.0\n",
        )
        archive.writestr(
            "nvdiffrec_render-0.0.1+cu128torch29.dist-info/METADATA",
            metadata,
        )
    return path


class ValidateWheelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def test_metadata_version_must_match_the_filename(self):
        path = build_wheel(
            self.directory,
            "Metadata-Version: 2.4\nName: nvdiffrec_render\nVersion: 0.0.1\n",
        )
        errors, _warnings = validate_wheel.validate_wheel(str(path))
        self.assertTrue(any("does not match filename version" in error for error in errors), errors)

    def test_metadata_version_header_is_not_the_metadata_version_field(self):
        """'Version:' is a substring of 'Metadata-Version:', so a missing Version slipped through."""
        path = build_wheel(
            self.directory,
            "Metadata-Version: 2.4\nName: nvdiffrec_render\n",
        )
        errors, _warnings = validate_wheel.validate_wheel(str(path))
        self.assertTrue(any("missing Version field" in error for error in errors), errors)

    def test_matching_version_is_accepted(self):
        path = build_wheel(
            self.directory,
            "Metadata-Version: 2.4\nName: nvdiffrec_render\nVersion: 0.0.1+cu128torch29\n",
        )
        errors, warnings = validate_wheel.validate_wheel(str(path))
        self.assertEqual(errors, [], errors)
        self.assertEqual(warnings, [])

    def test_cli_validates_every_wheel_argument(self):
        good = build_wheel(
            self.directory,
            "Metadata-Version: 2.4\nName: nvdiffrec_render\nVersion: 0.0.1+cu128torch29\n",
        )
        other_name = "nvdiffrec_render-0.0.1+cu124torch25-cp310-cp310-linux_x86_64.whl"
        bad = build_wheel(
            self.directory,
            "Metadata-Version: 2.4\nName: nvdiffrec_render\nVersion: 0.0.1\n",
            filename=other_name,
        )
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                validate_wheel.main([str(good), str(bad)])
        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
