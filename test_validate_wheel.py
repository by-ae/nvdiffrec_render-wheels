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


ROOT = Path(__file__).resolve().parent
OPS_SOURCE = (ROOT / "renderutils_ops.py").read_text(encoding="utf-8")
PACKAGE_INIT = '"""Pre-built nvdiffrec render utilities."""\n'
RENDERUTILS_INIT = "from .ops import xfm_points, xfm_vectors, lambert\n"


def build_wheel(
    directory: Path,
    metadata: str,
    filename: str = WHEEL_NAME,
    ops_source: str = OPS_SOURCE,
    include_python_api: bool = True,
    extension_name: str = "nvdiffrec_render/renderutils/_C.cpython-310-x86_64-linux-gnu.so",
) -> Path:
    path = directory / filename
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(extension_name, b"\0")
        if include_python_api:
            archive.writestr("nvdiffrec_render/__init__.py", PACKAGE_INIT)
            archive.writestr("nvdiffrec_render/renderutils/__init__.py", RENDERUTILS_INIT)
            archive.writestr("nvdiffrec_render/renderutils/ops.py", ops_source)
        archive.writestr(
            "nvdiffrec_render-0.0.1+cu128torch29.dist-info/WHEEL",
            "Wheel-Version: 1.0\n",
        )
        archive.writestr(
            "nvdiffrec_render-0.0.1+cu128torch29.dist-info/METADATA",
            metadata,
        )
    return path


WINDOWS_METADATA = (
    "Metadata-Version: 2.1\r\n"
    "Name: nvdiffrec-render\r\n"
    "Version: 0.0.1+cu128torch29\r\n"
)


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

    def test_windows_metadata_name_and_crlf_are_accepted(self):
        """The wheel package normalizes the name and writes CRLF on Windows.

        A published cu128 Windows wheel has Name: nvdiffrec-render and
        Version lines ending in CR. Treating that as a missing name and a
        missing version fails the build after the wheel is already good.
        """
        path = build_wheel(self.directory, WINDOWS_METADATA)
        errors, warnings = validate_wheel.validate_wheel(str(path))
        self.assertEqual(errors, [], errors)
        self.assertEqual(warnings, [])

    def test_crlf_version_mismatch_is_still_reported(self):
        metadata = WINDOWS_METADATA.replace(
            "Version: 0.0.1+cu128torch29\r\n",
            "Version: 0.0.1\r\n",
        )
        path = build_wheel(self.directory, metadata)
        errors, _warnings = validate_wheel.validate_wheel(str(path))
        self.assertTrue(any("does not match filename version" in error for error in errors), errors)
        self.assertFalse(any("missing Version field" in error for error in errors), errors)

    def test_crlf_metadata_without_a_version_header_is_rejected(self):
        metadata = (
            "Metadata-Version: 2.1\r\n"
            "Name: nvdiffrec-render\r\n"
        )
        path = build_wheel(self.directory, metadata)
        errors, _warnings = validate_wheel.validate_wheel(str(path))
        self.assertTrue(any("missing Version field" in error for error in errors), errors)

    def test_unrelated_package_name_is_rejected(self):
        path = build_wheel(
            self.directory,
            "Metadata-Version: 2.4\nName: other-package\nVersion: 0.0.1+cu128torch29\n",
        )
        errors, _warnings = validate_wheel.validate_wheel(str(path))
        self.assertTrue(any("package name" in error for error in errors), errors)

    def test_extension_outside_renderutils_is_rejected(self):
        path = build_wheel(
            self.directory,
            "Metadata-Version: 2.4\nName: nvdiffrec_render\nVersion: 0.0.1+cu128torch29\n",
            extension_name="nvdiffrec_render/other.so",
        )
        errors, _warnings = validate_wheel.validate_wheel(str(path))
        self.assertTrue(any("renderutils/_C" in error for error in errors), errors)

    def test_windows_extension_name_is_accepted(self):
        path = build_wheel(
            self.directory,
            "Metadata-Version: 2.4\nName: nvdiffrec_render\nVersion: 0.0.1+cu128torch29\n",
            filename="nvdiffrec_render-0.0.1+cu128torch29-cp310-cp310-win_amd64.whl",
            extension_name="nvdiffrec_render/renderutils/_C.cp310-win_amd64.pyd",
        )
        errors, warnings = validate_wheel.validate_wheel(str(path))
        self.assertEqual(errors, [], errors)
        self.assertEqual(warnings, [])

    def test_wheel_without_the_python_api_is_rejected(self):
        path = build_wheel(
            self.directory,
            "Metadata-Version: 2.4\nName: nvdiffrec_render\nVersion: 0.0.1+cu128torch29\n",
            include_python_api=False,
        )
        errors, _warnings = validate_wheel.validate_wheel(str(path))
        self.assertTrue(any("ops.py" in error for error in errors), errors)
        self.assertTrue(any("__init__.py" in error for error in errors), errors)

    def test_aliased_ops_names_are_rejected(self):
        path = build_wheel(
            self.directory,
            "Metadata-Version: 2.4\nName: nvdiffrec_render\nVersion: 0.0.1+cu128torch29\n",
            ops_source="xfm_points = _C.xfm_points\nlambert = _C.lambert\n",
        )
        errors, _warnings = validate_wheel.validate_wheel(str(path))
        self.assertTrue(any("torch_bindings.cpp" in error for error in errors), errors)

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
