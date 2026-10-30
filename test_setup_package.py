"""The wheel build has to ship renderutils/*.py, not only the extension."""
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKFLOWS = ROOT / ".github" / "workflows"


def load_setup():
    sys.modules.pop("nvdiffrec_setup", None)
    spec = importlib.util.spec_from_file_location("nvdiffrec_setup", ROOT / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreparePackageTreeTest(unittest.TestCase):
    def test_wrapper_is_installed_and_renderer_modules_are_left_out(self):
        setup = load_setup()
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        package = Path(tmp.name) / "nvdiffrec_render"
        renderutils = package / "renderutils"
        renderutils.mkdir(parents=True)
        (package / "render.py").write_text("import nvdiffrast\n", encoding="utf-8")
        (package / "light.py").write_text("import nvdiffrast\n", encoding="utf-8")
        (renderutils / "bsdf.py").write_text("def bsdf_lambert():\n    pass\n", encoding="utf-8")
        (renderutils / "__init__.py").write_text(
            "from .ops import xfm_points\n",
            encoding="utf-8",
        )
        wrapper = Path(tmp.name) / "renderutils_ops.py"
        wrapper.write_text("def xfm_points():\n    return 'wrapped'\n", encoding="utf-8")

        setup.prepare_package_tree(package, wrapper)

        self.assertFalse((package / "render.py").exists())
        self.assertFalse((package / "light.py").exists())
        self.assertIn("render utilities", (package / "__init__.py").read_text(encoding="utf-8"))
        self.assertEqual((renderutils / "ops.py").read_text(encoding="utf-8"), wrapper.read_text(encoding="utf-8"))
        self.assertTrue((renderutils / "bsdf.py").exists())

    def test_missing_clone_or_wrapper_fails(self):
        setup = load_setup()
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with self.assertRaises(SystemExit):
            setup.prepare_package_tree(Path(tmp.name) / "missing", Path(tmp.name) / "nope.py")
        package = Path(tmp.name) / "nvdiffrec_render"
        (package / "renderutils").mkdir(parents=True)
        with self.assertRaises(SystemExit) as raised:
            setup.prepare_package_tree(package, Path(tmp.name) / "renderutils_ops.py")
        self.assertIn("renderutils_ops.py", str(raised.exception))


class WorkflowWrapperTest(unittest.TestCase):
    def test_builds_copy_the_wrapper_and_do_not_alias_extension_names(self):
        for path in sorted(WORKFLOWS.glob("build-cu*-torch*.yml")):
            text = path.read_text(encoding="utf-8")
            clones = text.count("git clone https://github.com/NVlabs/nvdiffrec.git")
            copies = text.count("cp setup.py renderutils_ops.py nvdiffrec_render_pkg/")
            self.assertEqual(copies, clones, path.name)
            self.assertGreater(copies, 0, path.name)
            self.assertNotIn("_C.xfm_points", text, path.name)
            self.assertNotIn("Patch ops.py", text, path.name)


if __name__ == "__main__":
    unittest.main()
