"""The ops wrapper must call the names torch_bindings.cpp actually exports.

Tests use a fake extension and a fake torch. They check forward arguments
and that backward returns one gradient per forward input.
"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OPS_PATH = ROOT / "renderutils_ops.py"

SINGLE_TENSOR_BWD = {"diffuse_cubemap_bwd", "specular_cubemap_bwd", "xfm_bwd"}
BWD_TUPLE_LEN = {
    "fresnel_shlick_bwd": 3,
    "ndf_ggx_bwd": 2,
    "lambda_ggx_bwd": 2,
    "masking_smith_bwd": 3,
    "prepare_shading_normal_bwd": 6,
    "lambert_bwd": 2,
    "frostbite_bwd": 4,
    "pbr_specular_bwd": 5,
    "pbr_bsdf_bwd": 6,
    "image_loss_bwd": 2,
}
ALLOWED = set(BWD_TUPLE_LEN) | SINGLE_TENSOR_BWD | {
    "fresnel_shlick_fwd",
    "ndf_ggx_fwd",
    "lambda_ggx_fwd",
    "masking_smith_fwd",
    "prepare_shading_normal_fwd",
    "lambert_fwd",
    "frostbite_fwd",
    "pbr_specular_fwd",
    "pbr_bsdf_fwd",
    "diffuse_cubemap_fwd",
    "specular_cubemap_fwd",
    "specular_bounds",
    "image_loss_fwd",
    "xfm_fwd",
}


class FakeTensor:
    def __init__(self, tag, shape=(1, 2, 3, 4)):
        self.tag = tag
        self.shape = shape

    def __getitem__(self, item):
        return FakeTensor(f"{self.tag}[{item!r}]", self.shape)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return other
        return FakeTensor(f"div:{self.tag}", self.shape)

    def contiguous(self):
        return self


class Ctx:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class Plugin:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name not in ALLOWED:
            raise AttributeError(name)

        def call(*args):
            self.calls.append((name, args))
            if name in SINGLE_TENSOR_BWD or name == "specular_bounds" or name.endswith("_fwd"):
                return FakeTensor(name)
            return tuple(FakeTensor(f"{name}:{index}") for index in range(BWD_TUPLE_LEN[name]))

        return call


def install_fake_torch():
    torch = types.ModuleType("torch")
    autograd = types.ModuleType("torch.autograd")
    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")

    class Function:
        @classmethod
        def apply(cls, *args):
            ctx = Ctx()
            cls.last_ctx = ctx
            cls.last_args = args
            return cls.forward(ctx, *args)

    def pad(tensor, pad, mode="constant", value=0.0):
        return FakeTensor(f"pad:{tensor.tag}", tensor.shape)

    functional.pad = pad
    nn.functional = functional
    torch.nn = nn
    torch.autograd = autograd
    autograd.Function = Function
    torch.is_anomaly_enabled = lambda: False
    torch.sum = lambda tensor: tensor
    torch.all = lambda tensor: True
    torch.isfinite = lambda tensor: tensor
    torch.tensor = lambda *args, **kwargs: FakeTensor("const", shape=(3,))
    torch.matmul = lambda left, right: FakeTensor("matmul", left.shape)
    torch.transpose = lambda tensor, dim0, dim1: tensor
    sys.modules["torch"] = torch
    sys.modules["torch.autograd"] = autograd
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional
    return torch


def load_ops(plugin):
    install_fake_torch()
    for name in list(sys.modules):
        if name == "nvdiffrec_render" or name.startswith("nvdiffrec_render."):
            del sys.modules[name]
    package = types.ModuleType("nvdiffrec_render")
    package.__path__ = []
    package.__package__ = "nvdiffrec_render"
    renderutils = types.ModuleType("nvdiffrec_render.renderutils")
    renderutils.__path__ = []
    renderutils.__package__ = "nvdiffrec_render.renderutils"
    renderutils._C = plugin
    sys.modules["nvdiffrec_render"] = package
    sys.modules["nvdiffrec_render.renderutils"] = renderutils
    spec = importlib.util.spec_from_file_location(
        "nvdiffrec_render.renderutils.ops",
        OPS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "nvdiffrec_render.renderutils"
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._ndf_bounds_cache.clear()
    return module


def names(plugin):
    return [name for name, _args in plugin.calls]


class RenderutilsOpsTest(unittest.TestCase):
    def setUp(self):
        self.plugin = Plugin()
        self.ops = load_ops(self.plugin)
        self.tensor = FakeTensor("t")
        self.matrix = FakeTensor("m", shape=(1, 4, 4))

    def test_xfm_points_calls_xfm_fwd_and_returns_three_gradients(self):
        self.ops.xfm_points(self.tensor, self.matrix)
        name, args = self.plugin.calls[-1]
        self.assertEqual(name, "xfm_fwd")
        self.assertEqual(args, (self.tensor, self.matrix, True, False))
        grads = self.ops._Xfm.backward(self.ops._Xfm.last_ctx, FakeTensor("g"))
        self.assertEqual(len(grads), 3)
        self.assertIsInstance(grads[0], FakeTensor)
        self.assertIsNone(grads[1])
        self.assertIsNone(grads[2])
        self.assertIn("xfm_bwd", names(self.plugin))

    def test_xfm_vectors_pass_is_points_false(self):
        self.ops.xfm_vectors(self.tensor, self.matrix)
        name, args = self.plugin.calls[-1]
        self.assertEqual(name, "xfm_fwd")
        self.assertFalse(args[2])
        self.assertFalse(args[3])

    def test_lambert_backward_has_one_gradient_per_tensor(self):
        self.ops.lambert(self.tensor, self.tensor)
        self.assertEqual(self.plugin.calls[-1][0], "lambert_fwd")
        self.assertFalse(self.plugin.calls[-1][1][-1])
        grads = self.ops._Lambert.backward(self.ops._Lambert.last_ctx, FakeTensor("g"))
        self.assertEqual(len(grads), 2)
        self.assertTrue(all(isinstance(grad, FakeTensor) for grad in grads))

    def test_prepare_shading_normal_backward_covers_the_two_flags(self):
        args = (self.tensor,) * 6
        self.ops.prepare_shading_normal(*args, two_sided_shading=False, opengl=True)
        fwd = self.plugin.calls[-1]
        self.assertEqual(fwd[0], "prepare_shading_normal_fwd")
        self.assertEqual(fwd[1][-3:], (False, True, False))
        grads = self.ops._PrepareShadingNormal.backward(
            self.ops._PrepareShadingNormal.last_ctx,
            FakeTensor("g"),
        )
        self.assertEqual(len(grads), 8)
        self.assertTrue(all(isinstance(grad, FakeTensor) for grad in grads[:6]))
        self.assertEqual(grads[6:], (None, None))

    def test_pbr_bsdf_selects_frostbite_and_returns_eight_gradients(self):
        self.ops.pbr_bsdf(*((self.tensor,) * 6), min_roughness=0.08, bsdf="frostbite")
        fwd_args = self.plugin.calls[-1][1]
        self.assertEqual(fwd_args[-2:], (1, False))
        grads = self.ops._PbrBsdf.backward(self.ops._PbrBsdf.last_ctx, FakeTensor("g"))
        self.assertEqual(len(grads), 8)
        self.assertEqual(grads[-2:], (None, None))

    def test_unknown_bsdf_name_selects_lambert(self):
        self.ops.pbr_bsdf(*((self.tensor,) * 6), bsdf="phong")
        self.assertEqual(self.plugin.calls[-1][1][-2], 0)

    def test_image_loss_reduces_by_batch_height_and_width(self):
        image = FakeTensor("img", shape=(2, 4, 5, 3))
        result = self.ops.image_loss(image, image, loss="mse", tonemapper="log_srgb")
        self.assertEqual(
            self.plugin.calls[-1][1],
            (image, image, "mse", "log_srgb", False),
        )
        self.assertEqual(result, image.shape[0] * image.shape[1] * image.shape[2])
        grads = self.ops._ImageLoss.backward(self.ops._ImageLoss.last_ctx, FakeTensor("g"))
        self.assertEqual(len(grads), 4)
        self.assertEqual(grads[-2:], (None, None))

    def test_diffuse_backward_is_a_single_gradient(self):
        cubemap = FakeTensor("cube", shape=(6, 8, 8, 4))
        self.ops.diffuse_cubemap(cubemap)
        grad = self.ops._DiffuseCubemap.backward(
            self.ops._DiffuseCubemap.last_ctx,
            FakeTensor("g"),
        )
        self.assertIsInstance(grad, FakeTensor)
        self.assertNotIsInstance(grad, tuple)

    def test_fresnel_backward_does_not_add_an_extra_none(self):
        self.ops._fresnel_shlick(self.tensor, self.tensor, self.tensor)
        grads = self.ops._FresnelShlick.backward(
            self.ops._FresnelShlick.last_ctx,
            FakeTensor("g"),
        )
        self.assertEqual(len(grads), 3)

    def test_use_python_does_not_call_the_extension(self):
        bsdf = types.ModuleType("nvdiffrec_render.renderutils.bsdf")
        bsdf.bsdf_lambert = lambda nrm, wi: "python-lambert"
        sys.modules["nvdiffrec_render.renderutils.bsdf"] = bsdf
        self.assertEqual(self.ops.lambert(self.tensor, self.tensor, use_python=True), "python-lambert")
        self.assertEqual(self.plugin.calls, [])

    def test_missing_extension_explains_the_cuda_build(self):
        renderutils = sys.modules["nvdiffrec_render.renderutils"]
        del renderutils._C
        with self.assertRaises(ImportError) as raised:
            self.ops.lambert(self.tensor, self.tensor)
        self.assertIn("torch.version.cuda", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
