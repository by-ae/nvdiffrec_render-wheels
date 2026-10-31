"""Python API for the pre-built nvdiffrec renderutils extension.

NVlabs/nvdiffrec ``render/renderutils/c_src/torch_bindings.cpp`` exports
``xfm_fwd``, ``lambert_fwd``, and the other ``*_fwd`` / ``*_bwd`` names.
``renderutils/__init__.py`` imports the names defined here. The ``fp16`` flag
is a constant inside each forward call, not an autograd input, so backward
returns one gradient per forward input and does not append a gradient for it.
"""
import torch


def _plugin():
    try:
        from . import _C
    except ImportError as exc:
        raise ImportError(
            "nvdiffrec_render.renderutils._C failed to import. "
            "Each wheel is built for one PyTorch CUDA version; "
            "check torch.version.cuda against the wheel you installed."
        ) from exc
    return _C


def _check_finite(name, value):
    if torch.is_anomaly_enabled():
        if not bool(torch.all(torch.isfinite(value))):
            raise AssertionError(f"Output of {name} contains inf or NaN")


def _grads(raw, n_values):
    grads = raw if isinstance(raw, tuple) else (raw,)
    if len(grads) != n_values:
        raise RuntimeError(
            f"Compiled backward returned {len(grads)} values, expected {n_values}"
        )
    return grads


def _with_nones(raw, n_values, n_nones):
    result = _grads(raw, n_values) + (None,) * n_nones
    if len(result) == 1:
        return result[0]
    return result


class _FresnelShlick(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f0, f90, cos_theta):
        out = _plugin().fresnel_shlick_fwd(f0, f90, cos_theta, False)
        ctx.save_for_backward(f0, f90, cos_theta)
        return out

    @staticmethod
    def backward(ctx, grad):
        f0, f90, cos_theta = ctx.saved_tensors
        return _grads(_plugin().fresnel_shlick_bwd(f0, f90, cos_theta, grad), 3)


def _fresnel_shlick(f0, f90, cos_theta, use_python=False):
    if use_python:
        from .bsdf import bsdf_fresnel_shlick
        out = bsdf_fresnel_shlick(f0, f90, cos_theta)
    else:
        out = _FresnelShlick.apply(f0, f90, cos_theta)
    _check_finite("_fresnel_shlick", out)
    return out


class _NdfGgx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_sqr, cos_theta):
        out = _plugin().ndf_ggx_fwd(alpha_sqr, cos_theta, False)
        ctx.save_for_backward(alpha_sqr, cos_theta)
        return out

    @staticmethod
    def backward(ctx, grad):
        alpha_sqr, cos_theta = ctx.saved_tensors
        return _grads(_plugin().ndf_ggx_bwd(alpha_sqr, cos_theta, grad), 2)


def _ndf_ggx(alpha_sqr, cos_theta, use_python=False):
    if use_python:
        from .bsdf import bsdf_ndf_ggx
        out = bsdf_ndf_ggx(alpha_sqr, cos_theta)
    else:
        out = _NdfGgx.apply(alpha_sqr, cos_theta)
    _check_finite("_ndf_ggx", out)
    return out


class _LambdaGgx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_sqr, cos_theta):
        out = _plugin().lambda_ggx_fwd(alpha_sqr, cos_theta, False)
        ctx.save_for_backward(alpha_sqr, cos_theta)
        return out

    @staticmethod
    def backward(ctx, grad):
        alpha_sqr, cos_theta = ctx.saved_tensors
        return _grads(_plugin().lambda_ggx_bwd(alpha_sqr, cos_theta, grad), 2)


def _lambda_ggx(alpha_sqr, cos_theta, use_python=False):
    if use_python:
        from .bsdf import bsdf_lambda_ggx
        out = bsdf_lambda_ggx(alpha_sqr, cos_theta)
    else:
        out = _LambdaGgx.apply(alpha_sqr, cos_theta)
    _check_finite("_lambda_ggx", out)
    return out


class _MaskingSmith(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_sqr, cos_theta_i, cos_theta_o):
        ctx.save_for_backward(alpha_sqr, cos_theta_i, cos_theta_o)
        return _plugin().masking_smith_fwd(alpha_sqr, cos_theta_i, cos_theta_o, False)

    @staticmethod
    def backward(ctx, grad):
        alpha_sqr, cos_theta_i, cos_theta_o = ctx.saved_tensors
        return _grads(
            _plugin().masking_smith_bwd(alpha_sqr, cos_theta_i, cos_theta_o, grad),
            3,
        )


def _masking_smith(alpha_sqr, cos_theta_i, cos_theta_o, use_python=False):
    if use_python:
        from .bsdf import bsdf_masking_smith_ggx_correlated
        out = bsdf_masking_smith_ggx_correlated(alpha_sqr, cos_theta_i, cos_theta_o)
    else:
        out = _MaskingSmith.apply(alpha_sqr, cos_theta_i, cos_theta_o)
    _check_finite("_masking_smith", out)
    return out


class _PrepareShadingNormal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl):
        ctx.two_sided_shading = two_sided_shading
        ctx.opengl = opengl
        ctx.save_for_backward(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm)
        return _plugin().prepare_shading_normal_fwd(
            pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm,
            two_sided_shading, opengl, False,
        )

    @staticmethod
    def backward(ctx, grad):
        pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm = ctx.saved_tensors
        return _with_nones(
            _plugin().prepare_shading_normal_bwd(
                pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm,
                grad, ctx.two_sided_shading, ctx.opengl,
            ),
            6,
            2,
        )


def prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading=True, opengl=True, use_python=False):
    """Final shading normal. Tensors are [minibatch, height, width, 3], or broadcastable.

    A missing ``perturbed_nrm`` is treated as an unperturbed tangent-space normal.
    """
    if perturbed_nrm is None:
        perturbed_nrm = torch.tensor(
            [0, 0, 1], dtype=torch.float32, device="cuda", requires_grad=False,
        )[None, None, None, ...]
    if use_python:
        from .bsdf import bsdf_prepare_shading_normal
        out = bsdf_prepare_shading_normal(
            pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm,
            two_sided_shading, opengl,
        )
    else:
        out = _PrepareShadingNormal.apply(
            pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm,
            two_sided_shading, opengl,
        )
    _check_finite("prepare_shading_normal", out)
    return out


class _Lambert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nrm, wi):
        out = _plugin().lambert_fwd(nrm, wi, False)
        ctx.save_for_backward(nrm, wi)
        return out

    @staticmethod
    def backward(ctx, grad):
        nrm, wi = ctx.saved_tensors
        return _grads(_plugin().lambert_bwd(nrm, wi, grad), 2)


def lambert(nrm, wi, use_python=False):
    """Lambertian diffuse. Returns [minibatch, height, width, 1]."""
    if use_python:
        from .bsdf import bsdf_lambert
        out = bsdf_lambert(nrm, wi)
    else:
        out = _Lambert.apply(nrm, wi)
    _check_finite("lambert", out)
    return out


class _FrostbiteDiffuse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nrm, wi, wo, linear_roughness):
        out = _plugin().frostbite_fwd(nrm, wi, wo, linear_roughness, False)
        ctx.save_for_backward(nrm, wi, wo, linear_roughness)
        return out

    @staticmethod
    def backward(ctx, grad):
        nrm, wi, wo, linear_roughness = ctx.saved_tensors
        return _grads(_plugin().frostbite_bwd(nrm, wi, wo, linear_roughness, grad), 4)


def frostbite_diffuse(nrm, wi, wo, linear_roughness, use_python=False):
    """Frostbite normalized Disney diffuse. Returns [minibatch, height, width, 1]."""
    if use_python:
        from .bsdf import bsdf_frostbite
        out = bsdf_frostbite(nrm, wi, wo, linear_roughness)
    else:
        out = _FrostbiteDiffuse.apply(nrm, wi, wo, linear_roughness)
    _check_finite("frostbite_diffuse", out)
    return out


class _PbrSpecular(torch.autograd.Function):
    @staticmethod
    def forward(ctx, col, nrm, wo, wi, alpha, min_roughness):
        ctx.min_roughness = min_roughness
        ctx.save_for_backward(col, nrm, wo, wi, alpha)
        return _plugin().pbr_specular_fwd(col, nrm, wo, wi, alpha, min_roughness, False)

    @staticmethod
    def backward(ctx, grad):
        col, nrm, wo, wi, alpha = ctx.saved_tensors
        return _with_nones(
            _plugin().pbr_specular_bwd(col, nrm, wo, wi, alpha, ctx.min_roughness, grad),
            5,
            1,
        )


def pbr_specular(col, nrm, wo, wi, alpha, min_roughness=0.08, use_python=False):
    """Physically based specular lobe. ``alpha`` is [minibatch, height, width, 1]."""
    if use_python:
        from .bsdf import bsdf_pbr_specular
        out = bsdf_pbr_specular(col, nrm, wo, wi, alpha, min_roughness=min_roughness)
    else:
        out = _PbrSpecular.apply(col, nrm, wo, wi, alpha, min_roughness)
    _check_finite("pbr_specular", out)
    return out


class _PbrBsdf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kd, arm, pos, nrm, view_pos, light_pos, min_roughness, bsdf_id):
        ctx.min_roughness = min_roughness
        ctx.bsdf_id = bsdf_id
        ctx.save_for_backward(kd, arm, pos, nrm, view_pos, light_pos)
        return _plugin().pbr_bsdf_fwd(
            kd, arm, pos, nrm, view_pos, light_pos, min_roughness, bsdf_id, False,
        )

    @staticmethod
    def backward(ctx, grad):
        kd, arm, pos, nrm, view_pos, light_pos = ctx.saved_tensors
        return _with_nones(
            _plugin().pbr_bsdf_bwd(
                kd, arm, pos, nrm, view_pos, light_pos,
                ctx.min_roughness, ctx.bsdf_id, grad,
            ),
            6,
            2,
        )


def pbr_bsdf(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08, bsdf="lambert", use_python=False):
    """Diffuse plus specular. ``bsdf`` is ``lambert`` or ``frostbite``; anything else selects lambert."""
    bsdf_id = 1 if bsdf == "frostbite" else 0
    if use_python:
        from .bsdf import bsdf_pbr
        out = bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, bsdf_id)
    else:
        out = _PbrBsdf.apply(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, bsdf_id)
    _check_finite("pbr_bsdf", out)
    return out


class _DiffuseCubemap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        ctx.save_for_backward(cubemap)
        return _plugin().diffuse_cubemap_fwd(cubemap)

    @staticmethod
    def backward(ctx, grad):
        cubemap, = ctx.saved_tensors
        return _with_nones(_plugin().diffuse_cubemap_bwd(cubemap, grad), 1, 0)


def diffuse_cubemap(cubemap, use_python=False):
    if use_python:
        raise AssertionError("diffuse_cubemap has no PyTorch fallback")
    out = _DiffuseCubemap.apply(cubemap)
    _check_finite("diffuse_cubemap", out)
    return out


class _SpecularCubemap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap, roughness, costheta_cutoff, bounds):
        ctx.roughness = roughness
        ctx.costheta_cutoff = costheta_cutoff
        ctx.save_for_backward(cubemap, bounds)
        return _plugin().specular_cubemap_fwd(cubemap, bounds, roughness, costheta_cutoff)

    @staticmethod
    def backward(ctx, grad):
        cubemap, bounds = ctx.saved_tensors
        return _with_nones(
            _plugin().specular_cubemap_bwd(
                cubemap, bounds, grad, ctx.roughness, ctx.costheta_cutoff,
            ),
            1,
            3,
        )


_ndf_bounds_cache = {}


def _ndf_bounds(resolution, roughness, cutoff):
    """GGX NDF angle that keeps ``cutoff`` of the energy, plus texel bounds from the extension."""
    import numpy as np

    def ndf_ggx(alpha_sqr, costheta):
        costheta = np.clip(costheta, 0.0, 1.0)
        denom = (costheta * alpha_sqr - costheta) * costheta + 1.0
        return alpha_sqr / (denom * denom * np.pi)

    costheta = np.cos(np.linspace(0, np.pi / 2.0, 1000000))
    integrated = np.cumsum(ndf_ggx(roughness ** 4, costheta))
    index = int(np.argmax(integrated >= integrated[-1] * cutoff))
    angle = float(costheta[index])
    return angle, _plugin().specular_bounds(resolution, angle)


def specular_cubemap(cubemap, roughness, cutoff=0.99, use_python=False):
    """Prefilter a cubemap. ``cubemap`` is [6, resolution, resolution, channels].

    The returned tensor drops the alpha channel the filter uses as weight,
    matching ``out[..., 0:3] / out[..., 3:]`` in nvdiffrec.
    """
    if cubemap.shape[0] != 6 or cubemap.shape[1] != cubemap.shape[2]:
        raise AssertionError(f"Bad shape for cubemap tensor: {cubemap.shape}")
    if use_python:
        raise AssertionError("specular_cubemap has no PyTorch fallback")
    key = (cubemap.shape[1], roughness, cutoff)
    if key not in _ndf_bounds_cache:
        _ndf_bounds_cache[key] = _ndf_bounds(cubemap.shape[1], roughness, cutoff)
    costheta_cutoff, bounds = _ndf_bounds_cache[key]
    out = _SpecularCubemap.apply(cubemap, roughness, costheta_cutoff, bounds)
    _check_finite("specular_cubemap", out)
    return out[..., 0:3] / out[..., 3:]


class _ImageLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, target, loss, tonemapper):
        ctx.loss = loss
        ctx.tonemapper = tonemapper
        ctx.save_for_backward(img, target)
        return _plugin().image_loss_fwd(img, target, loss, tonemapper, False)

    @staticmethod
    def backward(ctx, grad):
        img, target = ctx.saved_tensors
        return _with_nones(
            _plugin().image_loss_bwd(img, target, grad, ctx.loss, ctx.tonemapper),
            2,
            2,
        )


def image_loss(img, target, loss="l1", tonemapper="none", use_python=False):
    """HDR image loss reduced to a scalar.

    ``loss`` is ``l1``, ``mse``, ``smape``, or ``relmse``. ``tonemapper`` is
    ``none`` or ``log_srgb``. The compiled kernel returns a per-pixel map;
    this divides the sum by batch, height, and width.
    """
    if use_python:
        from .loss import image_loss_fn
        out = image_loss_fn(img, target, loss, tonemapper)
    else:
        out = _ImageLoss.apply(img, target, loss, tonemapper)
        out = torch.sum(out) / (img.shape[0] * img.shape[1] * img.shape[2])
    _check_finite("image_loss", out)
    return out


class _Xfm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, matrix, is_points):
        ctx.is_points = is_points
        ctx.save_for_backward(points, matrix)
        return _plugin().xfm_fwd(points, matrix, is_points, False)

    @staticmethod
    def backward(ctx, grad):
        points, matrix = ctx.saved_tensors
        # The extension returns a point gradient only. The matrix is not differentiated.
        return _with_nones(_plugin().xfm_bwd(points, matrix, grad, ctx.is_points), 1, 2)


def xfm_points(points, matrix, use_python=False):
    """Transform points. ``points`` is [minibatch, vertices, 3] and ``matrix`` is [minibatch, 4, 4].

    The compiled kernel returns homogeneous points [matrix_batch, vertices, 4].
    """
    if use_python:
        ones = torch.nn.functional.pad(points, pad=(0, 1), mode="constant", value=1.0)
        out = torch.matmul(ones, torch.transpose(matrix, 1, 2))
    else:
        out = _Xfm.apply(points, matrix, True)
    _check_finite("xfm_points", out)
    return out


def xfm_vectors(vectors, matrix, use_python=False):
    """Transform directions. Same argument shapes as ``xfm_points``.

    The compiled kernel returns [matrix_batch, vertices, 3], not homogeneous 4D.
    """
    if use_python:
        zeros = torch.nn.functional.pad(vectors, pad=(0, 1), mode="constant", value=0.0)
        out = torch.matmul(zeros, torch.transpose(matrix, 1, 2))[..., 0:3].contiguous()
    else:
        out = _Xfm.apply(vectors, matrix, False)
    _check_finite("xfm_vectors", out)
    return out
