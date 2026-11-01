"""
Setup script for nvdiffrec_render wheel building.
This builds the render utilities from NVlabs/nvdiffrec as a standalone package.
"""
from pathlib import Path
import os
import re
import shutil

from setuptools import setup


PACKAGE_NAME = "nvdiffrec_render"
WRAPPER_FILENAME = "renderutils_ops.py"


def windows_needs_use_cuda(torch_version):
    """True when this PyTorch build rejects compiled_autograd.h without USE_CUDA.

    The v2.9.1 header skips the ``::std::string`` if-constexpr branch only
    when ``defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))``.
    A downstream extension does not define USE_CUDA, and MSVC then reports
    C2872 (ambiguous ``std``) while compiling torch_bindings.cpp. v2.8.0
    still skips the branch for every Windows compile (``#ifdef _WIN32``),
    so older builds stay as they were.
    https://github.com/pytorch/pytorch/issues/166123
    """
    base = torch_version.split("+", 1)[0]
    match = re.match(r"^(\d+)\.(\d+)", base)
    if not match:
        raise SystemExit(f"Cannot read PyTorch version: {torch_version}")
    return (int(match.group(1)), int(match.group(2))) >= (2, 9)


def extension_compile_args(os_name, torch_version):
    """Flags for nvdiffrec_render.renderutils._C."""
    cxx = ["-O3", "-DNVDR_TORCH"]
    nvcc = ["-O3", "-DNVDR_TORCH"]
    if os_name == "nt" and windows_needs_use_cuda(torch_version):
        cxx.append("-DUSE_CUDA")
        nvcc.append("-DUSE_CUDA")
    return {"cxx": cxx, "nvcc": nvcc}


def prepare_package_tree(package_root, wrapper_path):
    """Make the cloned render tree importable and install the ops wrapper.

    NVlabs/nvdiffrec ``render/`` has no ``__init__.py``. ``find_packages``
    will not descend into that directory, so the wheel contains only the
    compiled ``_C`` extension and ``import`` of ``renderutils.ops`` fails.
    Modules next to ``renderutils`` (``render.py`` and the others) import
    nvdiffrast, which this package does not depend on, so they are not shipped.
    """
    package_root = Path(package_root)
    renderutils = package_root / "renderutils"
    if not renderutils.is_dir():
        raise SystemExit(
            f"Missing {renderutils}. Copy NVlabs/nvdiffrec render/ to "
            f"{PACKAGE_NAME}/ before building."
        )
    wrapper_path = Path(wrapper_path)
    if not wrapper_path.is_file():
        raise SystemExit(
            f"Missing {wrapper_path}. Keep {WRAPPER_FILENAME} next to setup.py "
            "so the wheel includes the renderutils Python API."
        )
    for path in package_root.glob("*.py"):
        path.unlink()
    (package_root / "__init__.py").write_text(
        '"""Pre-built nvdiffrec render utilities."""\n',
        encoding="utf-8",
    )
    shutil.copyfile(wrapper_path, renderutils / "ops.py")


def main():
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    root = Path(__file__).resolve().parent
    prepare_package_tree(root / PACKAGE_NAME, root / WRAPPER_FILENAME)

    # Source files for renderutils CUDA extension
    cuda_sources = [
        'nvdiffrec_render/renderutils/c_src/mesh.cu',
        'nvdiffrec_render/renderutils/c_src/loss.cu',
        'nvdiffrec_render/renderutils/c_src/bsdf.cu',
        'nvdiffrec_render/renderutils/c_src/normal.cu',
        'nvdiffrec_render/renderutils/c_src/cubemap.cu',
        'nvdiffrec_render/renderutils/c_src/common.cpp',
        'nvdiffrec_render/renderutils/c_src/torch_bindings.cpp',
    ]

    # Version with optional local version identifier from env
    base_version = '0.0.1'
    local_version = os.environ.get('NVDIFFREC_LOCAL_VERSION', '')
    version = f'{base_version}+{local_version}' if local_version else base_version

    setup(
        name='nvdiffrec_render',
        version=version,
        description='Render utilities from NVlabs/nvdiffrec for differentiable rendering',
        long_description='Pre-built CUDA extensions for nvdiffrec rendering utilities including differentiable mesh operations, BSDF functions, and cubemap handling. Extracted from NVIDIA nvdiffrec for standalone use.',
        author='NVIDIA Corporation (original), by-ae (packaging)',
        url='https://github.com/by-ae/nvdiffrec_render-wheels',
        license='NVIDIA Source Code License',
        packages=['nvdiffrec_render', 'nvdiffrec_render.renderutils'],
        package_data={
            'nvdiffrec_render': ['*.bin'],
        },
        include_package_data=True,
        python_requires='>=3.10',
        ext_modules=[
            CUDAExtension(
                name='nvdiffrec_render.renderutils._C',
                sources=cuda_sources,
                extra_compile_args=extension_compile_args(os.name, torch.__version__),
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        install_requires=[
            'torch>=2.5.0',
            'numpy>=1.21.0',
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: Other/Proprietary License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: Python :: 3.13',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )


if __name__ == '__main__':
    main()
