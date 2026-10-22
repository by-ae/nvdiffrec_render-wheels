"""
Setup script for nvdiffrec_render wheel building.
This builds the render utilities from NVlabs/nvdiffrec as a standalone package.
"""
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

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

setup(
    name='nvdiffrec_render',
    version='0.0.1',
    description='Render utilities from NVlabs/nvdiffrec for differentiable rendering',
    long_description='Pre-built CUDA extensions for nvdiffrec rendering utilities including differentiable mesh operations, BSDF functions, and cubemap handling. Extracted from NVIDIA nvdiffrec for standalone use.',
    author='NVIDIA Corporation (original), Andrea Pozzetti (packaging)',
    url='https://github.com/PozzettiAndrea/nvdiffrec_render-wheels',
    license='NVIDIA Source Code License',
    packages=find_packages(),
    package_data={
        'nvdiffrec_render': ['*.bin'],
    },
    include_package_data=True,
    python_requires='>=3.10',
    ext_modules=[
        CUDAExtension(
            name='nvdiffrec_render.renderutils._C',
            sources=cuda_sources,
            extra_compile_args={
                'cxx': ['-O3', '-DNVDR_TORCH'],
                'nvcc': ['-O3', '-DNVDR_TORCH'],
            },
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
