#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
rasterizor_root = "submodules/diff-gaussian-rasterization"
rasterizor_sources = [
    "cuda_rasterizer/rasterizer_impl.cu",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/backward.cu",
    "rasterize_points.cu",
    "ext.cpp"]
simpleknn_root = "submodules/simple-knn"
simpleknn_sources = [
    "spatial.cu",
    "simple_knn.cu",
    "ext.cpp"]

package_dir = {
    'gaussian_splatting': 'gaussian_splatting',
    'gaussian_splatting.utils': 'gaussian_splatting/utils',
    'gaussian_splatting.trainer': 'gaussian_splatting/trainer',
    'gaussian_splatting.dataset': 'gaussian_splatting/dataset',
    'gaussian_splatting.dataset.colmap': 'gaussian_splatting/dataset/colmap',
    'gaussian_splatting.diff_gaussian_rasterization': 'submodules/diff-gaussian-rasterization/diff_gaussian_rasterization',
    'gaussian_splatting.simple_knn': 'submodules/simple-knn/simple_knn',
}


cxx_compiler_flags = []
nvcc_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")
    nvcc_compiler_flags.append("-allow-unsupported-compiler")

setup(
    name="gaussian_splatting",
    packages=[key for key in package_dir],
    package_dir=package_dir,
    ext_modules=[
        CUDAExtension(
            name="gaussian_splatting.diff_gaussian_rasterization._C",
            sources=[os.path.join(rasterizor_root, source) for source in rasterizor_sources],
            extra_compile_args={"nvcc": nvcc_compiler_flags + ["-I" + os.path.join(os.path.abspath(rasterizor_root), "third_party/glm/")]}
        ),
        CUDAExtension(
            name="gaussian_splatting.simple_knn._C",
            sources=[os.path.join(simpleknn_root, source) for source in simpleknn_sources],
            extra_compile_args={"nvcc": nvcc_compiler_flags, "cxx": cxx_compiler_flags}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
