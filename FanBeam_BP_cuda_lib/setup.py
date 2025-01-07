from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules = [
    CUDAExtension('FanBeam_BP_cuda_lib', [
        'CUDA_ext/helper_grid.cu',
        'CUDA_ext/helper_math.cu',
        'CUDA_ext/FanBeam_BP_cuda.cpp',
        'CUDA_ext/FanBeam_BP_Aw.cu',
    ])
    ]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image']

setup(
    description='PyTorch implementation of FanBeam Projection and Back-Projection base on CUDA',
    author='Zhang Pengcheng',
    author_email='zhangpc198456@163.com',
    license='North University of China License',
    version='0.0.2',
    name='FanBeam_BP_cuda_lib',
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)