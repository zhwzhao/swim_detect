from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if __name__ == '__main__':

    setup(
        name='example',
        version='0.0.0',
        description='Examples illustrating how to use c++ and CUDA in python.',
        install_requires=[
            'numpy',
            'torch>=1.1',
        ],
        author='Jeff Wang',
        license='Apache License 2.0',
        packages=find_packages(),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            CUDAExtension(
                name="cpp_CUDA_code.imgblend_cuda",
                sources=[
                    "cpp_CUDA_code/imgblend_api.cpp",
                    "cpp_CUDA_code/imgblend.cpp",
                    "cpp_CUDA_code/imgblend_gpu.cu",
                ]   
            ),
        ],
    )
