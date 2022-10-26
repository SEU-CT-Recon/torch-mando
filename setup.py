from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from buildLib import buildLib

with open("README.md", "r") as fh:
    long_description = fh.read()

cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
print(f"Using CUDA_HOME={cuda_home}")
buildLib((61,), nvcc='/usr/local/cuda/bin/nvcc')

setup(
    name='torch_mango',
    version="0.0.1",
    author="z0gSh1u",
    author_email="zx.cs@qq.com",
    description="Fan-beam CT Forward Projection And Reconstruction in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/z0gSh1u/torch-mango",
    packages=['torch_mango'],
    package_dir={
        'torch_mango': './torch_mango',
    },
    ext_modules=[
        CUDAExtension('torch_mango_cuda', [os.path.abspath('src/torchbind.cpp')],
                      include_dirs=[os.path.abspath('include')],
                      library_dirs=[os.path.abspath("objs")],
                      libraries=["mango"],
                      extra_compile_args=["-static", "-static-libgcc", "-static-libstdc++"],
                      extra_link_args=["-Wl,--strip-all"])
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
)
