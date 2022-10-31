from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from build import build
import torch


cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
print(f"Using CUDA_HOME={cuda_home}")
cc = torch.cuda.get_device_capability()
cc = f'{cc[0]}{cc[1]}'
build(cc, nvcc='/usr/local/cuda/bin/nvcc')

setup(
    name='torch_mando',
    version="0.0.1",
    author="CandleHouse",
    author_email="769316285@qq.com",
    description="Differentiable Domain Transform in PyTorch based on MandoCT",
    url="https://github.com/CandleHouse/torch-mando",
    packages=['torch_mando'],
    package_dir={
        'torch_mando': './torch_mando',
    },
    ext_modules=[
        CUDAExtension('torch_mando_cuda', [os.path.abspath('src/torchbind.cpp')],
                      include_dirs=[os.path.abspath('include')],
                      library_dirs=[os.path.abspath("objs")],
                      libraries=["mando"],
                      extra_compile_args=["-static", "-static-libgcc", "-static-libstdc++"],
                      extra_link_args=["-Wl,--strip-all"])
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    install_requires=[
        "torch"
    ],
)
