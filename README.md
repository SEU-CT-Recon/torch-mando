# torch-mango

<p align="center">
  <img src="logo.png"></img>
</p>

<p align="center" style="font-weight: bold">
  Differentiable Fan-beam CT in PyTorch based on MangoCT
</p>

This library introduces **Differentiable Forward Projection (FPJ) and Filtered Back Projection (FBP)** for equidistant Fan-beam CT to [PyTorch](https://pytorch.org/) to enable dual-domain deep learning.

## Installation

- **Check the prerequisites**

  - torch-mango only works on Linux.
  - Prepare CUDA compiler nvcc, PyTorch, and GCC compiler.
  - We've tested on PyTorch 1.10 and CUDA 10.1. Other versions should also work.

- **Clone the repository**

  ```sh
  git clone https://github.com/z0gSh1u/torch-mango.git --depth 1
  cd torch-mango
  ```

- **Modify the setup script**

  In [setup.py](./setup.py), modify the following items according to your machine

  ```python
  COMPUTE_CAPABILITY = (61, ) # See https://developer.nvidia.com/cuda-gpus
  cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
  buildLib(COMPUTE_CAPABILITY, nvcc='/usr/local/cuda/bin/nvcc', cxx='g++')
  ```

- **Install by running**

  ```
  python setup.py install
  ```

## Example

The main idea is to assemble a `MangoConfig` object which includes all necessary settings to perform FPJ and FBP, and pass it to `MangoFanbeamFbp` / `MangoFanbeamFpj` along with the image / sinogram.

```py
cfg = MangoConfig(sid, sdd, 0, totalAngle, views, 2, 0.2, views, detEleCount, imgDim, \
  imgPixelSize, 0, 0, 0, True, KERNEL_RAMP, 0, detEleSize, 0)

# In your network
# If you prefer nn.functional F style
def forward(self, x):
   x = MangoFanbeamFbp(x, cfg)

# If you prefer nn.Module layer style
def __init__(self) -> None:
   self.fbpLayer = MangoFanbeamFbpLayer(cfg)
def forward(self, x):
   x = self.fbpLayer(x)
```

View [example](./example) for code details. We will add a section to introduce all config parameters later.

## Declaration

This work highly depends on previous works by [ustcfdm/mangoct](https://github.com/ustcfdm/mangoct), [njjixu/mangoct](https://gitee.com/njjixu/mangoct), [CandleHouse/mandoct](https://github.com/CandleHouse/mandoct) and [matteo-ronchetti/torch-radon](https://github.com/matteo-ronchetti/torch-radon). This work draws lots of lessons from them. Thanks for their wonderful work.

## Cite this

Please use the following BibTex to cite this work, or click *Cite this repository* on the right.

```
@software{torch_mango,
  author = {Zhuo, Xu and Lu, Yuchen},
  license = {MIT},
  title = {torch-mango: Differentiable Fan-beam CT in PyTorch based on MangoCT},
  url = {https://github.com/z0gSh1u/torch-mango}
}
```

## License

MIT

