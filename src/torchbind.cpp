#include <torch/extension.h>
#include "fpj.h"
#include "fbp.h"
#include "utils.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
  CHECK_CUDA(x);                                                                                   \
  CHECK_CONTIGUOUS(x)

torch::Tensor radon_forward(torch::Tensor x, float offcenter, float sid, float sdd, int views,
                            int detElementCount, int oversample, float startAngle,
                            float totalScanAngle, int imgDim, float imgPixelSize,
                            float fpjStepSize) {
  CHECK_INPUT(x);

  auto dtype = x.dtype();
  const int batch_size = x.size(0);
  const int device = x.device().index();

  auto y = torch::empty({batch_size, rays_cfg.n_angles, rays_cfg.det_count},
                        torch::TensorOptions().dtype(dtype).device(x.device()));

  mangoCudaFpj(x, batch_size, offcenter, sid, sdd, views, detElementCount, oversample, startAngle,
               totalScanAngle, imgDim, imgPixelSize, fpjStepSize, y);

  return y;
}

torch::Tensor radon_backward(torch::Tensor x, int sgmHeight, int sgmWidth, int views,
                             std::string reconKernelName, float reconKernelParam,
                             float totalScanAngle, float detElementSize, float detOffCenter,
                             float sid, float sdd, int imgDim, float imgPixelSize, float imgRot,
                             float imgXCenter, float imgYCenter) {
  CHECK_INPUT(x);

  auto dtype = x.dtype();
  const int batch_size = x.size(0);
  const int device = x.device().index();

  auto y = torch::empty({batch_size, rays_cfg.height, rays_cfg.width},
                        torch::TensorOptions().dtype(dtype).device(x.device()));

  mangoCudaFbp(x, batch_size, sgmHeight, sgmWidth, views, reconKernelName, reconKernelParam,
               totalScanAngle, detElementSize, detOffCenter, sid, sdd, imgDim, imgPixelSize, imgRot,
               imgXCenter, imgYCenter, y);

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &radon_forward, "Radon forward projection");
  m.def("backward", &radon_backward, "Radon back projection");
}
