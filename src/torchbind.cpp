#include <torch/extension.h>
#include "fpj.h"
#include "fbp.h"
#include "utils.h"

#include <stdio.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
  CHECK_CUDA(x);                                                                                   \
  CHECK_CONTIGUOUS(x)

torch::Tensor fanbeam_fpj(torch::Tensor x, float sid, float sdd, int views, 
                          int detElementCount, float detEleSize, 
                          int imgDim, float imgPixelSize, 
                          float startAngle, float totalScanAngle, 
                          float offcenter, float imgXCenter, float imgYCenter
                          float fpjStepSize, int oversample, bool curvedDetector,
                          bool pmatrixFlag, torch::Tensor pmatrix_array, float pmatrix_eltsize, 
                          bool nonuniformSID, torch::Tensor sid_array, 
                          bool nonuniformSDD, torch::Tensor sdd_array,
                          bool nonuniformScanAngle, torch::Tensor scan_angle_array,
                          bool nonuniformOffCenter, torch::Tensor offcenter_array) {
  CHECK_INPUT(x);

  auto dtype = x.dtype();
  const int batch_size = x.size(0);

  auto options = torch::TensorOptions().dtype(dtype).device(x.device());
  auto y = torch::empty({batch_size, views, detElementCount}, options);

  mandoCudaFpj(x.data_ptr<float>(), batch_size, sid, sdd, views, 
               detElementCount, detEleSize,
               imgDim, imgPixelSize, 
               startAngle, totalScanAngle, 
               offcenter, imgXCenter, imgYCenter,
               fpjStepSize, oversample, curvedDetector,
               pmatrixFlag, pmatrix_array.data_ptr<float>(), pmatrix_eltsize, 
               nonuniformSID, sid_array.data_ptr<float>(), 
               nonuniformSDD, sdd_array.data_ptr<float>(),
               nonuniformScanAngle, scan_angle_array.data_ptr<float>(), 
               nonuniformOffCenter, offcenter_array.data_ptr<float>(), 
               y.data_ptr<float>());

  return y;
}

torch::Tensor fanbeam_fbp(torch::Tensor x, int sgmHeight, int sgmWidth, int views, int reconKernelEnum, 
                          float reconKernelParam, float totalScanAngle, float detElementSize, float detOffCenter, 
                          float sid, float sdd, int imgDim, float imgPixelSize, float imgRot, 
                          float imgXCenter, float imgYCenter, bool curvedDetector, bool fovCrop,
                          bool pmatrixFlag, torch::Tensor pmatrix_array, float pmatrix_eltsize, 
                          bool nonuniformSID, torch::Tensor sid_array, 
                          bool nonuniformSDD, torch::Tensor sdd_array,
                          bool nonuniformScanAngle, torch::Tensor scan_angle_array,
                          bool nonuniformOffCenter, torch::Tensor offcenter_array) {
  CHECK_INPUT(x);

  auto dtype = x.dtype();
  const int batch_size = x.size(0);

  auto options = torch::TensorOptions().dtype(dtype).device(x.device());
  auto y = torch::empty({batch_size, imgDim, imgDim}, options);

  mandoCudaFbp(x.data_ptr<float>(), batch_size, sgmHeight, sgmWidth, views, reconKernelEnum,
               reconKernelParam, totalScanAngle, detElementSize, detOffCenter, 
               sid, sdd, imgDim, imgPixelSize, imgRot, 
               imgXCenter, imgYCenter, curvedDetector, fovCrop,
               pmatrixFlag, pmatrix_array.data_ptr<float>(), pmatrix_eltsize, 
               nonuniformSID, sid_array.data_ptr<float>(), 
               nonuniformSDD, sdd_array.data_ptr<float>(),
               nonuniformScanAngle, scan_angle_array.data_ptr<float>(), 
               nonuniformOffCenter, offcenter_array.data_ptr<float>(),
               y.data_ptr<float>());

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fanbeam_fpj, "Fan beam forward projection");
  m.def("backward", &fanbeam_fbp, "Fan beam back projection");
}
