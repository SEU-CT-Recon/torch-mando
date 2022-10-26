#pragma once
#ifndef FBP_H
#define FBP_H

#include <string>

void mangoCudaFbp(float *sgm, int batchsize, int sgmHeight, int sgmWidth, int views, std::string reconKernelName,
              float reconKernelParam, float totalScanAngle, float detElementSize,
              float detOffcenter, float sid, float sdd, int imgDim, float imgPixelSize,
              float imgRot, float imgXCenter, float imgYCenter, float *img);

#endif