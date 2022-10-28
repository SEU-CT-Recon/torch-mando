#pragma once
#ifndef FPJ_H
#define FPJ_H

void mangoCudaFpj(float *img, int batchsize, float offcenter, float sid, float sdd, int views,
                  int detElementCount, float detEleSize, int oversample, float startAngle, float totalScanAngle,
                  int imgDim, float imgPixelSize, float fpjStepSize, float *sgm);

#endif