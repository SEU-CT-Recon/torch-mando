#pragma once
#ifndef FPJ_H
#define FPJ_H

void mandoCudaFpj(float *img, int batchsize, float offcenter, float sid, float sdd, int views,
                  int detElementCount, float detEleSize, int oversample, float startAngle, float totalScanAngle,
                  int imgDim, float imgPixelSize, float fpjStepSize, 
                  bool pmatrixFlag, float *pmatrix_array, float pmatrix_eltsize, 
                  bool nonuniformSID, float *sid_array, bool nonuniformSDD, float *sdd_array,
                  bool nonuniformScanAngle, float *scan_angle_array,
                  bool nonuniformOffCenter, float *offcenter_array, float *sgm);

#endif