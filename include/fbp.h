#pragma once
#ifndef FBP_H
#define FBP_H

void mandoCudaFbp(float *sgm, int batchsize, int sgmHeight, int sgmWidth, int views, int reconKernelEnum, 
                  float reconKernelParam, float totalScanAngle, float detElementSize, float detOffcenter, 
                  float sid, float sdd, int imgDim, float imgPixelSize, float imgRot, 
                  float imgXCenter, float imgYCenter, bool curvedDetector, bool fovCrop,
                  bool pmatrixFlag, float *pmatrix_array, float pmatrix_eltsize, 
                  bool nonuniformSID, float *sid_array, 
                  bool nonuniformSDD, float *sdd_array,
                  bool nonuniformScanAngle, float *scan_angle_array,
                  bool nonuniformOffCenter, float *offcenter_array, 
                  float *img);

#endif
