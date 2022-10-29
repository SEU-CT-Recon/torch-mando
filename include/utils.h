#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                                                           \
  {                                                                                                \
    cudaError_t e = cudaGetLastError();                                                            \
    if (e != cudaSuccess) {                                                                        \
      fprintf(stderr, "Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
      exit(-1);                                                                                     \
    }                                                                                              \
  }

#define PI 3.1415926536f

#define KERNEL_NONE 0
#define KERNEL_RAMP 1
#define KERNEL_HAMMING 2
#define KERNEL_GAUSSIAN_RAMP 4

#endif