// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                                                           \
  {                                                                                                \
    cudaError_t e = cudaGetLastError();                                                            \
    if (e != cudaSuccess) {                                                                        \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));             \
      exit(0);                                                                                     \
    }                                                                                              \
  }

#define PI 3.1415926536f

#define KERNEL_RAMP "Ramp"
#define KERNEL_HAMMING "HammingFilter"
#define KERNEL_GAUSSIAN_RAMP "GaussianApodizedRamp"
#define KERNEL_NONE "None"