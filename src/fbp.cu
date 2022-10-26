#include "fbp.h"
#include "utils.h"

__global__ void InitDistance(float *distance_array, const float distance, const int V) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < V) {
    distance_array[tid] = distance;
  }
}

__global__ void InitU(float *u, const int N, const float du, const float offcenter) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < N) {
    u[tid] = (tid - (N - 1) / 2.0f) * du + offcenter;
  }
}

__global__ void InitBeta(float *beta, const int V, const float rotation,
                         const float totalScanAngle) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < V) {
    beta[tid] = (totalScanAngle / V * tid + rotation) * PI / 180;
  }
}

__global__ void InitReconKernel_Hamming(float *reconKernel, const int N, const float du,
                                        const float t) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < 2 * N - 1) {
    // the center element index is N-1
    int n = tid - (N - 1);

    // ramp part
    if (n == 0)
      reconKernel[tid] = t / (4 * du * du);
    else if (n % 2 == 0)
      reconKernel[tid] = 0;
    else
      reconKernel[tid] = -t / (n * n * PI * PI * du * du);

    // cosine part
    int sgn = n % 2 == 0 ? 1 : -1;

    reconKernel[tid] +=
        (1 - t) * (sgn / (2 * PI * du * du) * (1.0f / (1 + 2 * n) + 1.0f / (1 - 2 * n)) -
                   1 / (PI * PI * du * du) *
                       (1.0f / (1 + 2 * n) / (1 + 2 * n) + 1.0f / (1 - 2 * n) / (1 - 2 * n)));
  }
}

/*
  Initialize a Gaussian kernel
  This kernel will be used along with the ramp kernel
  delta is in number of pixels, which is the standard deviation of the gaussian
  This kernel is normalized
*/
__global__ void InitReconKernel_GaussianApodized(float *reconKernel, const int N, const float du,
                                                 const float delta) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < 1) {
    // the center element index is N-1
    float temp_sum = 0;
    for (int i = 0; i < 2 * N - 1; i++) {
      int n = i - (N - 1);
      reconKernel[i] = exp(-float(n) * float(n) / 2.0 / delta / delta);
      temp_sum = temp_sum + reconKernel[i];
    }

    for (int i = 0; i < 2 * N - 1; i++) {
      reconKernel[i] = reconKernel[i] / temp_sum / du;
    }
  }
}

/*
  weight the sinogram data
  sgm: sinogram (width x height x slice)
  N: width
  H: height
  V: views
  S: slice
  sdd: source to detector distance
*/
__global__ void WeightSinogram_device(float *sgm, int batch, const float *u, const int N,
                                      const int H, const int V, float *sdd_array,
                                      float totalScanAngle, bool shortScan, float *beta_array,
                                      float *offcenter_array) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  if (col < N && row < V) {
    float offcenter_bias = offcenter_array[row] - offcenter_array[0];
    float u_actual = u[col] + offcenter_bias; // actual u value due to non uniform offcenter

    float sdd = sdd_array[row];
    sgm[batch * N * H + row * N + col] *= sdd * sdd / sqrtf((u_actual) * (u_actual) + sdd * sdd);

    if (shortScan) {
      float beta = abs(beta_array[row] - beta_array[0]);
      float rotation_direction = abs(totalScanAngle) / (totalScanAngle);
      float gamma = atan(u_actual / sdd) * rotation_direction;

      float gamma_max = abs(totalScanAngle) * PI / 180.0f - PI;

      // calculation of the parker weighting
      float weighting = 0;
      if (beta >= 0 && beta < gamma_max - 2 * gamma) {
        weighting = sin(PI / 2 * beta / (gamma_max - 2 * gamma));
        weighting = weighting * weighting;
      } else if (beta >= gamma_max - 2 * gamma && beta < PI - 2 * gamma) {
        weighting = 1;
      } else if (beta >= PI - 2 * gamma && beta <= PI + gamma_max) {
        weighting = sin(PI / 2 * (PI + gamma_max - beta) / (gamma_max + 2 * gamma));
        weighting = weighting * weighting;
      } else {
        // printf("ERROR!");
      }
      sgm[batch * N * H + row * N + col] *= weighting;
    } else {
      ;
    }
  }
}

/*
  convolve the sinogram data
  sgm_flt: sinogram data after convolving
  sgm: initial sinogram data
  reconKernel: reconstruction kernel
  N: sinogram width
  H: sinogram height
  V: number of views
  u: the position (coordinate) of each detector element
  du: detector element size [mm]
*/
__global__ void ConvolveSinogram_device(float *sgm_flt, int batch, const float *sgm,
                                        float *reconKernel, const int N, const int H, const int V,
                                        const float *u, const float du) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  if (col < N && row < V) {
    // temporary variable to speed up
    float sgm_flt_local = 0;

    for (int i = 0; i < N; i++) {
      sgm_flt_local += sgm[batch * N * H + row * N + i] * reconKernel[N - 1 - col + i];
    }
    sgm_flt[batch * N * H + row * N + col] = sgm_flt_local * du;
  }
}

/*
  Copy the sinogram data from one array(pointer) to another array(pointer). This is for KERNEL_NONE
  kernel reconstruction.
  sgm_flt: sinogram data after copy
  sgm: initial sinogram data
  N: sinogram width
  H: sinogram height
  V: number of views
  S: number of slices
*/
__global__ void CopySinogram_device(float *sgm_flt, int batch, const float *sgm, const int N,
                                    const int H, const int V) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  if (col < N && row < V) {
    sgm_flt[batch * N * H + row * N + col] = sgm[batch * N * H + row * N + col];
  }
}

/*
  backproject the image using pixel-driven method
  sgm: sinogram data
  img: image data
  U: each detector element position [mm]
  u: detector pixel array
  beta: view angle [radius]
  N: number of detector elements
  V: number of views
  M: image dimension
  sdd: source to detector distance [mm]
  sid: source to isocenter distance [mm]
  du: detector element size [mm]
  dx: image pixel size [mm]
  (xc, yc): image center position [mm, mm]
*/
__global__ void BackprojectPixelDriven_device(float *sgm, int batch, float *img, float *u,
                                              float *beta, bool shortScan, const int N, const int V,
                                              const int M, float *sdd_array, float *sid_array,
                                              float *offcenter_array, const float dx,
                                              const float xc, const float yc) {

  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  float du = u[1] - u[0];

  if (col < M && row < M) {

    float x = (col - (M - 1) / 2.0f) * dx + xc;
    float y = ((M - 1) / 2.0f - row) * dx + yc;
    float U, u0;
    float mag_factor;
    float w;
    int k;
    float delta_beta; // delta_beta for the integral calculation (nonuniform scan angle)

    // temporary local variable to speed up
    float img_local = 0;

    for (int view = 0; view < V; view++) {
      float offcenter_bias = offcenter_array[view] - offcenter_array[0];
      float sid = sid_array[view];
      float sdd = sdd_array[view];
      // calculation of delta_beta for the integral calculation
      if (view == 0)
        delta_beta = abs(beta[1] - beta[0]);
      else if (view == V - 1)
        delta_beta = abs(beta[view] - beta[view - 1]);
      else
        delta_beta = abs(beta[view + 1] - beta[view - 1]) / 2.0f;

      U = sid - x * cosf(beta[view]) - y * sinf(beta[view]);

      // calculate the magnification
      mag_factor = sdd / U;

      // find u0
      u0 = mag_factor * (x * sinf(beta[view]) - y * cosf(beta[view]));

      k = floorf((u0 - (u[0] + offcenter_bias)) / du);
      if (k < 0 || k + 1 > N - 1) {
        img_local = 0;
        break;
      }

      w = (u0 - (u[k] + offcenter_bias)) / du;

      // Dont consider cone-beam.
      img_local += sid / U / U *
                   (w * sgm[batch * N * V + view * N + k + 1] +
                    (1 - w) * sgm[batch * N * V + view * N + k]) *
                   delta_beta;
    }

    if (shortScan) {
      img[batch * M * M + row * M + col] = img_local;
    } else
      img[batch * M * M + row * M + col] = img_local / 2.0f;
  }
}

void InitializeDistance_Agent(float *&distance_array, const float distance, const int V) {
  if (distance_array != nullptr)
    cudaFree(distance_array);

  cudaMalloc((void **)&distance_array, V * sizeof(float));
  InitDistance<<<(V + 511) / 512, 512>>>(distance_array, distance, V);
}

void InitializeU_Agent(float *&u, const int N, const float du, const float offcenter) {
  if (u != nullptr)
    cudaFree(u);

  cudaMalloc((void **)&u, N * sizeof(float));
  InitU<<<(N + 511) / 512, 512>>>(u, N, du, offcenter);
}

void InitializeBeta_Agent(float *&beta, const int V, const float rotation,
                          const float totalScanAngle) {
  if (beta != nullptr)
    cudaFree(beta);

  cudaMalloc((void **)&beta, V * sizeof(float));
  InitBeta<<<(V + 511) / 512, 512>>>(beta, V, rotation, totalScanAngle);
}

void InitializeReconKernel_Agent(float *&reconKernel, const int N, const float du,
                                 const std::string &kernelName, float kernelParam) {
  if (reconKernel != nullptr)
    cudaFree(reconKernel);

  cudaMalloc((void **)&reconKernel, (2 * N - 1) * sizeof(float));

  if (kernelName == KERNEL_RAMP) {
    InitReconKernel_Hamming<<<(2 * N - 1 + 511) / 512, 512>>>(reconKernel, N, du, 1.0f);
  } else if (kernelName == KERNEL_HAMMING) {
    InitReconKernel_Hamming<<<(2 * N - 1 + 511) / 512, 512>>>(reconKernel, N, du, kernelParam);
  } else if (kernelName == KERNEL_GAUSSIAN_RAMP) {
    InitReconKernel_GaussianApodized<<<(2 * N - 1 + 511) / 512, 512>>>(reconKernel, N, du,
                                                                       kernelParam);
  } else if (kernelName == KERNEL_NONE) {
    // Do not need to do anything
  }
}

void MallocManaged_Agent(float *&p, const int size) { cudaMallocManaged((void **)&p, size); }

void FilterSinogram_Agent(float *sgm, int batch, float *sgm_flt, float *reconKernel, float *u,
                          int sgmWidth, int sgmHeight, int views, float totalScanAngle,
                          bool shortScan, float *beta, float *sdd_array, std::string kernelName,
                          float detEltSize, float *offcenter_array) {
  // Step 1: weight the sinogram
  dim3 grid((sgmWidth + 15) / 16, (sgmHeight + 15) / 16);
  dim3 block(16, 16);

  // Common attenuation imaging
  WeightSinogram_device<<<grid, block>>>(sgm, batch, u, sgmWidth, sgmHeight, views, sdd_array,
                                         totalScanAngle, shortScan, beta, offcenter_array);

  cudaDeviceSynchronize();

  // Step 2: convolve the sinogram
  if (kernelName == KERNEL_GAUSSIAN_RAMP) {
    // if Guassian aposied kernel is used, the sinogram need to be filtered twice
    // first by the ramp filter, then by the gaussian filter
    float du = detEltSize;
    float *reconKernel_ramp;
    cudaMalloc((void **)&reconKernel_ramp, (2 * sgmWidth - 1) * sizeof(float));
    InitReconKernel_Hamming<<<(2 * sgmWidth - 1 + 511) / 512, 512>>>(reconKernel_ramp, sgmWidth, du,
                                                                     1);
    cudaDeviceSynchronize();

    // intermidiate filtration result is saved in sgm_flt_ramp
    float *sgm_flt_ramp;
    cudaMalloc((void **)&sgm_flt_ramp, sgmWidth * views * sizeof(float));
    ConvolveSinogram_device<<<grid, block>>>(sgm_flt_ramp, batch, sgm, reconKernel_ramp, sgmWidth,
                                             sgmHeight, views, u, detEltSize);
    cudaDeviceSynchronize();
    // the height of the filtered sinogram shrinks to number of views, so the convolution parameters
    // need to be adjusted accordingly
    ConvolveSinogram_device<<<grid, block>>>(sgm_flt, batch, sgm_flt_ramp, reconKernel, sgmWidth,
                                             views, views, u, detEltSize);
    cudaDeviceSynchronize();

    cudaFree(reconKernel_ramp);
    cudaFree(sgm_flt_ramp);
  } else if (kernelName == KERNEL_NONE) {
    CopySinogram_device<<<grid, block>>>(sgm_flt, batch, sgm, sgmWidth, sgmHeight, views);
    cudaDeviceSynchronize();
  } else {
    ConvolveSinogram_device<<<grid, block>>>(sgm_flt, batch, sgm, reconKernel, sgmWidth, sgmHeight,
                                             views, u, detEltSize);
    cudaDeviceSynchronize();
  }
}

void BackprojectPixelDriven_Agent(float *sgm_flt, int batch, float *img, float *sdd_array,
                                  float *sid_array, float *offcenter_array, float *u, float *beta,
                                  int imgDim, bool shortScan, int sgmWidth, int views,
                                  float imgPixelSize, float xCenter, float yCenter) {
  dim3 grid((imgDim + 15) / 16, (imgDim + 15) / 16);
  dim3 block(16, 16);

  BackprojectPixelDriven_device<<<grid, block>>>(sgm_flt, batch, img, u, beta, shortScan, sgmWidth,
                                                 views, imgDim, sdd_array, sid_array,
                                                 offcenter_array, imgPixelSize, xCenter, yCenter);
  cudaDeviceSynchronize();
}

void FreeMemory_Agent(float *&p) {
  cudaFree(p);
  p = nullptr;
}

/**
 * This is the very main.
 */
void mangoCudaFbp(float *sgm, int batchsize, int sgmHeight, int sgmWidth, int views,
                  std::string reconKernelName, float reconKernelParam, float totalScanAngle,
                  float detElementSize, float detOffcenter, float sid, float sdd, int imgDim,
                  float imgPixelSize, float imgRot, float imgXCenter, float imgYCenter,
                  float *img) {
  // Initialize parameters.
  float *sddArray = nullptr;
  InitializeDistance_Agent(sddArray, sdd, views);
  float *sidArray = nullptr;
  InitializeDistance_Agent(sidArray, sid, views);
  float *offcenterArray = nullptr;
  InitializeDistance_Agent(offcenterArray, detOffcenter, views);
  float *u = nullptr;
  InitializeU_Agent(u, sgmWidth, detElementSize, detOffcenter);
  float *beta = nullptr;
  InitializeBeta_Agent(beta, views, imgRot, totalScanAngle);
  bool shortScan = 360.0f - abs(totalScanAngle) < 0.01f;
  float *reconKernel = nullptr;
  InitializeReconKernel_Agent(reconKernel, sgmWidth, detElementSize, reconKernelName,
                              reconKernelParam);

  // Filter the sinogram.
  float *filteredSgm = nullptr;
  MallocManaged_Agent(filteredSgm, sgmWidth * sgmWidth * sizeof(float));

  for (int batch = 0; batch < batchsize; batch++) {
    FilterSinogram_Agent(sgm, batch, filteredSgm, reconKernel, u, sgmWidth, sgmHeight, views,
                         totalScanAngle, shortScan, beta, sddArray, reconKernelName, detElementSize,
                         offcenterArray);
    BackprojectPixelDriven_Agent(filteredSgm, batch, img, sddArray, sidArray, offcenterArray, u,
                                 beta, imgDim, shortScan, sgmWidth, views, imgPixelSize, imgXCenter,
                                 imgYCenter);
  }
}