#include "fpj.h"
#include "utils.h"

#include <stdio.h>

__global__ void Fpj_InitDistance(float *distance_array, const float distance, const int V) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < V) {
    distance_array[tid] = distance;
  }
}

__global__ void Fpj_d(float *u, const int N, const float du, const float offcenter) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < N) {
    u[tid] = (tid - (N - 1) / 2.0f) * du + offcenter;
  }
}

__global__ void Fpj_InitBeta(float *beta, const int V, const float startAngle,
                             const float totalScanAngle) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < V) {
    beta[tid] = (totalScanAngle / V * tid + startAngle) * PI / 180.0f;
  }
}

void Fpj_InitializeDistance_Agent(float *&distance_array, const float distance, const int V) {
  if (distance_array != nullptr)
    cudaFree(distance_array);

  cudaMalloc((void **)&distance_array, V * sizeof(float));
  Fpj_InitDistance<<<(V + 511) / 512, 512>>>(distance_array, distance, V);
}

void Fpj_InitializeU_Agent(float *&u, const int N, const float du, const float offcenter) {
  if (u != nullptr)
    cudaFree(u);

  cudaMalloc((void **)&u, N * sizeof(float));
  Fpj_d<<<(N + 511) / 512, 512>>>(u, N, du, offcenter);
}

void Fpj_InitializeBeta_Agent(float *&beta, const int V, const float startAngle,
                              const float totalScanAngle) {
  if (beta != nullptr)
    cudaFree(beta);

  cudaMalloc((void **)&beta, V * sizeof(float));
  Fpj_InitBeta<<<(V + 511) / 512, 512>>>(beta, V, startAngle, totalScanAngle);
}

/*
  img: image data
  sgm: sinogram data
  u: array of each detector element position
  beta: array of each view angle [radian]
  M: image dimension
  N: number of detector elements (sinogram width)
  V: number of views (sinogram height)
  dx: image pixel size [mm]
  sid: source to isocenter distance
  sdd: source to detector distance
*/
__global__ void ForwardProjectionBilinear_device(float *img, int batchsize, float *sgm,
                                                 const float *u, const float *offcenter_array,
                                                 const float *beta, int M, int N, int V, float dx,
                                                 const float *sid_array, const float *sdd_array,
                                                 float fpjStepSize) {
  int col = threadIdx.x + blockDim.x * blockIdx.x; // column is direction of elements
  int row = threadIdx.y + blockDim.y * blockIdx.y; // row is direction of views
  int batch = threadIdx.z + blockDim.z * blockIdx.z;

  if (col < N && row < V && batch < batchsize) {
    // half of image side length
    float D = float(M) * dx / 2.0f;

    // get the sid and sdd for a given view
    float sid = sid_array[row];
    float sdd = sdd_array[row];

    // current source position
    float xs = sid * cosf(beta[row]);
    float ys = sid * sinf(beta[row]);

    // calculate offcenter bias
    float offcenter_bias = offcenter_array[row] - offcenter_array[0];

    // current detector element position
    float xd =
        -(sdd - sid) * cosf(beta[row]) + (u[col] + offcenter_bias) * cosf(beta[row] - PI / 2.0f);
    float yd =
        -(sdd - sid) * sinf(beta[row]) + (u[col] + offcenter_bias) * sinf(beta[row] - PI / 2.0f);

    // step point region
    float L_min = sid - sqrt(2 * D * D);
    float L_max = sid + sqrt(2 * D * D);

    // source to detector element distance
    float sed = sqrtf((xs - xd) * (xs - xd) + (ys - yd) * (ys - yd)); // for fan beam case

    // the point position
    float x, y;
    // the point index
    int kx, ky;
    // weighting factor for linear interpolation
    float wx, wy;

    // the most upper left image pixel position
    float x0 = -D + dx / 2.0f;
    float y0 = D - dx / 2.0f;

    sgm[batch * N * V + row * N + col] = 0;

    // calculate line integration
    for (float L = L_min; L <= L_max; L += fpjStepSize * sqrt(dx * dx / 2.0f)) {
      // ratio of [distance: current position to source] to [distance: source to element]
      float ratio_L_sed = L / sed;

      // get the current point position
      x = xs + (xd - xs) * ratio_L_sed;
      y = ys + (yd - ys) * ratio_L_sed;

      // get the current point index
      kx = floorf((x - x0) / dx);
      ky = floorf((y0 - y) / dx);

      // get the image pixel value at the current point
      if (kx >= 0 && kx + 1 < M && ky >= 0 && ky + 1 < M) {
        // get the weighting factor
        wx = (x - kx * dx - x0) / dx;
        wy = (y0 - y - ky * dx) / dx;

        // perform bilinear interpolation
        sgm[batch * N * V + row * N + col] +=
            (1 - wx) * (1 - wy) * img[batch * M * M + ky * M + kx]   // upper left
            + wx * (1 - wy) * img[batch * M * M + ky * M + kx + 1]   // upper right
            + (1 - wx) * wy * img[batch * M * M + (ky + 1) * M + kx] // bottom left
            + wx * wy * img[batch * M * M + (ky + 1) * M + kx + 1];  // bottom right
      }
    }

    sgm[batch * N * V + row * N + col] *= fpjStepSize * sqrt(dx * dx / 2.0f);
  }
}

// sgm_large: sinogram data before binning
// sgm: sinogram data after binning
// N: number of detector elements (after binning)
// V: number of views
// S: number of slices
// binSize: bin size
__global__ void BinSinogram_device(float *sgm_large, int batchsize, float *sgm, int N, int V,
                                   int binSize) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int batch = threadIdx.z + blockDim.z * blockIdx.z;

  if (col < N && row < V && batch < batchsize) {
    // initialization
    sgm[batch * N * V + row * N + col] = 0;
    // sum over each bin
    for (int i = 0; i < binSize; i++) {
      sgm[batch * N * V + row * N + col] +=
          sgm_large[batch * N * binSize * V + row * N * binSize + col * binSize + i];
    }
    // take average
    sgm[batch * N * V + row * N + col] /= binSize;
  }
}

void ForwardProjectionBilinear_Agent(float *&image, int batchsize, float *&sinogram,
                                     const float *sid_array, const float *sdd_array,
                                     const float *offcenter_array, const float *u,
                                     const float *beta, int detEltCount, int oversampleSize,
                                     int views, int imgDim, float pixelSize, float fpjStepSize) {
  dim3 grid((detEltCount * oversampleSize + 7) / 8, (views + 7) / 8, batchsize);
  dim3 block(8, 8, 1);

  ForwardProjectionBilinear_device<<<grid, block>>>(
      image, batchsize, sinogram, u, offcenter_array, beta, imgDim, detEltCount * oversampleSize,
      views, pixelSize, sid_array, sdd_array, fpjStepSize);

  cudaDeviceSynchronize();
}

void BinSinogram(float *&sinogram_large, int batchsize, float *&sinogram, int detEltCount,
                 int views, int oversampleSize) {
  dim3 grid((detEltCount + 7) / 8, (views + 7) / 8, batchsize);
  dim3 block(8, 8, 1);

  BinSinogram_device<<<grid, block>>>(sinogram_large, batchsize, sinogram, detEltCount, views,
                                      oversampleSize);

  cudaDeviceSynchronize();
}

void Fpj_MallocManaged_Agent(float *&p, const int size) { cudaMallocManaged((void **)&p, size); }

void Fpj_FreeMemory_Agent(float *&p) {
  cudaFree(p);
  p = nullptr;
}

/**
 * This is the very main.
 */
void mangoCudaFpj(float *img, int batchsize, float offcenter, float sid, float sdd, int views,
                  int detElementCount, int oversample, float startAngle, float totalScanAngle,
                  int imgDim, float imgPixelSize, float fpjStepSize, float *sgm) {
  float *sddArray = nullptr;
  Fpj_InitializeDistance_Agent(sddArray, sdd, views);
  float *sidArray = nullptr;
  Fpj_InitializeDistance_Agent(sidArray, sid, views);
  float *offcenterArray = nullptr;
  Fpj_InitializeDistance_Agent(offcenterArray, offcenter, views);
  float *u = nullptr;
  Fpj_InitializeU_Agent(u, detElementCount * oversample, detElementCount / (float)oversample,
                        offcenter);
  float *beta = nullptr;
  Fpj_InitializeBeta_Agent(beta, views, startAngle, totalScanAngle);
  // float *sgm_large = nullptr;
  // Fpj_MallocManaged_Agent(sgm_large, detElementCount * oversample * views * sizeof(float));
  // cudaMalloc((void **)&sgm_large, detElementCount * oversample * views * sizeof(float));
  // cudaCheckError();
  cudaCheckError();

  float *img_device = nullptr;
  cudaMalloc((void **)&img_device, imgDim * imgDim * sizeof(float));
  cudaCheckError();
  cudaMemcpy(img_device, img, imgDim * imgDim * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError();

  float *sgm_device = nullptr;
  cudaMalloc((void **)&sgm_device, detElementCount * oversample * views * sizeof(float));
  cudaCheckError();
  printf("FPJ inside batch\n");
  ForwardProjectionBilinear_Agent(img_device, batchsize, sgm_device, sidArray, sddArray,
                                  offcenterArray, u, beta, detElementCount, oversample, views,
                                  imgDim, imgPixelSize, fpjStepSize);
  cudaCheckError();

  // BinSinogram(sgm_large, batchsize, sgm, detElementCount, views, oversample);
  // cudaCheckError();

  cudaMemcpy(sgm, sgm_device, detElementCount * views * sizeof(float), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cudaFree(img_device);
  cudaFree(sgm_device);
}