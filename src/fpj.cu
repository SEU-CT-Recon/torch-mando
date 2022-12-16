#include "fpj.h"
#include "utils.h"

#include <cuda_runtime.h>

__global__ void Fpj_InitDistance(float *distance_array, const float distance, const int V) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < V) {
    distance_array[tid] = distance;
  }
}

__global__ void Fpj_InitU(float *u, const int N, const float du, const float offcenter) {
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

__global__ void PMatrixInv2_device(float *pmatrix)
{
    __shared__ float PMatrixInv2[4];

    int isx = threadIdx.x;
    int isy = threadIdx.y;
    float tmpIn;
    float tmpInv;
    // initialize E
    if(isx == isy)
        PMatrixInv2[isy*2 + isx] = 1;
    else
        PMatrixInv2[isy*2 + isx] = 0;

    // Gaussian elimination method for matrix inverse
    for (int i = 0; i < 2; i++)
    {
        if (i == isy && isx < 2 && isy < 2)
        {
            // The main diagonal element is reduced to 1
            tmpIn = pmatrix[i*2 + i];
            pmatrix[i*2 + isx] /= tmpIn;
            PMatrixInv2[i*2 + isx] /= tmpIn;
        }
        __syncthreads();
        if (i != isy && isx < 2 && isy < 2)
        {
            // Reduce the element in the pivot column to 0, and the element in the row changes simultaneously
            tmpInv = pmatrix[isy*2 + i];
            pmatrix[isy*2 + isx] -= tmpInv * pmatrix[i*2 + isx];
            PMatrixInv2[isy*2 + isx] -= tmpInv * PMatrixInv2[i*2 + isx];
        }
        __syncthreads();
    }

    pmatrix[isy*2 + isx] = PMatrixInv2[isy*2 + isx];
}

void Fpj_InitializeDistance_Agent(float *&distance_array, const float distance, const int V) {
  if (distance_array != nullptr) {
    cudaFree(distance_array);
  }

  cudaMalloc((void **)&distance_array, V * sizeof(float));
  Fpj_InitDistance<<<(V + 511) / 512, 512>>>(distance_array, distance, V);
}

void Fpj_InitializeU_Agent(float *&u, const int N, const float du, const float offcenter) {
  if (u != nullptr) {
    cudaFree(u);
  }

  cudaMalloc((void **)&u, N * sizeof(float));
  Fpj_InitU<<<(N + 511) / 512, 512>>>(u, N, du, offcenter);
}

void Fpj_InitializeBeta_Agent(float *&beta, const int V, const float startAngle,
                              const float totalScanAngle) {
  if (beta != nullptr) {
    cudaFree(beta);
  }

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
    float xd = -(sdd - sid) * cosf(beta[row]) + (u[col] + offcenter_bias) * cosf(beta[row] - PI / 2.0f);
    float yd = -(sdd - sid) * sinf(beta[row]) + (u[col] + offcenter_bias) * sinf(beta[row] - PI / 2.0f);
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

__global__ void ForwardProjectionBilinear_pmatrix_device(float* img, int batchsize, float* sgm, 
                                                         const float* u, const float* pmatrix, 
                                                         const float* beta, int M, int N, int V, float dx,
                                                         const float* sid_array, const float* sdd_array, 
                                                         int osSize, float pixelRatio, float fpjStepSize){
  // Notice that in 'ForwardProjectionBilinear_device', outside 'if (conebeam)' stand for fan beam case,
  // however, this is not true in this function, that means fan beam case pmatrix need special treatment.

  int col = threadIdx.x + blockDim.x * blockIdx.x;  //column is direction of elements
  int row = threadIdx.y + blockDim.y * blockIdx.y;  //row is direction of views
  int batch = threadIdx.z + blockDim.z * blockIdx.z;
  //function is parallelly run for each element in each view

  if (col < N && row < V && batch < batchsize) {
    // half of image side length
    float D = float(M) * dx / 2.0f;
    float D_z = 0.0f;
    float sid = sid_array[row];  // now useless for cone beam and fan beam
    // float sdd = sdd_array[row];  // now useless for cone beam and fan beam
    // pmatrix index and params
    int pos_in_matrix = 12 * row;
    float p_14 = pmatrix[pos_in_matrix + 3];
    // float p_24 = pmatrix[pos_in_matrix + 7];
    float p_34 = pmatrix[pos_in_matrix + 11];
    // current source position
    // fan beam case
    float xs = pmatrix[pos_in_matrix + 0] * -p_14 
              + pmatrix[pos_in_matrix + 1] * -p_34;

    float ys = pmatrix[pos_in_matrix + 8] * -p_14 \
              + pmatrix[pos_in_matrix + 9] * -p_34;
    // reset SID from source position
    sid = sqrtf(xs * xs + ys * ys);

    float xd = pmatrix[pos_in_matrix + 0] * (1 * ((col + 0.5f) / float(osSize) * pixelRatio - 0.5f) - p_14) \
              + pmatrix[pos_in_matrix + 1] * (1 - p_34);

    float yd = pmatrix[pos_in_matrix + 8] * (1 * ((col + 0.5f) / float(osSize) * pixelRatio - 0.5f) - p_14) \
              + pmatrix[pos_in_matrix + 9] * (1 - p_34);

    // step point region
    float L_min = sid - sqrt(2 * D * D + D_z * D_z);
    float L_max = sid + sqrt(2 * D * D + D_z * D_z);
    // source to detector element distance
    float sed = sqrtf((xs - xd)*(xs - xd) + (ys - yd)*(ys - yd));// for fan beam case
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
    for (float L = L_min; L <= L_max; L+= fpjStepSize * sqrt(dx * dx / 2.0f)){
      // ratio of [distance: current position to source] to [distance: source to element]
      float ratio_L_sed = L / sed;

      // get the current point position
      x = xs + (xd - xs) * ratio_L_sed;
      y = ys + (yd - ys) * ratio_L_sed;

      // get the current point index
      kx = floorf((x - x0) / dx);
      ky = floorf((y0 - y) / dx);

      // get the image pixel value at the current point
      if(kx>=0 && kx+1<M && ky>=0 && ky+1<M){
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

/*
 sgm_large: sinogram data before binning
 sgm: sinogram data after binning
 N: number of detector elements (after binning)
 V: number of views
 binSize: bin size
*/
__global__ void BinSinogram_device(float *sgm_large, int batchsize, int N, int V, int binSize,
                                   float *sgm) {
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

void ForwardProjectionBilinear_Agent(float *&image, int batchsize, const float *sid_array,
                                     const float *sdd_array, const float *offcenter_array,
                                     const float *u, const float *beta, int detEltCount, float detEltSize,
                                     int oversampleSize, int views, int imgDim, float pixelSize,
                                     float fpjStepSize, bool pmatrixFlag, const float *pmatrix_array,
                                     float pmatrix_eltsize, float *&sinogram) {
  dim3 grid((detEltCount * oversampleSize + 7) / 8, (views + 7) / 8, batchsize);
  dim3 block(8, 8, 1);

  if (pmatrixFlag == false)  // if pmatrix is not applied
    ForwardProjectionBilinear_device<<<grid, block>>>(image, batchsize, sinogram, u, offcenter_array, 
                                                      beta, imgDim, detEltCount * oversampleSize,
                                                      views, pixelSize, sid_array, sdd_array, fpjStepSize);
  else  // if pmatrix is applied
    ForwardProjectionBilinear_pmatrix_device<<<grid, block>>>(image, batchsize, sinogram, u, pmatrix_array, 
                                                              beta, imgDim, detEltCount * oversampleSize,
                                                              views, pixelSize, sid_array, sdd_array, 
                                                              oversampleSize, detEltSize / pmatrix_eltsize, fpjStepSize);

  cudaDeviceSynchronize();
}

void BinSinogram(float *&sinogram_large, int batchsize, int detEltCount, int views,
                 int oversampleSize, float *&sinogram) {
  dim3 grid((detEltCount + 7) / 8, (views + 7) / 8, batchsize);
  dim3 block(8, 8, 1);

  BinSinogram_device<<<grid, block>>>(sinogram_large, batchsize, detEltCount, views, oversampleSize,
                                      sinogram);

  cudaDeviceSynchronize();
}

void Fpj_FreeMemory_Agent(float *&p) {
  cudaFree(p);
  p = nullptr;
}

/**
 * This is the very main.
 */
void mandoCudaFpj(float *img, int batchsize, float offcenter, float sid, float sdd, int views,
                  int detElementCount, float detEleSize, int oversample, float startAngle,
                  float totalScanAngle, int imgDim, float imgPixelSize, float fpjStepSize,
                  bool pmatrixFlag, float *pmatrix_array, float pmatrix_eltsize, 
                  bool nonuniformSID, float *sid_array, bool nonuniformSDD, float *sdd_array,
                  bool nonuniformScanAngle, float *scan_angle_array,
                  bool nonuniformOffCenter, float *offcenter_array, 
                  float *sgm) {
  const unsigned int SgmBytes = batchsize * detElementCount * views * sizeof(float);
  const unsigned int ImgBytes = batchsize * imgDim * imgDim * sizeof(float);
  const unsigned int PmatrixBytes = 12 * views * sizeof(float);

  // Initialize parameters.
  // Pmatrix
  float *pmatrix_array_device = nullptr;
  if (pmatrixFlag == true){
    cudaMalloc(&pmatrix_array_device, PmatrixBytes);
    cudaMemcpy(pmatrix_array_device, pmatrix_array, PmatrixBytes, cudaMemcpyHostToDevice);
    // fan beam case
    // Inverse pMatrix for the first two dimensions
    float* pmatrix = nullptr;
    cudaMalloc((void**)&pmatrix, 4 * sizeof(float));
    uint3 s;s.x = 2;s.y = 2;s.z = 1;

    for (int i = 0; i < views; i++){
        cudaMemcpy(&pmatrix[0], &pmatrix_array_device[12*i + 0], 2 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&pmatrix[2], &pmatrix_array_device[12*i + 8], 2 * sizeof(float), cudaMemcpyDeviceToDevice);

        PMatrixInv2_device <<<1, s>>>(pmatrix);

        cudaMemcpy(&pmatrix_array_device[12*i + 0], &pmatrix[0], 2 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&pmatrix_array_device[12*i + 8], &pmatrix[2], 2 * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    Fpj_FreeMemory_Agent(pmatrix);
  }

  // SID
  float *sidArray = nullptr;
  if (nonuniformSID == true){
    cudaMalloc(&sidArray, views * sizeof(float));
    cudaMemcpy(sidArray, sid_array, views * sizeof(float), cudaMemcpyHostToDevice);
  }
  else
    Fpj_InitializeDistance_Agent(sidArray, sid, views);

  // SDD
  float *sddArray = nullptr;
  if (nonuniformSDD == true){
    cudaMalloc(&sddArray, views * sizeof(float));
    cudaMemcpy(sddArray, sdd_array, views * sizeof(float), cudaMemcpyHostToDevice);
  }
  else
    Fpj_InitializeDistance_Agent(sddArray, sdd, views);

  // Offcenter
  float *offcenterArray = nullptr;
  if (nonuniformOffCenter == true){
    offcenter = offcenter_array[0];
    cudaMalloc(&offcenterArray, views * sizeof(float));
    cudaMemcpy(offcenterArray, offcenter_array, views * sizeof(float), cudaMemcpyHostToDevice);
  }
  else
    Fpj_InitializeDistance_Agent(offcenterArray, offcenter, views);

  // Scan angle
  float *beta = nullptr;
  if (nonuniformScanAngle == true){
    totalScanAngle = (scan_angle_array[views - 1] - scan_angle_array[0] + scan_angle_array[1] - scan_angle_array[0]);  // degree
    for (int i = 0; i < views; i++)
		  scan_angle_array[i] = (startAngle + scan_angle_array[i]) / 180.0f * PI;
    cudaMalloc(&beta, views * sizeof(float));
    cudaMemcpy(beta, scan_angle_array, views * sizeof(float), cudaMemcpyHostToDevice);
  }
  else
    Fpj_InitializeBeta_Agent(beta, views, startAngle, totalScanAngle);

  // Others
  float *u = nullptr;
  Fpj_InitializeU_Agent(u, detElementCount * oversample, detEleSize / (float)oversample, offcenter);
  // Make sure parameters are correct
  cudaCheckError();

  float *sgm_large = nullptr;
  cudaMalloc(&sgm_large, SgmBytes * oversample);
  float *sgm_device = nullptr;
  cudaMalloc(&sgm_device, SgmBytes);
  float *img_device = nullptr;
  cudaMalloc(&img_device, ImgBytes);
  cudaMemcpy(img_device, img, ImgBytes, cudaMemcpyHostToDevice);

  ForwardProjectionBilinear_Agent(img_device, batchsize, sidArray, sddArray, offcenterArray, u,
                                  beta, detElementCount, detEleSize, oversample, views, imgDim, 
                                  imgPixelSize, fpjStepSize, pmatrixFlag, pmatrix_array_device, 
                                  pmatrix_eltsize, sgm_large);
  cudaCheckError();

  BinSinogram(sgm_large, batchsize, detElementCount, views, oversample, sgm_device);
  cudaCheckError();

  cudaMemcpy(sgm, sgm_device, SgmBytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  Fpj_FreeMemory_Agent(img_device);
  Fpj_FreeMemory_Agent(sgm_device);
  Fpj_FreeMemory_Agent(sgm_large);
  // Others parameters
  Fpj_FreeMemory_Agent(pmatrix_array_device);
  Fpj_FreeMemory_Agent(sidArray);
  Fpj_FreeMemory_Agent(sdd_array);
  Fpj_FreeMemory_Agent(offcenterArray);
  Fpj_FreeMemory_Agent(beta);
  Fpj_FreeMemory_Agent(u);
}