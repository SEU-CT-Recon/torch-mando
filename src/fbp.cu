#include "fbp.h"
#include "utils.h"

#include <cuda_runtime.h>

__global__ void Fbp_InitDistance(float *distance_array, const float distance, const int V) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < V) {
    distance_array[tid] = distance;
  }
}

__global__ void Fbp_InitU(float *u, const int N, const float du, const float offcenter) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < N) {
    u[tid] = (tid - (N - 1) / 2.0f) * du + offcenter;
  }
}

__global__ void Fbp_InitBeta(float *beta, const int V, const float rotation,
                             const float totalScanAngle) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < V) {
    beta[tid] = (totalScanAngle / V * tid + rotation) * PI / 180;
  }
}

__global__ void InitReconKernel_Hamming(float *reconKernel, const int N, const float du,
                                        const float t, bool curvedDetector, float sdd) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < 2 * N - 1) {
    // the center element index is N-1
    int n = tid - (N - 1);

    // ramp part
    if (n == 0)
      reconKernel[tid] = t / (4 * du * du);
    else if (n % 2 == 0)
      reconKernel[tid] = 0;
    else {
			// the weighting for curved detector and flat panel detector is different 
			if (curvedDetector)
				reconKernel[tid] = -t / (PI * PI * sdd * sdd * sin(float(n) * du / sdd) * sin(float(n) * du / sdd));
			else
				reconKernel[tid] = -t / (n * n * PI * PI * du * du);
		}

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
  sgm: sinogram (width x height)
  N: width
  H: height
  V: views
  sdd: source to detector distance
*/
__global__ void WeightSinogram_device(float *sgm, const float *u, const int N, const int H,
                                      const int V, float *sdd_array, float totalScanAngle,
                                      bool shortScan, float *beta_array, float *offcenter_array,
                                      bool curvedDetector) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int batch = threadIdx.z + blockDim.z * blockIdx.z;

  if (col < N && row < V) {
    float offcenter_bias = offcenter_array[row] - offcenter_array[0];
    float u_actual = u[col] + offcenter_bias; // actual u value due to non uniform offcenter

    float sdd = sdd_array[row];
    // for curved detector or flat panel detector, the function to calculate cos gamma is different
    if (curvedDetector)
      sgm[batch * N * H + row * N + col] *= sdd * cos(u_actual / sdd) * sdd / sqrtf(sdd * sdd);
    else
      sgm[batch * N * H + row * N + col] *= sdd * sdd / sqrtf((u_actual) * (u_actual) + sdd * sdd);

    if (shortScan) {
      // for scans longer than 360 degrees but not muliples of 360, we also need to apply parker weighting
			// for example, for a 600 degrees scan, we also need to apply parker weighting
			uint64_t num_rounds = floorf(abs(totalScanAngle) / 360.0f);
			float remain_angle = abs(totalScanAngle) - num_rounds * 360.0f;

      float beta = abs(beta_array[row] - beta_array[0]);
      float rotation_direction = abs(totalScanAngle) / (totalScanAngle);
      float gamma;
      // for curved detector or flat panel detector, the function to calculate gamma is different
			if (curvedDetector)
				gamma = (u_actual / sdd) * rotation_direction;
			else
				gamma = atan(u_actual / sdd) * rotation_direction;

			float gamma_max = remain_angle * PI / 180.0f - PI;// maximum gamma defined by remain angle

      //calculation of the parker weighting
			float weighting = 0;
			if (remain_angle > 180.0f) {//remain angle is larger than 180 degrees, need to apply parker weighting
				if (beta >= 0 && beta < gamma_max - 2 * gamma) {
					weighting = sin(PI / 2 * beta / (gamma_max - 2 * gamma));
					weighting = weighting * weighting;
				}
				else if (beta >= gamma_max - 2 * gamma && beta < PI * (2 * num_rounds + 1) - 2 * gamma) {
					weighting = 1;
				}
				else if (beta >= PI * (2 * num_rounds + 1) - 2 * gamma && beta <= PI * (2 * num_rounds + 1) + gamma_max) {
					weighting = sin(PI / 2 * (PI + gamma_max - (beta - PI * 2 * num_rounds)) / (gamma_max + 2 * gamma));
					weighting = weighting * weighting;
				}
				else {
					//printf("ERROR!");
				}
			}
			else {//remain angle is less than 180 degree, need to apply a different weighting
				weighting = 1;// This weighting has not been fully investigated;
			}

      sgm[batch * N * H + row * N + col] *= weighting;
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
__global__ void ConvolveSinogram_device(float *sgm_flt, const float *sgm, float *reconKernel,
                                        const int N, const int H, const int V, const float *u,
                                        const float du) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int batch = threadIdx.z + blockDim.z * blockIdx.z;

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
__global__ void CopySinogram_device(float *sgm_flt, const float *sgm, const int N, const int H,
                                    const int V) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int batch = threadIdx.z + blockDim.z * blockIdx.z;

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
__global__ void BackprojectPixelDriven_device(float *sgm, float *u, float *beta, bool shortScan,
                                              const int N, const int V, const int M,
                                              float *sdd_array, float *sid_array,
                                              float *offcenter_array, const float dx,
                                              const float xc, const float yc, bool curvedDetector,
                                              float *img) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int batch = threadIdx.z + blockDim.z * blockIdx.z;

  float du = u[1] - u[0];

  if (col < M && row < M) {
    float x = (col - (M - 1) / 2.0f) * dx + xc;
    float y = ((M - 1) / 2.0f - row) * dx + yc;
    float U, u0;
    float mag_factor;
    float w;
    int k;
    float delta_beta;    // delta_beta for the integral calculation (nonuniform scan angle)
    float total_scan_angle = abs((beta[V - 1] - beta[0])) / float(V - 1) * float(V);
    float img_local = 0; // temporary local variable to speed up

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
      if (curvedDetector)
        u0 = sdd * atan((x * sinf(beta[view]) - y * cosf(beta[view])) / U);
      else
        u0 = mag_factor * (x * sinf(beta[view]) - y * cosf(beta[view]));

      k = floorf((u0 - (u[0] + offcenter_bias)) / du);
      if (k < 0 || k + 1 > N - 1) {
        img_local = 0;
        break;
      }

      w = (u0 - (u[k] + offcenter_bias)) / du;

      float distance_weight = 0;
      if (curvedDetector)
        distance_weight = 1 / (U * U + (x * sinf(beta[view]) - y * cosf(beta[view])) * (x * sinf(beta[view]) - y * cosf(beta[view])));
      else
        distance_weight = 1 / (U * U);

      // Dont consider cone-beam.
      img_local += sid * distance_weight *
                   (w * sgm[batch * N * V + view * N + k + 1] +
                   (1 - w) * sgm[batch * N * V + view * N + k]) *
                   delta_beta;
    }

    //judge whether the scan is a full scan or a short scan
    uint64_t num_rounds = floorf((abs(total_scan_angle) + 0.001f) / (2 * PI));
    if (shortScan)
      img[batch * M * M + row * M + col] = img_local / float(2 * num_rounds + 1.0f);
    else
      img[batch * M * M + row * M + col] = img_local / float(2 * num_rounds);
  }
}

__global__ void BackprojectPixelDriven_pmatrix_device(float* sgm, float* u, float* beta, float* pmatrix, float pmatrix_eltsize,
                                                      bool shortScan, const int N, const int V, const int M, float* sdd_array, 
                                                      float* sid_array, const float dx, const float xc, const float yc, 
                                                      float imgRot, float* img)
//pmatrix_du is the detector elements size when pmatrix calibration is performed
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
  int batch = threadIdx.z + blockDim.z * blockIdx.z;

	float du = u[1] - u[0];

	float imgRot_in_rad = imgRot * PI / 180.0f;
	if (col < M && row < M){
		float x_after_rotation = (col - (M - 1) / 2.0f)*dx + xc;
		float y_after_rotation = ((M - 1) / 2.0f - row)*dx + yc;
		float x = x_after_rotation * cos(imgRot_in_rad) + y_after_rotation * sin(imgRot_in_rad);  // (col - (M - 1) / 2.0f)*dx + xc;
		float y = y_after_rotation * cos(imgRot_in_rad) - x_after_rotation * sin(imgRot_in_rad);  // ((M - 1) / 2.0f - row)*dx + yc;
		float U;
		float w;
		int k;
    float z = 0;
		float delta_beta;     // delta_beta for the integral calculation (nonuniform scan angle)
    float img_local = 0;  // temporary local variable to speed up

    for (int view = 0; view < V; view++){
      float sid = sid_array[view];
      // float sdd = sdd_array[view];
      //calculation of delta_beta for the integral calculation
      if (view == 0)
        delta_beta = abs(beta[1] - beta[0]);
      else if (view == V - 1)
        delta_beta = abs(beta[view] - beta[view - 1]);
      else
        delta_beta = abs(beta[view + 1] - beta[view - 1]) / 2.0f;

      //use pmatrix to calculate the corresponding index on the detector
      int pos_in_matrix = 12 * view;
      float k_u_divide_mag = pmatrix[pos_in_matrix] * x + pmatrix[pos_in_matrix + 1] * y + pmatrix[pos_in_matrix + 2] * z + pmatrix[pos_in_matrix + 3] * 1;
      float one_divide_mag = pmatrix[pos_in_matrix + 8] * x + pmatrix[pos_in_matrix + 9] * y + pmatrix[pos_in_matrix + 10] * z + pmatrix[pos_in_matrix + 11] * 1;

      // the pmatrix maybe calculated when the detector is another binning mode than the recon
      float k_f = k_u_divide_mag / one_divide_mag;// float number of k
      float u_position_true = (k_f + 0.5f)*pmatrix_eltsize;// convert the u_position to real physical size
      k_f = u_position_true / du - 0.5f;// convert the u_position to current pixel index
      
      //float k_f = k_u_divide_mag / one_divide_mag;//float number of k
      k = floorf(k_f);

      //the pmatrix is acquired assuming beta[0]=0
      //however, in a real recon, the image need to be rotated
      //we need to retrieve the beta value for the pmatrix recon
      //for calculation of U
      float beta_pmatrix = beta[view] - beta[0];
      U = sid - x * cosf(beta_pmatrix) - y * sinf(beta_pmatrix);

      if (k<0 || k + 1>N - 1){
        img_local = 0;
        break;
      }

      w = k_f - k;
      // Dont consider cone-beam.
      img_local += sid / U / U *
                   (w * sgm[batch * N * V + view * N + k + 1] +
                   (1 - w) * sgm[batch * N * V + view * N + k]) *
                   delta_beta;
    }

    if (shortScan)
      img[batch * M * M + row * M + col] = img_local;
    else
      img[batch * M * M + row * M + col] = img_local / 2.0f;
	}
}

void Fbp_InitializeDistance_Agent(float *&distance_array, const float distance, const int V) {
  if (distance_array != nullptr) {
    cudaFree(distance_array);
  }
  
  cudaMalloc(&distance_array, V * sizeof(float));
  Fbp_InitDistance<<<(V + 511) / 512, 512>>>(distance_array, distance, V);
}

void Fbp_InitializeU_Agent(float *&u, const int N, const float du, const float offcenter) {
  if (u != nullptr) {
    cudaFree(u);
  }

  cudaMalloc(&u, N * sizeof(float));
  Fbp_InitU<<<(N + 511) / 512, 512>>>(u, N, du, offcenter);
}

void Fbp_InitializeBeta_Agent(float *&beta, const int V, const float rotation,
                              const float totalScanAngle) {
  if (beta != nullptr) {
    cudaFree(beta);
  }

  cudaMalloc(&beta, V * sizeof(float));
  Fbp_InitBeta<<<(V + 511) / 512, 512>>>(beta, V, rotation, totalScanAngle);
}

void Fbp_InitializeReconKernel_Agent(float *&reconKernel, const int N, const float du,
                                     int kernelEnum, float kernelParam, 
                                     bool curvedDetector, float sdd) {
  if (reconKernel != nullptr) {
    cudaFree(reconKernel);
  }

  cudaMalloc(&reconKernel, (2 * N - 1) * sizeof(float));

  if (kernelEnum == KERNEL_RAMP) {
    InitReconKernel_Hamming<<<(2 * N - 1 + 511) / 512, 512>>>(reconKernel, N, du, 1.0f, curvedDetector, sdd);
  } else if (kernelEnum == KERNEL_HAMMING) {
    InitReconKernel_Hamming<<<(2 * N - 1 + 511) / 512, 512>>>(reconKernel, N, du, kernelParam, curvedDetector, sdd);
  } else if (kernelEnum == KERNEL_GAUSSIAN_RAMP) {
    InitReconKernel_GaussianApodized<<<(2 * N - 1 + 511) / 512, 512>>>(reconKernel, N, du,kernelParam);
  } else if (kernelEnum == KERNEL_NONE) {
    // Do not need to do anything
  }
}

void FilterSinogram_Agent(float *sgm, int batchsize, float *reconKernel, float *u, int sgmWidth,
                          int sgmHeight, int views, float totalScanAngle, bool shortScan,
                          float *beta, float *sdd_array, int kernelEnum, float detEltSize,
                          bool curvedDetector, float sdd, float *offcenter_array, float *sgm_flt) {
  // Step 1: weight the sinogram
  dim3 grid((sgmWidth + 15) / 16, (sgmHeight + 15) / 16, batchsize);
  dim3 block(16, 16, 1);

  // Common attenuation imaging
  WeightSinogram_device<<<grid, block>>>(sgm, u, sgmWidth, sgmHeight, views, sdd_array,
                                         totalScanAngle, shortScan, beta, offcenter_array, 
                                         curvedDetector);

  cudaDeviceSynchronize();

  // Step 2: convolve the sinogram
  if (kernelEnum == KERNEL_GAUSSIAN_RAMP) {
    // if Guassian aposied kernel is used, the sinogram need to be filtered twice
    // first by the ramp filter, then by the gaussian filter
    float du = detEltSize;
    float *reconKernel_ramp;
    cudaMalloc(&reconKernel_ramp, (2 * sgmWidth - 1) * sizeof(float));
    InitReconKernel_Hamming<<<(2 * sgmWidth - 1 + 511) / 512, 512>>>(reconKernel_ramp, sgmWidth, du,
                                                                     1, curvedDetector, sdd);
    cudaDeviceSynchronize();

    // intermidiate filtration result is saved in sgm_flt_ramp
    float *sgm_flt_ramp;
    cudaMalloc(&sgm_flt_ramp, sgmWidth * views * batchsize * sizeof(float));
    ConvolveSinogram_device<<<grid, block>>>(sgm_flt_ramp, sgm, reconKernel_ramp, sgmWidth,
                                             sgmHeight, views, u, detEltSize);
    cudaDeviceSynchronize();
    // the height of the filtered sinogram shrinks to number of views, so the convolution parameters
    // need to be adjusted accordingly
    ConvolveSinogram_device<<<grid, block>>>(sgm_flt, sgm_flt_ramp, reconKernel, sgmWidth, views,
                                             views, u, detEltSize);
    cudaDeviceSynchronize();

    cudaFree(reconKernel_ramp);
    cudaFree(sgm_flt_ramp);
  } else if (kernelEnum == KERNEL_NONE) {
    CopySinogram_device<<<grid, block>>>(sgm_flt, sgm, sgmWidth, sgmHeight, views);
    cudaDeviceSynchronize();
  } else {
    ConvolveSinogram_device<<<grid, block>>>(sgm_flt, sgm, reconKernel, sgmWidth, sgmHeight, views,
                                             u, detEltSize);
    cudaDeviceSynchronize();
  }
}

void BackprojectPixelDriven_Agent(float *sgm_flt, int batchsize, float *sdd_array, float *sid_array,
                                  float *offcenter_array, float *u, float *beta, int imgDim,
                                  bool shortScan, int sgmWidth, int views, float imgPixelSize,
                                  float xCenter, float yCenter, bool curvedDetector, 
                                  bool pmatrixFlag, float* pmatrix_array, float pmatrix_eltsize, 
                                  float imgRot, float *img) {
  dim3 grid((imgDim + 15) / 16, (imgDim + 15) / 16, batchsize);
  dim3 block(16, 16, 1);

  if (pmatrixFlag == false)  // if pmatrix is not applied
    BackprojectPixelDriven_device<<<grid, block>>>(sgm_flt, u, beta, shortScan, sgmWidth, views,
                                                   imgDim, sdd_array, sid_array, offcenter_array,
                                                   imgPixelSize, xCenter, yCenter, curvedDetector, img);
  else  // if pmatrix is applied
    BackprojectPixelDriven_pmatrix_device<<<grid, block>>>(sgm_flt, u, beta, pmatrix_array, pmatrix_eltsize, 
                                                           shortScan, sgmWidth, views, imgDim, sdd_array, 
                                                           sid_array, imgPixelSize, xCenter, yCenter, imgRot, img);

  cudaDeviceSynchronize();
}

__global__ void FOVCrop_device(float *img, int imgDim) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int batch = threadIdx.z + blockDim.z * blockIdx.z;

  if (row < imgDim && col < imgDim) {
    float dist2 = powf(imgDim / 2.0f - row, 2) + powf(imgDim / 2.0f - col, 2);
    if (dist2 > powf(imgDim / 2.0f, 2))
      img[batch * imgDim * imgDim + row * imgDim + col] = 0;
  }
}

void FOVCrop_Agent(float *img, int batchsize, int imgDim) {
  dim3 grid((imgDim + 15) / 16, (imgDim + 15) / 16, batchsize);
  dim3 block(16, 16, 1);

  FOVCrop_device<<<grid, block>>>(img, imgDim);

  cudaDeviceSynchronize();
}

void Fbp_FreeMemory_Agent(float *&p) {
  cudaFree(p);
  p = nullptr;
}

/**
 * This is the very main.
 */
void mandoCudaFbp(float *sgm, int batchsize, int sgmHeight, int sgmWidth, int views, int reconKernelEnum, 
                  float reconKernelParam, float totalScanAngle, float detElementSize, float detOffcenter, 
                  float sid, float sdd, int imgDim, float imgPixelSize, float imgRot, 
                  float imgXCenter, float imgYCenter, bool curvedDetector, bool fovCrop,
                  bool pmatrixFlag, float *pmatrix_array, float pmatrix_eltsize, 
                  bool nonuniformSID, float *sid_array, 
                  bool nonuniformSDD, float *sdd_array,
                  bool nonuniformScanAngle, float *scan_angle_array,
                  bool nonuniformOffCenter, float *offcenter_array, 
                  float *img) {
  const unsigned int SgmBytes = batchsize * sgmWidth * sgmHeight * sizeof(float);
  const unsigned int ImgBytes = batchsize * imgDim * imgDim * sizeof(float);
  const unsigned int PmatrixBytes = 12 * views * sizeof(float);

  // Initialize parameters.
  // Pmatrix
  float *pmatrix_array_device = nullptr;
  if (pmatrixFlag == true){
    cudaMalloc(&pmatrix_array_device, PmatrixBytes);
    cudaMemcpy(pmatrix_array_device, pmatrix_array, PmatrixBytes, cudaMemcpyHostToDevice);
  }
  // SID
  float *sidArray = nullptr;
  if (nonuniformSID == true){
    cudaMalloc(&sidArray, views * sizeof(float));
    cudaMemcpy(sidArray, sid_array, views * sizeof(float), cudaMemcpyHostToDevice);
  }
  else
    Fbp_InitializeDistance_Agent(sidArray, sid, views);

  // SDD
  float *sddArray = nullptr;
  if (nonuniformSDD == true){
    cudaMalloc(&sddArray, views * sizeof(float));
    cudaMemcpy(sddArray, sdd_array, views * sizeof(float), cudaMemcpyHostToDevice);
  }
  else
    Fbp_InitializeDistance_Agent(sddArray, sdd, views);

  // Offcenter
  float *offcenterArray = nullptr;
  if (nonuniformOffCenter == true){
    cudaMalloc(&offcenterArray, views * sizeof(float));
    cudaMemcpy(offcenterArray, offcenter_array, views * sizeof(float), cudaMemcpyHostToDevice);
  }
  else
    Fbp_InitializeDistance_Agent(offcenterArray, detOffcenter, views);

  // Scan angle
  float *beta = nullptr;
  if (nonuniformScanAngle == true){
    totalScanAngle = (scan_angle_array[views - 1] - scan_angle_array[0]) / float(views) * float(views + 1);  // degree
    for (int i = 0; i < views; i++)
		  scan_angle_array[i] = (imgRot + scan_angle_array[i]) / 180.0f * PI;
    cudaMalloc(&beta, views * sizeof(float));
    cudaMemcpy(beta, scan_angle_array, views * sizeof(float), cudaMemcpyHostToDevice);
  }
  else
    Fbp_InitializeBeta_Agent(beta, views, imgRot, totalScanAngle);
  
  // Others
  float *u = nullptr;
  Fbp_InitializeU_Agent(u, sgmWidth, detElementSize, detOffcenter);
  bool shortScan = 360.0f > abs(totalScanAngle);
  float *reconKernel = nullptr;
  Fbp_InitializeReconKernel_Agent(reconKernel, sgmWidth, detElementSize, reconKernelEnum, reconKernelParam, 
                                  curvedDetector, sdd);
  // Make sure parameters are correct
  cudaCheckError();

  float *sgm_device = nullptr;
  cudaMalloc(&sgm_device, SgmBytes);
  cudaMemcpy(sgm_device, sgm, SgmBytes, cudaMemcpyHostToDevice);
  float *filtered_sgm = nullptr;
  cudaMalloc(&filtered_sgm, SgmBytes);
  float *img_device = nullptr;
  cudaMalloc(&img_device, ImgBytes);

  // Filter the sinogram.
  FilterSinogram_Agent(sgm_device, batchsize, reconKernel, u, sgmWidth, sgmHeight, views,
                       totalScanAngle, shortScan, beta, sddArray, reconKernelEnum, detElementSize,
                       curvedDetector, sdd, offcenterArray, filtered_sgm);
  cudaCheckError();

  BackprojectPixelDriven_Agent(filtered_sgm, batchsize, sddArray, sidArray, offcenterArray, u, beta,
                               imgDim, shortScan, sgmWidth, views, imgPixelSize, imgXCenter, imgYCenter,
                               curvedDetector, pmatrixFlag, pmatrix_array_device, pmatrix_eltsize, imgRot, img_device);
  cudaCheckError();

  if (fovCrop) {
    FOVCrop_Agent(img_device, batchsize, imgDim);
    cudaCheckError();
  }

  cudaMemcpy(img, img_device, ImgBytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  Fbp_FreeMemory_Agent(filtered_sgm);
  Fbp_FreeMemory_Agent(sgm_device);
  Fbp_FreeMemory_Agent(img_device);
  // Others parameters
  Fbp_FreeMemory_Agent(pmatrix_array_device);
  Fbp_FreeMemory_Agent(sidArray);
  Fbp_FreeMemory_Agent(sddArray);
  Fbp_FreeMemory_Agent(offcenterArray);
  Fbp_FreeMemory_Agent(beta);
  Fbp_FreeMemory_Agent(u);
  Fbp_FreeMemory_Agent(reconKernel);
}