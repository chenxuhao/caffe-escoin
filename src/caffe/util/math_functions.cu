#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/cuda_launch_config.hpp"
#include "caffe/util/cutil_subset.h"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

// cxh
template <>
void caffe_gpu_sparse_csrmm<float>(const int M, const int N, const int K,
    const int nnz, const float alpha, const float* A_nonzero_buf, 
		const int* A_idx_pointer_buf, const int* A_nonzero_idx_buf,
    const float* B, const float beta, float* C, float *transpose_C) {
	CUSPARSE_CHECK(cusparseScsrmm2(Caffe::cusparse_handle(),
		CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE,
		M, N, K, nnz, &alpha, Caffe::cusparse_matdescr(), A_nonzero_buf, 
		A_idx_pointer_buf, A_nonzero_idx_buf, B, N, &beta, transpose_C, M));
	//transpose C
	const float one = 1;
	const float zero = 0;
	CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_T,
			N, M, &one, transpose_C, M, &zero, transpose_C, M, C, N));
}

template <>
void caffe_gpu_sparse_csrmm<double>(const int M, const int N, const int K,
    const int nnz, const double alpha, const double* A_nonzero_buf, 
		const int* A_idx_pointer_buf, const int* A_nonzero_idx_buf,
    const double* B, const double beta, double* C, double *transpose_C) {
	// This function performs sparse matrix (A) dense matrix (B) multiplication
	CUSPARSE_CHECK(cusparseDcsrmm2(Caffe::cusparse_handle(),
		CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE,
		M, N, K, nnz, &alpha, Caffe::cusparse_matdescr(), A_nonzero_buf, 
		A_idx_pointer_buf, A_nonzero_idx_buf, B, N, &beta, transpose_C, M));
	//transpose C
	const double one = 1;
	const double zero = 0;
	CUBLAS_CHECK(cublasDgeam(Caffe::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_T,
			N, M, &one, transpose_C, M, &zero, transpose_C, M, C, N));
}

template <>
void caffe_gpu_sparse_mmcsr<float>(const int M, const int N, const int K,
    const int nnz, const float alpha,
    const float* A_nonzero_buf, const int* A_idx_pointer_buf, const int* A_nonzero_idx_buf,
		const float* B, const float beta, float* C) {
	CUSPARSE_CHECK(cusparseScsrmm(Caffe::cusparse_handle(),
			CUSPARSE_OPERATION_TRANSPOSE, K, M, N, nnz, &alpha,
			Caffe::cusparse_matdescr(), A_nonzero_buf, 
			A_idx_pointer_buf, A_nonzero_idx_buf, B, K, &beta, C, N));
}

template <>
void caffe_gpu_sparse_mmcsr<double>(const int M, const int N, const int K,
    const int nnz, const double alpha,
    const double* A_nonzero_buf, const int* A_idx_pointer_buf, const int* A_nonzero_idx_buf,
    const double* B, const double beta, double* C) {
	CUSPARSE_CHECK(cusparseDcsrmm(Caffe::cusparse_handle(),
			CUSPARSE_OPERATION_TRANSPOSE, K, M, N, nnz, &alpha,
			Caffe::cusparse_matdescr(), A_nonzero_buf, 
			A_idx_pointer_buf, A_nonzero_idx_buf, B, K, &beta, C, N));
}

template <>
void caffe_gpu_sparse_dense2csr<float>(const int M, const int N, const float* A, int* nnzPerRow,
    float* A_nonzero_buf, int* A_idx_pointer_buf, int* A_nonzero_idx_buf, int *nnz_total) {
#if 0
	// This function computes the number of nonzero elements per row or column 
	// and the total number of nonzero elements in a dense matrix.
	CUSPARSE_CHECK(cusparseSnnz(Caffe::cusparse_handle(),
			CUSPARSE_DIRECTION_ROW, M, N,
			Caffe::cusparse_matdescr(), A, M,
			nnzPerRow, nnz_total));

    // This function converts the matrix A in dense format into a sparse matrix in CSR format.
	CUSPARSE_CHECK(cusparseSdense2csr(Caffe::cusparse_handle(),
			M, N, Caffe::cusparse_matdescr(), A, M, nnzPerRow,
			A_nonzero_buf, A_nonzero_idx_buf, A_idx_pointer_buf));
#else
	CUSPARSE_CHECK(cusparseSnnz(Caffe::cusparse_handle(),
			CUSPARSE_DIRECTION_COLUMN, N, M,
			Caffe::cusparse_matdescr(), A, N,
			nnzPerRow, nnz_total));

	CUSPARSE_CHECK(cusparseSdense2csc(Caffe::cusparse_handle(),
			N, M, Caffe::cusparse_matdescr(), A, N, nnzPerRow,
			A_nonzero_buf, A_nonzero_idx_buf, A_idx_pointer_buf));
#endif
}

template <>
void caffe_gpu_sparse_dense2csr<double>(const int M, const int N, const double* A, int* nnzPerRow,
    double* A_nonzero_buf, int* A_idx_pointer_buf, int* A_nonzero_idx_buf,int *nnz_total) {
#if 0
	CUSPARSE_CHECK(cusparseDnnz(Caffe::cusparse_handle(),
			CUSPARSE_DIRECTION_ROW, M, N,
			Caffe::cusparse_matdescr(), A, M,
			nnzPerRow, nnz_total));

	CUSPARSE_CHECK(cusparseDdense2csr(Caffe::cusparse_handle(),
			M, N, Caffe::cusparse_matdescr(), A, M, nnzPerRow,
			A_nonzero_buf, A_nonzero_idx_buf, A_idx_pointer_buf));
#else
	CUSPARSE_CHECK(cusparseDnnz(Caffe::cusparse_handle(),
			CUSPARSE_DIRECTION_COLUMN, N, M,
			Caffe::cusparse_matdescr(), A, N,
			nnzPerRow, nnz_total));

	CUSPARSE_CHECK(cusparseDdense2csc(Caffe::cusparse_handle(),
			N, M, Caffe::cusparse_matdescr(), A, N, nnzPerRow,
			A_nonzero_buf, A_nonzero_idx_buf, A_idx_pointer_buf));
#endif
}

template <typename Dtype>
__global__ void sconv_kernel(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input_padded, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w,
		Dtype *output, int out_channels, const int output_h, const int output_w) {
	int output_row = blockIdx.x * blockDim.x + threadIdx.x;
	int output_col = blockIdx.y * blockDim.y + threadIdx.y;
	int out_channel = blockIdx.z * blockDim.z + threadIdx.z;
	if (output_row < output_h)
		if (output_col < output_w)
			if(out_channel < out_channels) {
			//for (int out_channel = 0; out_channel < out_channels; ++out_channel) {
				Dtype sum = 0;
				for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
					int col = colidx[j];
					int kernel_col = col%(width + pad_w);
					int kernel_row = (col/(width + pad_w))%(height + pad_h);
					int in_channel = col/((width + pad_w)*(height + pad_h));
					int input_row = kernel_row * dilation_h + output_row * stride_h;
					int input_col = kernel_col * dilation_w + output_col * stride_w;
					sum += values[j]*input_padded[(in_channel * (height + pad_h) + input_row) * (width + pad_w) + input_col];
				}
				output[(out_channel*output_h + output_row)*output_w + output_col] = sum;
			}
}

template <typename Dtype>
__global__ void sconv_kernel_base(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input_padded, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int kernel_h, int kernel_w,
		Dtype *output, int out_channels, const int output_h, const int output_w) {
	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int oc = blockIdx.z * blockDim.z + threadIdx.z;
	if (oc < out_channels) {
		if (output_row < output_h) {
			if (output_col < output_w) {
				const Dtype *in_ptr = input_padded + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				Dtype sum = 0;
				for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
					sum += values[j] * in_ptr[colidx[j]];
				}
				output[(oc * output_h + output_row) * output_w + output_col] = sum;
			}
		}
	}
}

#define BLOCK_SIZE 256 // 4*4*32
#define WARP_SIZE 32
#define VECTOR_SIZE 16
#define TILE_SIZE 16
#define OC_BLOCK (BLOCK_SIZE/TILE_SIZE/TILE_SIZE) // OC_BLOCK should be larger than WARP_SIZE when using WARP_KERNEL
//#define OC_BLOCK 32
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
//#define WARP_KERNEL

// the WARPED version uses one thread block to process one output channel 
template <typename Dtype>
__global__ void sconv_kernel_warp(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input_padded, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int kernel_h, int kernel_w,
		Dtype *output, int out_channels, const int output_h, const int output_w) {
	const int gidx = blockIdx.x * blockDim.x + threadIdx.x; // global x index
	const int gidy = blockIdx.y * blockDim.y + threadIdx.y;
	//const int gidz = blockIdx.z * blockDim.z + threadIdx.z;
	const int lidx = threadIdx.x; // local x index 
	const int lidy = threadIdx.y;
	//const int lidz = threadIdx.z;

	__shared__ Dtype sdata[TILE_SIZE][TILE_SIZE*32+16];
	//__shared__ int ptrs[TILE_SIZE][TILE_SIZE][2];
	const int thread_lane = lidx & (WARP_SIZE-1);
	const int warp_id = gidx / WARP_SIZE;
	const int warp_lane = lidx / WARP_SIZE;
	//const int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x * gridDim.y;

	const int output_row = gidy;
	const int output_col = warp_id;
	const int row = lidy;
	//const int col = warp_lane;
	const int oc = blockIdx.z;
	if (oc < out_channels) { 
	//for(int oc = 0; oc < out_channels; oc ++) {
		if (output_row < output_h) {
			if (output_col < output_w) {
				const Dtype *input_ptr = input_padded + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				//if(thread_lane < 2)
				//	ptrs[row][col][thread_lane] = rowptr[oc + thread_lane];
				//__syncthreads();
				//const int row_start = ptrs[row][col][0]; //rowptr[oc]
				//const int row_end   = ptrs[row][col][1]; //rowptr[oc+1]
				const int row_start = rowptr[oc];
				const int row_end = rowptr[oc+1];
				Dtype sum = 0;
				for (int j = row_start + thread_lane; j < row_end; j += WARP_SIZE) {
					sum += values[j] * input_ptr[colidx[j]];
				}

				// reduce local sums to sum (ASSUME: warpsize 32)
				sdata[row][lidx] = sum; __syncthreads();
				sdata[row][lidx] = sum = sum + sdata[row][lidx+16]; __syncthreads(); 
				sdata[row][lidx] = sum = sum + sdata[row][lidx+ 8]; __syncthreads();
				sdata[row][lidx] = sum = sum + sdata[row][lidx+ 4]; __syncthreads();
				sdata[row][lidx] = sum = sum + sdata[row][lidx+ 2]; __syncthreads();
				sdata[row][lidx] = sum = sum + sdata[row][lidx+ 1]; __syncthreads();

				if (thread_lane == 0)
					output[(oc * output_h + output_row) * output_w + output_col] = sdata[row][lidx];
			}
		}
	}
}

template <typename Dtype>
void caffe_gpu_sconv(const Dtype *input_padded,
		const int *rowptr, const int *colidx, const Dtype *values,
		int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w,
		int kernel_h, int kernel_w, Dtype *output, int out_channels) 
{
	//print_device_info(0);
	const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int ntiles_h = (output_h - 1) / TILE_SIZE + 1;
	const int ntiles_w = (output_w - 1) / TILE_SIZE + 1;
	int nblocks = (out_channels - 1) / OC_BLOCK + 1;
#ifdef WARP_KERNEL
	cudaDeviceProp deviceProp;
	CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const int max_blocks_per_SM = maximum_residency(sconv_kernel_warp<Dtype>, BLOCK_SIZE, 0);
	const int max_num_blocks = max_blocks_per_SM * nSM;
	//nblocks = std::min(max_num_blocks, DIVIDE_INTO(out_channels, (BLOCK_SIZE/WARP_SIZE)));
	//printf("nSM=%d, max_blocks_per_SM=%d, max_num_blocks=%d, nblocks=%d\n", 
	//		nSM, max_blocks_per_SM, max_num_blocks, nblocks);
	dim3 threads(WARP_SIZE*TILE_SIZE, TILE_SIZE);
	dim3 grid(ntiles_w, ntiles_h, out_channels);
#else
	dim3 threads(TILE_SIZE, TILE_SIZE, OC_BLOCK);
	dim3 grid(ntiles_w, ntiles_h, nblocks);
#endif
	printf("m=out_channels=%d, height=%d, width=%d, ", out_channels, height, width);
	printf("output_h=%d, output_w=%d, ", output_h, output_w);
	printf("stride_h=%d, stride_w=%d, ", stride_h, stride_w);
	printf("pad_h=%d, pad_width=%d\n", pad_h, pad_w);
	if (dilation_h != 1 || dilation_w != 1) {
		sconv_kernel<Dtype><<<grid, threads>>>(rowptr, colidx, values, input_padded, 
					height, width, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
					kernel_h, kernel_w, output, out_channels, output_h, output_w);
	} else {
#ifdef WARP_KERNEL
		sconv_kernel_warp<Dtype><<<grid, threads>>>(rowptr, colidx, values, input_padded, 
					height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
					output, out_channels, output_h, output_w);
#else
		sconv_kernel_base<Dtype><<<grid, threads>>>(rowptr, colidx, values, input_padded, 
					height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
					output, out_channels, output_h, output_w);
#endif
	}
	CudaTest("sconv_kernel solving failed");
}

template void caffe_gpu_sconv<int>(const int *input_padded, const int *rowptr, const int *colidx, const int *values,
		int height, int width, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, 
		int kernel_h, int kernel_w, int *output, int out_channels);
template void caffe_gpu_sconv<float>(const float *input_padded, const int *rowptr, const int *colidx, const float *values,
		int height, int width, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, 
		int kernel_h, int kernel_w, float *output, int out_channels);
template void caffe_gpu_sconv<double>(const double *input_padded, const int *rowptr, const int *colidx, const double *values,
		int height, int width, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, 
		int kernel_h, int kernel_w, double *output, int out_channels);

__global__ void stretch_kernel(const int *rowptr, int *colidx, int M,
		int height, int width, int pad_h, int pad_w, int kernel_h, int kernel_w) {
	int out_channel = blockIdx.x * blockDim.x + threadIdx.x;
	if(out_channel < M) {
		for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
			int col = colidx[j];
			int kernel_col = col % kernel_w;
			int kernel_row = (col / kernel_w) % kernel_h;
			int in_channel = col / (kernel_w * kernel_h);
			//assert(in_channel < conv_in_channels_);
			colidx[j] = (in_channel * (height + pad_h) + kernel_row) * (width + pad_w) + kernel_col;
		}
	}
}

void caffe_gpu_stretch(const int *rowptr, int *colidx, int M, 
		int height, int width, int pad_h, int pad_w, int kernel_h, int kernel_w) {
	int nthreads = CAFFE_CUDA_NUM_THREADS;
	int nblocks = (M - 1) / nthreads + 1;
	stretch_kernel<<<nblocks, nthreads>>>(rowptr, colidx, M, 
			height, width, pad_h, pad_w, kernel_h, kernel_w);
}

template <typename Dtype>
__global__ void copy_input(Dtype *dst, const Dtype *src, int num_channels, int height, int width, int pad_h, int pad_w) {
	int xid = blockIdx.x * blockDim.x + threadIdx.x;
	int yid = blockIdx.y * blockDim.y + threadIdx.y;
	int zid = blockIdx.z * blockDim.z + threadIdx.z;
	Dtype * dst_ptr = dst + (xid * (height + pad_h) + yid + pad_h) * (width + pad_w) + pad_w;
	const Dtype * src_ptr = src + (xid * height + yid) * width;
	if(xid < num_channels)
		if(yid < height)
			if(zid < width)
				dst_ptr[zid] = src_ptr[zid];
}

template <typename Dtype>
void copy_input_data(Dtype *dst, const Dtype *src, int num_channels, int height, int width, int pad_h, int pad_w) {
	const int TILE_SZ = 4;
	const int BLOCK_SZ = 16;
	dim3 grid((num_channels-1)/BLOCK_SZ+1, (height-1)/TILE_SZ+1, (width-1)/TILE_SZ+1);
	dim3 threads(BLOCK_SZ, TILE_SZ, TILE_SZ);
	copy_input<Dtype><<<grid,threads>>>(dst, src, num_channels, height, width, pad_h, pad_w);
}

template void copy_input_data<float>(float *dst, const float *src, int num_channels, int height, int width, int pad_h, int pad_w);
template void copy_input_data<double>(double *dst, const double *src, int num_channels, int height, int width, int pad_h, int pad_w);
// end of cxh

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
