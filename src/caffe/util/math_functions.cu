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
__global__ void sconv_dilation(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input, int height, int width, int pad_h, int pad_w, int stride_h, int stride_w, 
		int dilation_h, int dilation_w, int kernel_h, int kernel_w, const Dtype *bias,
		Dtype *output, int num_oc, const int output_h, const int output_w) {
	int output_row = blockIdx.x * blockDim.x + threadIdx.x;
	int output_col = blockIdx.y * blockDim.y + threadIdx.y;
	int out_channel = blockIdx.z * blockDim.z + threadIdx.z;
	if (output_row < output_h) {
		if (output_col < output_w) {
			if(out_channel < num_oc) {
				Dtype sum = 0;
				for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
					int col = colidx[j];
					int kernel_col = col%(width + pad_w);
					int kernel_row = (col/(width + pad_w))%(height + pad_h);
					int in_channel = col/((width + pad_w)*(height + pad_h));
					int input_row = kernel_row * dilation_h + output_row * stride_h;
					int input_col = kernel_col * dilation_w + output_col * stride_w;
					sum += values[j] * input[(in_channel * (height + pad_h) + input_row) * (width + pad_w) + input_col];
				}
				output[(out_channel*output_h + output_row)*output_w + output_col] = sum;
			}
		}
	}
}

template <typename Dtype>
__global__ void sconv_base(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int kernel_h, int kernel_w, const Dtype *bias,
		Dtype *output, int num_oc, const int output_h, const int output_w) {
	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int oc = blockIdx.z * blockDim.z + threadIdx.z;
	if (oc < num_oc) {
		if (output_row < output_h) {
			if (output_col < output_w) {
				const Dtype *in_ptr = input + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				Dtype sum = 0;
				for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
					sum += values[j] * in_ptr[colidx[j]];
				}
				output[(oc * output_h + output_row) * output_w + output_col] = sum;
			}
		}
	}
}

template <typename Dtype>
__global__ void sconv_relu_base(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int kernel_h, int kernel_w, const Dtype *bias,
		Dtype *output, int num_oc, const int output_h, const int output_w) {
	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int oc = blockIdx.z * blockDim.z + threadIdx.z;
	if (oc < num_oc) {
		if (output_row < output_h) {
			if (output_col < output_w) {
				const Dtype *in_ptr = input + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				Dtype sum = bias[oc];
				for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
					sum += values[j] * in_ptr[colidx[j]];
				}
				output[(oc * output_h + output_row) * output_w + output_col] = max(sum, (Dtype)0);
			}
		}
	}
}

template <typename Dtype>
__global__ void sconv_batch_base(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input, int ifmap_size, int height, int width, int pad_h, int pad_w, 
		int stride_h, int stride_w, int kernel_h, int kernel_w, const Dtype *bias, 
		Dtype *output, int num_oc, const int output_h, const int output_w) {
	const int row = blockIdx.y * blockDim.y + threadIdx.y; // the row id of output channel
	const int col = blockIdx.x * blockDim.x + threadIdx.x; // the column id of output channel
	const int zid = blockIdx.z * blockDim.z + threadIdx.z;
	const int oc_id = zid % num_oc; // the output channel id
	const int fmap_id = zid / num_oc; // the feature map id
	const int ofmap_size = output_h * output_w * num_oc;
	//if (zid < batch_size * num_oc) {
	if (row < output_h) {
		if (col < output_w) {
			const Dtype *ifmap = input + fmap_id * ifmap_size;
			const Dtype *in_ptr = ifmap + row * stride_h * (width + pad_w) + col * stride_w;
			Dtype sum = 0;
			for (int j = rowptr[oc_id]; j < rowptr[oc_id + 1]; ++j) {
				sum += values[j] * in_ptr[colidx[j]];
			}
			output[fmap_id * ofmap_size + (oc_id * output_h + row) * output_w + col] = sum;
		}
	}
}


#define BLOCK_SIZE 256 // 4*4*32
#define WARP_SIZE 32
#define VLEN 32
#define OC_BLOCK 1
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define MIN(x,y) ((x < y)? x : y)
#define SHMEM_SIZE 1024
#define ITER (SHMEM_SIZE/BLOCK_SIZE)
#define REG_BLOCK_SIZE 32 // number of registers per warp
#define REG_BLOCK_H 4
#define REG_BLOCK_W 1
#define COARSEN 4

template <typename Dtype, int TILE_H, int TILE_W>
__global__ void sconv_shm(const int * rowptr, const int * colidx, const Dtype * values, 
		const Dtype * __restrict__ input, const int height, const int width, const int pad_h, const int pad_w, 
		const int stride_h, const int stride_w, const int kernel_h, const int kernel_w, const Dtype *bias,
		Dtype *output, const int num_oc, const int output_h, const int output_w) {
	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int oc_id = blockIdx.z * blockDim.z + threadIdx.z;
	__shared__ Dtype values_s[SHMEM_SIZE];
	__shared__ int colidx_s[SHMEM_SIZE];
	const int tid = threadIdx.y * TILE_W + threadIdx.x;

	for(int oc = oc_id; oc < num_oc; oc += gridDim.z) {
	const int row_start = rowptr[oc];
	const int row_end = rowptr[oc+1];
	const int length = row_end - row_start;
	const int BLK_SIZE = TILE_H * TILE_W;
	Dtype sum = 0;
	//Dtype sum = bias[oc];
	for(int i = 0; i < length; i += SHMEM_SIZE) {
		int base_addr = row_start + i;
		for (int j = 0; j < SHMEM_SIZE; j += BLK_SIZE) {
			int index_s = tid + j;
			int index = base_addr + index_s;
			if(index >= row_end) {
				colidx_s[index_s] = 0;
				values_s[index_s] = 0;
			} else {
				colidx_s[index_s] = colidx[index];
				values_s[index_s] = values[index];
			}
			__syncthreads();
		}

		if (output_row < output_h) {
			if (output_col < output_w) {
				const Dtype *in_ptr = input + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				int end = MIN(SHMEM_SIZE, length - i);
				for (int off = 0; off < end; ++off) {
					Dtype weight = values_s[off];
					int pos = colidx_s[off];
					sum += weight * __ldg(in_ptr+pos);
				}
			}
		}
		__syncthreads();
	}

	//if (oc < num_oc) {
		if (output_row < output_h) {
			if (output_col < output_w) {
				output[(oc * output_h + output_row) * output_w + output_col] = sum;
			}
		}
	}
}

template <typename Dtype, int TILE_H, int TILE_W, int WIDTH, int K, int PAD = (K - 1) / 2>
__global__ void sconv_coarsened(const int * rowptr, const int * colidx, const Dtype * values, 
		const Dtype * __restrict__ input, const int height, const int width, const int pad_h, const int pad_w, 
		const int stride_h, const int stride_w, const int kernel_h, const int kernel_w, const Dtype *bias,
		Dtype *output, const int num_oc, const int output_h, const int output_w) {
	//assert(PAD <= (K - 1) / 2);
	//const int WOUT = WIDTH + 2 * PAD - K + 1;
	//const int ALIGNED_W = (WOUT + 16 - 1) / 16 * 16;
	//const int REG_BLOCK_W = (WOUT + VLEN - 1) / VLEN;
	//assert(REG_BLOCK_W <= REG_BLOCK_SIZE);
	//const int REG_BLOCK_H = 4;//WOUT < REG_BLOCK_SIZE/REG_BLOCK_W ? WOUT : REG_BLOCK_SIZE/REG_BLOCK_W;
	// WIDTH = 13 (AlexNet conv3-5), AVX2 : REG_BLOCK_W = 2, REG_BLOCK_H = 7, ALIGNED_W = 16
	// WIDTH = 56 (GoogLeNet), AVX2 : REG_BLOCK_W = 7, REG_BLOCK_H = 2, ALIGNED_W = 64

	const int xid = blockIdx.x * blockDim.x + threadIdx.x;
	const int yid = blockIdx.y * blockDim.y + threadIdx.y;
	const int zid = blockIdx.z * blockDim.z + threadIdx.z;
	const int output_row = yid  * COARSEN;
	const int output_col = xid;
	const int oc = zid;
	__shared__ Dtype values_s[SHMEM_SIZE];
	__shared__ int colidx_s[SHMEM_SIZE];
	const int tid = threadIdx.y * TILE_W + threadIdx.x;
	const int BLK_SIZE = TILE_H * TILE_W;

	const int row_start = rowptr[oc];
	const int row_end = rowptr[oc+1];
	//int row_start = __ldg(rowptr+oc);
	//int row_end = __ldg(rowptr+oc+1);
	const int length = row_end - row_start;

	Dtype sum[REG_BLOCK_H][REG_BLOCK_W];
	for (int h = 0; h < REG_BLOCK_H; ++h) {
		for (int w = 0; w < REG_BLOCK_W; ++w) {
			//sum[h][w] = bias[oc];
			sum[h][w] = 0;
		}
	}

	for(int i = 0; i < length; i += SHMEM_SIZE) {
		int base_addr = row_start + i;
		for (int j = 0; j < SHMEM_SIZE; j += BLK_SIZE) {
			int index_s = tid + j;
			int index = base_addr + index_s;
			if(index >= row_end) {
				colidx_s[index_s] = 0;
				values_s[index_s] = 0;
			} else {
				colidx_s[index_s] = colidx[index];
				values_s[index_s] = values[index];
			}
			__syncthreads();
		}

		const Dtype *in_ptr = input + output_row * stride_h * (width + pad_w) + output_col * stride_w;
		int end = MIN(SHMEM_SIZE, length - i);
		for (int off = 0; off < end; ++off) {
			Dtype weight = values_s[off];
			int pos = colidx_s[off];
			for (int h = 0; h < REG_BLOCK_H; ++h) {
				for (int w = 0; w < REG_BLOCK_W; ++w) {
					if (output_row + h < output_h) {
						if (output_col + w < output_w) {
							sum[h][w] += weight * __ldg(in_ptr + pos + h * stride_h * (width + pad_w) + w * stride_w);
						}
					}
				}
			}
		}
		__syncthreads();
	}

	for (int h = 0; h < REG_BLOCK_H; ++h) {
		for (int w = 0; w < REG_BLOCK_W; ++w) {
			if (output_row + h < output_h) {
				if (output_col + w < output_w) {
					output[(oc * output_h + (output_row + h)) * output_w + output_col + w] = sum[h][w];
				}
			}
		}
	}
}

template <typename Dtype, int TILE_H, int TILE_W>
__global__ void sconv_relu_tiled(const int * rowptr, const int * colidx, 
		const Dtype * values, const Dtype * __restrict__ input, 
		int height, int width, int pad_h, int pad_w, int stride_h, 
		int stride_w, int kernel_h, int kernel_w, const Dtype *bias,
		Dtype *output, int num_oc, const int output_h, const int output_w) {
	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int oc = blockIdx.z * blockDim.z + threadIdx.z;
	__shared__ Dtype values_s[SHMEM_SIZE];
	__shared__ int colidx_s[SHMEM_SIZE];
	const int tid = threadIdx.y * TILE_W + threadIdx.x;
	const int BLK_SIZE = TILE_H * TILE_W;

	const int row_start = rowptr[oc];
	const int row_end = rowptr[oc+1];
	const int length = row_end - row_start;
	Dtype sum = bias[oc];
	for(int i = 0; i < length; i += SHMEM_SIZE) {
		int base_addr = row_start + i;
		for (int j = 0; j < SHMEM_SIZE; j += BLK_SIZE) {
			int index_s = tid + j;
			int index = base_addr + index_s;
			if(index >= row_end) {
				colidx_s[index_s] = 0;
				values_s[index_s] = 0;
			} else {
				colidx_s[index_s] = colidx[index];
				values_s[index_s] = values[index];
			}
			__syncthreads();
		}

		if (output_row < output_h) {
			if (output_col < output_w) {
				const Dtype *in_ptr = input + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				int end = MIN(SHMEM_SIZE, length - i);
				for (int off = 0; off < end; ++ off) {
					Dtype weight = values_s[off];
					int pos = colidx_s[off];
					sum += weight * __ldg(in_ptr+pos);
				}
			}
		}
		__syncthreads();
	}
	if (oc < num_oc) {
		if (output_row < output_h) {
			if (output_col < output_w) {
				output[(oc * output_h + output_row) * output_w + output_col] = max(sum, (Dtype)0);
			}
		}
	}
}

template <typename Dtype, int FMAP_BLOCK, int TILE_H, int TILE_W, int WIDTH, int K, int PAD = (K - 1) / 2>
__global__ void sconv_batch_tiled1(const int * rowptr, const int * colidx, const Dtype * values, 
		const Dtype * __restrict__ input, const int ifmap_size, const int height, const int width, 
		const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int kernel_h, const int kernel_w,
		const Dtype *bias, Dtype *output, const int num_oc, const int output_h, const int output_w) {
	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int zid = blockIdx.z * blockDim.z + threadIdx.z;
	const int oc = zid % num_oc; // the output channel id
	const int fmap_id = zid / num_oc; // the feature map id
	const int ofmap_size = output_h * output_w * num_oc;
	
	__shared__ Dtype values_s[SHMEM_SIZE];
	__shared__ int colidx_s[SHMEM_SIZE];
	const int tid = threadIdx.y * TILE_W + threadIdx.x;
	const int BLK_SIZE = TILE_H * TILE_W;

	const int row_start = rowptr[oc];
	const int row_end = rowptr[oc+1];
	const int length = row_end - row_start;
	Dtype sum[FMAP_BLOCK];
	for (int i=0; i < FMAP_BLOCK; i++)
		//sum[i] = bias[oc];
		sum[i] = 0;
	for(int i = 0; i < length; i += SHMEM_SIZE) {
		int base_addr = row_start + i;
		for (int j = 0; j < SHMEM_SIZE; j += BLK_SIZE) {
			int index_s = tid + j;
			int index = base_addr + index_s;
			if(index >= row_end) {
				colidx_s[index_s] = 0;
				values_s[index_s] = 0;
			} else {
				colidx_s[index_s] = colidx[index];
				values_s[index_s] = values[index];
			}
			__syncthreads();
		}
		if (output_row < output_h) {
			if (output_col < output_w) {
				const Dtype *ifmap = input + (fmap_id * FMAP_BLOCK) * ifmap_size;
				const Dtype *in_ptr = ifmap + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				int end = MIN(SHMEM_SIZE, length - i);
				for (int offset = 0; offset < end; ++ offset) { 
					Dtype weight = values_s[offset];
					int pos = colidx_s[offset];
					for(int k = 0; k < FMAP_BLOCK; k ++) {
						sum[k] += weight * __ldg(in_ptr + pos + k * ifmap_size);
					}
				}
			}
		}
		__syncthreads();
	}
	for(int k = 0; k < FMAP_BLOCK; k ++) {
		if (oc < num_oc) {
			if (output_row < output_h) {
				if (output_col < output_w) {
						output[(fmap_id * FMAP_BLOCK + k) * ofmap_size + (oc * output_h + output_row) * output_w + output_col] = sum[k];
				}
			}
		}
	}
}

template <typename Dtype, int FMAP_BLOCK, int TILE_H, int TILE_W, int WIDTH, int K, int PAD = (K - 1) / 2>
__global__ void sconv_batch_tiled(const int * rowptr, const int * colidx, const Dtype * values, 
		const Dtype * __restrict__ input, const int ifmap_size, const int height, const int width, 
		const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int kernel_h, const int kernel_w,
		const Dtype *bias, Dtype *output, const int num_oc, const int output_h, const int output_w) {
	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int zid = blockIdx.z * blockDim.z + threadIdx.z;
	const int oc = zid % num_oc; // the output channel id
	const int fmap_id = zid / num_oc; // the feature map id
	const int ofmap_size = output_h * output_w * num_oc;
	
	__shared__ Dtype values_s[SHMEM_SIZE];
	__shared__ int colidx_s[SHMEM_SIZE];
	const int tid = threadIdx.y * TILE_W + threadIdx.x;
	const int BLK_SIZE = TILE_H * TILE_W;

	const int row_start = rowptr[oc];
	const int row_end = rowptr[oc+1];
	const int length = row_end - row_start;
	
	Dtype sum[FMAP_BLOCK];
	for (int i=0; i < FMAP_BLOCK; i++)
		//sum[i] = bias[oc];
		sum[i] = 0;
	for(int i = 0; i < length; i += SHMEM_SIZE) {
		int base_addr = row_start + i;
		for (int j = 0; j < SHMEM_SIZE; j += BLK_SIZE) {
			int index_s = tid + j;
			int index = base_addr + index_s;
			if(index >= row_end) {
				colidx_s[index_s] = 0;
				values_s[index_s] = 0;
			} else {
				colidx_s[index_s] = colidx[index];
				values_s[index_s] = values[index];
			}
			__syncthreads();
		}
		for(int k = 0; k < FMAP_BLOCK; k ++) {
			if (output_row < output_h) {
				if (output_col < output_w) {
					const Dtype *ifmap = input + (fmap_id * FMAP_BLOCK) * ifmap_size;
					const Dtype *in_ptr = ifmap + output_row * stride_h * (width + pad_w) + output_col * stride_w;
					int end = MIN(SHMEM_SIZE, length - i);
					for (int offset = 0; offset < end; ++ offset) { 
						Dtype weight = values_s[offset];
						int pos = colidx_s[offset];
						sum[k] += weight * __ldg(in_ptr + pos + k * ifmap_size);
					}
				}
			}
		}
		__syncthreads();
	}
	for(int k = 0; k < FMAP_BLOCK; k ++) {
		//if (oc < num_oc) {
		if (output_row < output_h) {
			if (output_col < output_w) {
					output[(fmap_id * FMAP_BLOCK + k) * ofmap_size + (oc * output_h + output_row) * output_w + output_col] = sum[k];
			}
		}
	}
}

#define TILED_KERNEL
template <typename Dtype>
void caffe_gpu_sconv(bool FUSE_RELU, int num, const Dtype *input, const int ifmap_size, const int *rowptr, 
	const int *colidx, const Dtype *values, const Dtype *bias, int height, int width, int pad_h, int pad_w, 
	int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w, Dtype *output, int num_oc) 
{
	//print_device_info(0);
	const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

	int TILE_H = 16;
	int TILE_W = 16;
	int ntiles_h = (output_h - 1) / TILE_H + 1;
	int ntiles_w = (output_w - 1) / TILE_W + 1;
	int nblocks = (num_oc - 1) / OC_BLOCK + 1;
	//printf("num=%d, nblocks=%d, num_oc=%d\n", num, nblocks, num_oc);
	//printf("height=%d, width=%d, output_h=%d, output_w=%d\n", height, width, output_h, output_w);
	//printf("stride_h=%d, stride_w=%d, pad_h=%d, pad_width=%d\n", stride_h, stride_w, pad_h, pad_w);
	if (dilation_h != 1 || dilation_w != 1) {
		dim3 threads(TILE_W, TILE_H, OC_BLOCK);
		dim3 grid(ntiles_w, ntiles_h, nblocks);
		sconv_dilation<Dtype><<<grid, threads>>>(rowptr, colidx, values, input, 
			height, width, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
			kernel_h, kernel_w, bias, output, num_oc, output_h, output_w);
	} else if (stride_h == 1 && stride_w == 1 && height == width && kernel_h == kernel_w && pad_h == pad_w) {
		if(FUSE_RELU) {
			dim3 threads(16, 16, OC_BLOCK);
			dim3 grid(ntiles_w, ntiles_h, nblocks);
			sconv_relu_tiled<Dtype,16,16><<<grid, threads>>>(rowptr, colidx, values, input, 
				height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
				bias, output, num_oc, output_h, output_w);
		} else {
			if(num == 1) {
				//if(height == 27) {
				if(0) {
					ntiles_w = DIVIDE_INTO(output_w, 32);
					ntiles_h = DIVIDE_INTO(output_h, 32);
					dim3 grid(ntiles_w, ntiles_h, nblocks);
					//dim3 threads(32, 8, 1);
					//sconv_coarsened<Dtype,8,32,27,1><<<grid, threads>>>(rowptr, colidx, values, input, 
					//	height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
					//	bias, output, num_oc, output_h, output_w);
					dim3 threads(32, 32, 1);
					sconv_shm<Dtype,32,32><<<grid, threads>>>(rowptr, colidx, values, input, 
						height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
						bias, output, num_oc, output_h, output_w);
				//} else(height == 13) {
				} else {
/*
					const int nthreads = 256;
					cudaDeviceProp deviceProp;
					//CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
					const int nSM = 28;//deviceProp.multiProcessorCount;
					//const int max_blocks_per_SM = maximum_residency(sconv_shm<Dtype,16,16>, nthreads, 0);
					const int max_blocks_per_SM = 8;
					const int max_blocks = max_blocks_per_SM * nSM;
					nblocks = std::min(max_blocks, nblocks);
					//printf("Launching CUDA solver: %d CTAs (max %d/SM), %d threads/CTA ...\n", nblocks, max_blocks_per_SM, nthreads);
//*/	
					dim3 threads(TILE_W, TILE_H, 1);
					dim3 grid(ntiles_w, ntiles_h, nblocks);
					sconv_shm<Dtype,16,16><<<grid, threads>>>(rowptr, colidx, values, input, 
						height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
						bias, output, num_oc, output_h, output_w);
				}
			} else {
				dim3 threads(16, 16, 1);
				//if(nblocks >= 128 && nblocks < 224) {
				if(0) {
					nblocks = (num/1) * ((num_oc - 1) / OC_BLOCK + 1);
					dim3 grid(ntiles_w, ntiles_h, nblocks);
					sconv_batch_tiled<Dtype,1,16,16,56,1><<<grid, threads>>>(rowptr, colidx, values, input, 
						ifmap_size, height, width, pad_h, pad_w, stride_h, stride_w, 
						kernel_h, kernel_w, bias, output, num_oc, output_h, output_w);
				} else {	
					nblocks = (num/2) * ((num_oc - 1) / OC_BLOCK + 1);
					dim3 grid(ntiles_w, ntiles_h, nblocks);
					sconv_batch_tiled<Dtype,2,16,16,56,1><<<grid, threads>>>(rowptr, colidx, values, input, 
						ifmap_size, height, width, pad_h, pad_w, stride_h, stride_w, 
						kernel_h, kernel_w, bias, output, num_oc, output_h, output_w);
				}
			}
		}
	} else {
		// fall through to the default path
		dim3 threads(TILE_W, TILE_H, OC_BLOCK);
		dim3 grid(ntiles_w, ntiles_h, nblocks);
		if(FUSE_RELU) {
			sconv_relu_base<Dtype><<<grid, threads>>>(rowptr, colidx, values, input, 
				height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
				bias, output, num_oc, output_h, output_w);
		} else {
			if(num == 1)
				sconv_base<Dtype><<<grid, threads>>>(rowptr, colidx, values, input, 
					height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
					bias, output, num_oc, output_h, output_w);
			else {
				nblocks = num * ((num_oc - 1) / OC_BLOCK + 1);
				sconv_batch_base<Dtype><<<grid, threads>>>(rowptr, colidx, values, input, 
					ifmap_size, height, width, pad_h, pad_w, stride_h, stride_w, 
					kernel_h, kernel_w, bias, output, num_oc, output_h, output_w);
			}
		}
	}
	CudaTest("sconv_kernel solving failed");
}

template void caffe_gpu_sconv<int>(bool FUSE_RELU, int num, const int *input, const int ifmap_size, const int *rowptr, 
		const int *colidx, const int *values, const int *bias, int height, int width, int pad_h, int pad_w, 
		int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w, int *output, int num_oc);
template void caffe_gpu_sconv<float>(bool FUSE_RELU, int num, const float *input, const int ifmap_size, const int *rowptr, 
		const int *colidx, const float *values, const float *bias, int height, int width, int pad_h, int pad_w, 
		int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w, float *output, int num_oc);
template void caffe_gpu_sconv<double>(bool FUSE_RELU, int num, const double *input, const int ifmap_size, const int *rowptr, 
		const int *colidx, const double *values, const double *bias, int height, int width, int pad_h, int pad_w, 
		int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w, double *output, int num_oc);

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
#if 0
	Dtype * dst_ptr = dst + (xid * (height + pad_h) + yid + pad_h) * (width + pad_w) + pad_w;
	const Dtype * src_ptr = src + (xid * height + yid) * width;
	if(xid < num_channels)
		if(yid < height)
			if(zid < width)
				dst_ptr[zid] = src_ptr[zid];
#else
	Dtype * dst_ptr = dst + (zid * (height + pad_h) + yid + pad_h) * (width + pad_w) + pad_w;
	const Dtype * src_ptr = src + (zid * height + yid) * width;
	if(zid < num_channels)
		if(yid < height)
			if(xid < width)
				dst_ptr[xid] = src_ptr[xid];
#endif
}

template <typename Dtype>
void copy_input_data(Dtype *dst, const Dtype *src, int num_channels, int height, int width, int pad_h, int pad_w) {
	const int TILE_SZ = 16;
	const int BLOCK_SZ = 1;
	const int ntiles_h = (height - 1) / TILE_SZ + 1;
	const int ntiles_w = (width - 1) / TILE_SZ + 1;
	const int nblocks = (num_channels - 1) / BLOCK_SZ + 1;
	//dim3 grid(nblocks, ntiles_h, ntiles_w);
	//dim3 threads(BLOCK_SZ, TILE_SZ, TILE_SZ);
	dim3 grid(ntiles_h, ntiles_w, nblocks);
	dim3 threads(TILE_SZ, TILE_SZ, BLOCK_SZ);
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
