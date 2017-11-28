#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

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
__global__ void sconv_kernel1(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input_padded, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w,
		Dtype *output, int out_channels, const int output_h, const int output_w) {
	int output_row = blockIdx.x * blockDim.x + threadIdx.x;
	int output_col = blockIdx.y * blockDim.y + threadIdx.y;
	int begin = 0;
	int end = out_channels;
	for (int out_channel = begin; out_channel < end; ++out_channel) {
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
__global__ void sconv_kernel2(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input_padded, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w,
		Dtype *output, int out_channels, const int output_h, const int output_w) {
	int output_row = blockIdx.x * blockDim.x + threadIdx.x;
	int output_col = blockIdx.y * blockDim.y + threadIdx.y;
	int begin = 0;
	int end = out_channels;
	const Dtype *in_temp2 = input_padded + output_row * stride_h * (width + pad_w) + output_col * stride_w;
	for (int out_channel = begin; out_channel < end; ++out_channel) {
		Dtype sum = 0;
		for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
			//assert(in_temp2 + colidx[j] - input_padded < input_padded_len);
			sum += values[j]*in_temp2[colidx[j]];
		}
		output[(out_channel*output_h + output_row)*output_w + output_col] = sum;
	}
}

template <typename Dtype>
void caffe_gpu_sconv(const Dtype *input_padded,
		const int *rowptr, const int *colidx, const Dtype *values,
		int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w,
		int kernel_h, int kernel_w, Dtype *output, int out_channels) 
{
	const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int TILE_SZ = 16;//CAFFE_CUDA_NUM_THREADS;
	dim3 grid( output_h/TILE_SZ, output_w/TILE_SZ ), threads( TILE_SZ, TILE_SZ );
	if (dilation_h != 1 || dilation_w != 1) {
		sconv_kernel1<Dtype><<<grid, threads>>>(
			rowptr, colidx, values, input_padded, height, width, pad_h, pad_w, stride_h, stride_w, 
			dilation_h, dilation_w, kernel_h, kernel_w, output, out_channels, output_h, output_w);
	} else {
		sconv_kernel2<Dtype><<<grid, threads>>>(
			rowptr, colidx, values, input_padded, height, width, pad_h, pad_w, stride_h, stride_w, 
			dilation_h, dilation_w, kernel_h, kernel_w, output, out_channels, output_h, output_w);
	}
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
