#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>
#include <omp.h> // cxh
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/sconv.hpp" // cxh
#include "caffe/util/cpu_info.hpp" // cxh

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

// cxh
template <>
void caffe_cpu_sparse_csrmm<float>(const int M, const int N, const int K,
		const float alpha,
		const float* A_nonzero_buf, const int* A_nonzero_idx_buf, const int* A_idx_pointerB_,const int* A_idx_pointerE_,
		const float* B,
		const float beta,float* C){
#ifdef USE_MKL
	const char *matdescra = "GXXCX";//6 bytes
	const char transa = 'N';
	mkl_scsrmm (&transa, &M , &N, &K,
			&alpha , matdescra,
			A_nonzero_buf, A_nonzero_idx_buf, A_idx_pointerB_, A_idx_pointerE_,
			B, &N,
			&beta , C, &N);
#else
	NOT_IMPLEMENTED;
#endif
}

template <>
void caffe_cpu_sparse_csrmm<double>(const int M, const int N, const int K,
		const double alpha,
		const double* A_nonzero_buf, const int* A_nonzero_idx_buf, const int* A_idx_pointerB_,const int* A_idx_pointerE_,
		const double* B,
		const double beta,double* C){
#ifdef USE_MKL
	char matdescra[6];
	matdescra[0] = 'g';
	matdescra[3] = 'c';
	const char transa = 'N';
	mkl_dcsrmm (&transa, &M , &N, &K,
			&alpha , matdescra,
			A_nonzero_buf, A_nonzero_idx_buf, A_idx_pointerB_, A_idx_pointerE_,
			B, &N,
			&beta , C, &N);
#else
	NOT_IMPLEMENTED;
#endif
}

template <>
void caffe_cpu_sparse_dense2csr<float>(const int M, const int N,
		float* A,
		float* A_nonzero_buf, int* A_nonzero_idx_buf, int* A_idx_pointer_buf){
#ifdef USE_MKL
	MKL_INT info;
	const MKL_INT job[] = {0,0,0,2,M*N,1};
	mkl_sdnscsr(job, &M , &N , A,
			&N , A_nonzero_buf, A_nonzero_idx_buf, A_idx_pointer_buf,  &info);
	if(info){
		LOG(FATAL)<<"The routine is interrupted processing the "<<
			info<<"-th row "
			<<"because there is no space in the arrays acsr and ja according to the value nzmax.";
	}
#else
	int nnz = 0;
	A_idx_pointer_buf[0] = 0;
	for(int i = 0; i < M; i ++) {
		int nnz_per_row = 0;
		for(int j = 0; j < N; j ++) {
			if(A[i*N+j] != 0) {
				A_nonzero_buf[nnz] = A[i*N+j];
				A_nonzero_idx_buf[nnz] = j;
				nnz_per_row ++;
				nnz ++;
			}
		}
		A_idx_pointer_buf[i+1] = A_idx_pointer_buf[i] + nnz_per_row;
	}
#endif
}

template <>
void caffe_cpu_sparse_dense2csr<double>(const int M, const int N,
		double* A,
		double* A_nonzero_buf, int* A_nonzero_idx_buf, int* A_idx_pointer_buf){
#ifdef USE_MKL
	MKL_INT info;
	const MKL_INT job[] = {0,0,0,2,M*N,1};
	mkl_ddnscsr(job, &M , &N , A,
			&N , A_nonzero_buf, A_nonzero_idx_buf, A_idx_pointer_buf,  &info);
	if(info){
		LOG(FATAL)<<"The routine is interrupted processing the "<<
			info<<"-th row "
			<<"because there is no space in the arrays acsr and ja according to the value nzmax.";
	}
#else
	NOT_IMPLEMENTED;
#endif
}

template <typename Dtype>
void caffe_cpu_sconv(const Dtype *input_padded, int in_channels,
		int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w,
		const int *rowptr, const int *colidx, const Dtype *values,
		int kernel_h, int kernel_w,
		const Dtype *bias, Dtype *output, int out_channels,
		int input_padded_len) 
{
	const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	//assert(output_h*output_w == N);
	int begin = 0;
	int end = out_channels;
	if (dilation_h != 1 || dilation_w != 1) {
		for (int output_row = 0; output_row < output_h; ++output_row) {
			for (int output_col = 0; output_col < output_w; ++output_col) {
				for (int out_channel = begin; out_channel < end; ++out_channel) {
					Dtype sum = 0;
					for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
						int col = colidx[j];
						int kernel_col = col%(width + pad_w);
						int kernel_row = (col/(width + pad_w))%(height + pad_h);
						int in_channel = col/((width + pad_w)*(height + pad_h));
						int input_row = kernel_row * dilation_h + output_row * stride_h;
						int input_col = kernel_col * dilation_w + output_col * stride_w;
						sum += values[j] * input_padded[(in_channel * (height + pad_h) + input_row) * (width + pad_w) + input_col];
					}
					output[(out_channel * output_h + output_row) * output_w + output_col] = sum;
				}
			}
		}
	}
	else {
		for (int output_row = 0; output_row < output_h; ++output_row) {
			for (int output_col = 0; output_col < output_w; ++output_col) {
				const Dtype *in_temp2 = input_padded + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				for (int out_channel = begin; out_channel < end; ++out_channel) {
					Dtype sum = 0;
					for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
						assert(in_temp2 + colidx[j] - input_padded < input_padded_len);
						sum += values[j] * in_temp2[colidx[j]];
					}
					output[(out_channel * output_h + output_row) * output_w + output_col] = sum;
				}
			}
		}
	}
}

template void caffe_cpu_sconv<int>(const int *input_padded, int in_channels,
		int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w,
		const int *rowptr, const int *colidx, const int *values,
		int kernel_h, int kernel_w,
		const int*bias, int *output, int out_channels,
		int input_padded_len);

template void caffe_cpu_sconv<float>(const float *input_padded, int in_channels,
		int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w,
		const int *rowptr, const int *colidx, const float *values,
		int kernel_h, int kernel_w,
		const float *bias, float *output, int out_channels,
		int input_padded_len);

template void caffe_cpu_sconv<double>(const double *input_padded, int in_channels,
		int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w,
		const int *rowptr, const int *colidx, const double *values,
		int kernel_h, int kernel_w, const double *bias, 
		double *output, int out_channels, int input_padded_len);

template <typename Dtype, bool FUSE_RELU>
void caffe_cpu_blocked_sconv(const Dtype *input_padded, int in_channels,
		int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w,
		const int *rowptr, const int *colidx, const Dtype *values,
		int kernel_h, int kernel_w, const int **rowptr_blocked, 
		const int **colidx_blocked, const Dtype **values_blocked,
		int ncolblocks, const Dtype *bias, Dtype *output, 
		int out_channels, Dtype *output_scratch, int ninputs) 
{
	//const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	//const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	if (dilation_h != 1 || dilation_w != 1) {
		// Goto default
	} else if (stride_h == 1 && stride_w == 1 && height == width && kernel_h == kernel_w && pad_h == pad_w) {
		int num_oc_blocks = (out_channels + OC_BLOCK - 1)/OC_BLOCK;
		int oc_block_begin, oc_block_end;
#ifdef USE_ICC
		cpu::OpenMpManager::getSimpleGroupedThreadPartition(
				&oc_block_begin, &oc_block_end, num_oc_blocks, ninputs);
#endif
		int oc_begin = std::min(oc_block_begin*OC_BLOCK, out_channels);
		int oc_end = std::min(oc_block_end*OC_BLOCK, out_channels);
		if (kernel_h == 2*pad_h + 1) { // matched padding
			if (kernel_h == 1) {
				// the following sizes are used by GoogLeNet
				if (height == 4) {
					sconv_unit_stride<4, 1, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				else if (height == 7) {
					sconv_unit_stride<7, 1, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				else if (height == 14) {
					sconv_unit_stride<14, 1, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				else if (height == 28) {
					sconv_unit_stride<28, 1, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				else if (height == 56) {
					sconv_unit_stride<56, 1, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
			}
			else if (kernel_h == 3) {
				if (height == 12) {
					// overfeat
					sconv_unit_stride<12, 3, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				else if (height == 13) {
					// alexnet conv3-5
					sconv_unit_stride<13, 3, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				// the following sizes are used by GoogLeNet
				else if (height == 7) {
					sconv_unit_stride<7, 3, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				else if (height == 14) {
					sconv_unit_stride<14, 3, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				else if (height == 28) {
					sconv_unit_stride<28, 3, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				else if (height == 56) {
					sconv_unit_stride<56, 3, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				// OCR public
				else if (height == 3) {
					sconv_unit_stride<3, 3, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
			}
			else if (kernel_h == 5) {
				// AlexNet conv2
				if (height == 27) {
					sconv_unit_stride<27, 5, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				// the following sizes are used by GoogLeNet
				else if (height == 7) {
					sconv_unit_stride<7, 5, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				else if (height == 14) {
					sconv_unit_stride<14, 5, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
				else if (height == 28) {
					sconv_unit_stride<28, 5, FUSE_RELU>(
							input_padded,
							rowptr_blocked, colidx_blocked, values_blocked, ncolblocks,
							bias,
							output, oc_begin, oc_end, output_scratch,
							in_channels, out_channels);
					return;
				}
			}
		}
		else if (0 == pad_h) { // zero padding
		}
/*	} else if (height == 227 && width == 227 && pad_h == 0 && pad_w == 0 && stride_h == 4 && stride_w == 4 && kernel_w == 11 && kernel_h == 11) {
		// conv1 of AlexNet
		assert(!FUSE_RELU);
		int WIDTH = 227;
		int STRIDE = 4;
		int K = 11;
		int WOUT = (WIDTH - K)/STRIDE + 1; // 55
		const int JBLOCK = 128;
		const int HBLOCK = 8;
		const int WBLOCK = 9;
		//__declspec(aligned(64)) float sum[WOUT*WOUT];
		__attribute__ ((aligned(64))) float sum[WOUT*WOUT];
		for (int out_channel = 0; out_channel < out_channels; ++out_channel) {
			int jbegin = rowptr[out_channel];
			int jend = std::min(jbegin + JBLOCK, rowptr[out_channel + 1]);
			for (int hbegin = 0; hbegin < WOUT; hbegin += HBLOCK) {
				int hend = std::min(hbegin + HBLOCK, WOUT);
				for (int wbegin = 0; wbegin < WOUT; wbegin += WBLOCK) {
					int wend = std::min(wbegin + WBLOCK, WOUT);
					for (int k = 0; k < (hend - hbegin) * (wend - wbegin); ++k) {
						sum[k] = 0;
					}
					for (int j = jbegin; j < jend; ++j) {
						float c = values[j];
						int off = colidx[j];
						int k = 0;
						for (int h = hbegin; h < hend; ++h) {
							for (int w = wbegin; w < wend; ++w, ++k) {
								sum[k] += c*input_padded[off + (h*WIDTH + w)*STRIDE];
							}
						}
					}
					int k = 0;
					for (int h = hbegin; h < hend; ++h) {
						for (int w = wbegin; w < wend; ++w, ++k) {
							output[(out_channel*WOUT + h)*WOUT + w] = sum[k];
						}
					}
				}
			}
			jbegin += JBLOCK;
			for ( ; jbegin < rowptr[out_channel + 1]; jbegin += JBLOCK) {
				int jend = std::min(jbegin + JBLOCK, rowptr[out_channel + 1]);
				for (int hbegin = 0; hbegin < WOUT; hbegin += HBLOCK) {
					int hend = std::min(hbegin + HBLOCK, WOUT);
					for (int wbegin = 0; wbegin < WOUT; wbegin += WBLOCK) {
						int wend = std::min(wbegin + WBLOCK, WOUT);
						for (int k = 0; k < (hend - hbegin) * (wend - wbegin); ++k) {
							sum[k] = 0;
						}
						for (int j = jbegin; j < jend; ++j) {
							float c = values[j];
							int off = colidx[j];
							int k = 0;
							for (int h = hbegin; h < hend; ++h) {
								for (int w = wbegin; w < wend; ++w, ++k) {
									sum[k] += c*input_padded[off + (h*WIDTH + w)*STRIDE];
								}
							}
						}
						int k = 0;
						for (int h = hbegin; h < hend; ++h) {
							for (int w = wbegin; w < wend; ++w, ++k) {
								output[(out_channel*WOUT + h)*WOUT + w] += sum[k];
							}
						}
					}
				}
			}
		}
		return;
//*/
	}
	caffe_cpu_sconv_default<FUSE_RELU>(input_padded, in_channels, 
		height, width, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
		rowptr, colidx, values, kernel_h, kernel_w, bias, output, out_channels);
}

template 
void caffe_cpu_blocked_sconv<float,true>(const float *input_padded, 
		int in_channels, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w,
		const int *rowptr, const int *colidx, const float *values,
		int kernel_h, int kernel_w, const int **rowptr_blocked, 
		const int **colidx_blocked, const float **values_blocked,
		int ncolblocks, const float *bias, float *output, 
		int out_channels, float *output_scratch, int input_padded_len);

template 
void caffe_cpu_blocked_sconv<float,false>(const float *input_padded, 
		int in_channels, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w,
		const int *rowptr, const int *colidx, const float *values,
		int kernel_h, int kernel_w, const int **rowptr_blocked, 
		const int **colidx_blocked, const float **values_blocked,
		int ncolblocks, const float *bias, float *output, 
		int out_channels, float *output_scratch, int input_padded_len);

template <>
void caffe_cpu_blocked_sconv<double,true>(const double *input_padded, 
		int in_channels, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w,
		const int *rowptr, const int *colidx, const double *values,
		int kernel_h, int kernel_w, const int **rowptr_blocked, 
		const int **colidx_blocked, const double **values_blocked,
		int ncolblocks, const double *bias, double *output, 
		int out_channels, double *output_scratch, int input_padded_len) {
	NOT_IMPLEMENTED;
}

template <>
void caffe_cpu_blocked_sconv<double,false>(const double *input_padded, 
		int in_channels, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w,
		const int *rowptr, const int *colidx, const double *values,
		int kernel_h, int kernel_w, const int **rowptr_blocked, 
		const int **colidx_blocked, const double **values_blocked,
		int ncolblocks, const double *bias, double *output, 
		int out_channels, double *output_scratch, int input_padded_len) {
	NOT_IMPLEMENTED;
}

// end cxh

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_sqrt<float>(const int n, const float* a, float* y) {
  vsSqrt(n, a, y);
}

template <>
void caffe_sqrt<double>(const int n, const double* a, double* y) {
  vdSqrt(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

}  // namespace caffe
