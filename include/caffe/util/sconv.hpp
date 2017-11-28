#include "intrinsic.hpp"
static const int OC_BLOCK = 16;

static int get_col_major_ic_block(int nnz, int num_out_channels, int num_in_channels) {
	// # of in-channels to have on average 32 non-zeros per out-channel
	double nnz_per_oc_and_ic = (double)nnz/num_out_channels/num_in_channels;
	int ret = std::max(8, 1 << (int)round(log2(std::max(1., 32/nnz_per_oc_and_ic))));
	ret = std::min(num_in_channels, ret);
	while (num_in_channels%ret != 0) {
		++ret;
	}
	return ret;
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
						sum += values[j]*input_padded[(in_channel * (height + pad_h) + input_row) * (width + pad_w) + input_col];
					}
					output[(out_channel*output_h + output_row)*output_w + output_col] = sum;
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
						sum += values[j]*in_temp2[colidx[j]];
					}
					output[(out_channel*output_h + output_row)*output_w + output_col] = sum;
				}
			}
		}
	}
}


template <typename Dtype>
void caffe_cpu_blocked_sconv(const Dtype *input_padded, int in_channels,
		int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int dilation_h, int dilation_w,
		const int *rowptr, const int *colidx, const Dtype *values,
		int kernel_h, int kernel_w, const int **rowptr_blocked, 
		const int **colidx_blocked, const Dtype **values_blocked,
		int ncolblocks,
		const Dtype *bias, Dtype *output, int out_channels,
		Dtype *output_scratch, 
		int input_padded_len) 
{
	//const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	//const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	if (dilation_h != 1 || dilation_w != 1) {
		caffe_cpu_sconv<Dtype>(input_padded, in_channels, height, width, 
				pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
				rowptr, colidx, values, kernel_h, kernel_w, bias,
				output, out_channels, input_padded_len);
	} else {
		caffe_cpu_sconv<Dtype>(input_padded, in_channels, height, width, 
				pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
				rowptr, colidx, values, kernel_h, kernel_w, bias,
				output, out_channels, input_padded_len);
	}
}

