#include <algorithm>
#include <vector>
#include <omp.h> // cxh
#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/sconv.hpp" // cxh
#ifdef USE_ICC
#include "caffe/util/cpu_info.hpp" // cxh
#define BLOCKED_SCONV // cxh: only work with ICC
#endif
namespace caffe {
// cxh
template <typename Dtype>
BaseConvolutionLayer<Dtype>::~BaseConvolutionLayer() {
	if(Caffe::conv_mode() == Caffe::SCONV || Caffe::conv_mode() == Caffe::SCONV_PAR) {
		switch (Caffe::mode()) {
			case Caffe::CPU: {
				free(input_padded_);
				break; }
			case Caffe::GPU: {
#ifndef CPU_ONLY
				CUDA_CHECK(cudaFree(d_input_padded_));
#endif
				break; }
		}
	}
#ifdef BLOCKED_SCONV
	free(output_scratch_);
	for (int i = 0; i < weight_rowptr_.size(); ++i) {
		free(weight_rowptr_[i]);
		free(weight_colidx_[i]);
		free(weight_values_[i]);
	}
	for (int i = 0; i < weight_rowptr_blocked_.size(); ++i) {
		free(weight_rowptr_blocked_[i]);
		free(weight_colidx_blocked_[i]);
		free(weight_values_blocked_[i]);
	}
#endif
}

// cxh
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::WeightAlign() {
#ifdef BLOCKED_SCONV
	cpu::OpenMpManager::getThreadGroupBarriers(this->num_);
	int num_threads = 1;
	//omp_set_num_threads(4);
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	//printf("num_threads = %d\n", num_threads);
#endif
	CHECK_EQ(this->blobs_[0]->num_axes(),4);//caffe now supports any dimension
	//const LayerParameter& layerparam = this->layer_param();
	//LOG(INFO)<<"layer\t"<<layerparam.name()<<"\t"<<"has sparsity of "<< this->blobs_[0]->GetSparsity();
	const int M = this->blobs_[0]->shape(0)/group_;
	const int N = this->blobs_[0]->count(1,4);
	const int weight_offset = this->blobs_[0]->count()/group_;
	const int row_offset = this->blobs_[0]->shape(0)/group_ + 1;

	int height = 0, width = 0, pad_h = 0, pad_w = 0, input_len = 0, msg = 0;
	if(Caffe::conv_mode() == Caffe::SCONV || Caffe::conv_mode() == Caffe::SCONV_PAR) {
		height = conv_input_shape_.cpu_data()[1];
		width = conv_input_shape_.cpu_data()[2];
		pad_h = pad_.cpu_data()[0];
		pad_w = pad_.cpu_data()[1];
		input_len = conv_in_channels_ * (height + pad_h) * (width + pad_w) + pad_h * (width + 2 * pad_w);
#ifdef BLOCKED_SCONV
		input_len += (VLEN - 1);
		msg = posix_memalign((void **)&input_padded_, 4096, sizeof(Dtype) * num_threads * input_len);
		memset(input_padded_, 0, sizeof(Dtype) * num_threads * input_len);
#else
		msg = posix_memalign((void **)&input_padded_, 4096, sizeof(Dtype) * input_len);
		memset(input_padded_, 0, sizeof(Dtype) * input_len);
#endif
	}
	for (int g = 0; g < group_; ++g) {
		switch (Caffe::mode()) {
			case Caffe::CPU: {
				// cxh: create a CSR matrix
				caffe_cpu_sparse_dense2csr(M, N, 
					this->blobs_[0]->mutable_cpu_data() + weight_offset * g,
					nz_weight_values_.mutable_cpu_data() + weight_offset * g,
					nz_weight_indices_.mutable_cpu_data() + weight_offset * g,
					nz_weight_index_pointers_.mutable_cpu_data() + row_offset * g);
				if(Caffe::conv_mode() == Caffe::SCONV || Caffe::conv_mode() == Caffe::SCONV_PAR) {
					// direct sparse convolution
					int kernel_h = kernel_shape_.cpu_data()[0];
					int kernel_w = kernel_shape_.cpu_data()[1];
#ifndef BLOCKED_SCONV
					// transform the indices for direct convolution
					const int *rowptr = nz_weight_index_pointers_.cpu_data() + row_offset * g;
					int *colidx = nz_weight_indices_.mutable_cpu_data() + weight_offset * g;
					for (int out_channel = 0; out_channel < M; ++out_channel) {
						for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
							int col = colidx[j];
							int kernel_col = col%kernel_w;
							int kernel_row = (col/kernel_w)%kernel_h;
							int in_channel = col/(kernel_w*kernel_h);
							assert(in_channel < conv_in_channels_);
							colidx[j] = (in_channel*(height + pad_h) + kernel_row)*(width + pad_w) + kernel_col;
						}
					}
#else
					int temp_nnz = 0;
					for (int g = 0; g < group_; ++g) {
						for (int i = 0; i < M*N; ++i) {
							if (this->blobs_[0]->cpu_data()[weight_offset*g + i] != 0) ++temp_nnz;
						}
					}
					int col_block_size = get_col_major_ic_block(temp_nnz/group_, M, conv_in_channels_/group_);
					assert(conv_in_channels_/group_%col_block_size == 0);
					int ncolblocks = conv_in_channels_/col_block_size;
					assert(ncolblocks >= 1);
					//LOG(INFO) << "ncolblocks " << ncolblocks;
					weight_rowptr_blocked_.resize(ncolblocks);
					weight_colidx_blocked_.resize(ncolblocks);
					weight_values_blocked_.resize(ncolblocks);
					std::vector<int> nnzs_of_col_blocks(ncolblocks, 0);
					weight_rowptr_.resize(group_);
					weight_colidx_.resize(group_);
					weight_values_.resize(group_);

					for (int g = 0; g < group_; ++g) {
						int nnz = 0;
						for (int i = 0; i < M*N; ++i) {
							if (this->blobs_[0]->cpu_data()[weight_offset*g + i] != 0) ++nnz;
						}
						msg = posix_memalign((void **)&weight_rowptr_[g], 4096, sizeof(int)*(M + 1));
						msg = posix_memalign((void **)&weight_colidx_[g], 4096, sizeof(int)*nnz);
						msg = posix_memalign((void **)&weight_values_[g], 4096, sizeof(Dtype)*nnz);
						// first create a CSR matrix as for LOWERED_CSRMM
						caffe_cpu_sparse_dense2csr(M, N,
								this->blobs_[0]->mutable_cpu_data() + weight_offset * g,
								weight_values_[g],
								weight_colidx_[g],
								weight_rowptr_[g]);
						// declare variables for sparsity statistics
						vector<vector<int> > nnz_per_channel_pair(M);
						for(int i = 0; i < M; ++i) {
							nnz_per_channel_pair[i] = vector<int>(conv_in_channels_, 0);
						}
						vector<int> nnz_per_oc_fiber(N, 0);
						assert(N == conv_in_channels_/group_*kernel_h*kernel_w);
						int num_of_non_zero_kernels = 0;
						int num_of_non_zero_out_channels = 0;
						const int *rowptr = weight_rowptr_[g];
						assert(nnz == rowptr[M]);
						//int col_major_ic_block = get_col_major_ic_block(nnz, M, conv_in_channels_/group_);
						//LOG(INFO) << "col_major_ic_block = " << col_major_ic_block;	
						//assert(conv_in_channels_/group_%col_major_ic_block == 0);
						// transform the indices for direct convolution
						int *colidx = weight_colidx_[g];
						for (int oc = 0; oc < M; ++oc) {
							for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
								int col = colidx[j];
								int kernel_col = col%kernel_w;
								int kernel_row = (col/kernel_w)%kernel_h;
								int ic = col/(kernel_w*kernel_h);
								assert(ic < conv_in_channels_/group_);
								colidx[j] = (ic*(height + pad_h) + kernel_row)*(width + pad_w) + kernel_col;
								int bcol = ic/col_block_size + ncolblocks/group_*g;
								++nnzs_of_col_blocks[bcol];
								++nnz_per_channel_pair[oc][ic];
								++nnz_per_oc_fiber[col];
							}
							if (rowptr[oc + 1] > rowptr[oc]) {
								num_of_non_zero_out_channels++;
							}
							for (int in_channel = 0; in_channel < conv_in_channels_; ++in_channel) {
								if (nnz_per_channel_pair[oc][in_channel] != 0) {
									++num_of_non_zero_kernels;
								}
							}
						}
						int num_of_non_zero_oc_fibers = 0;
						for (int i = 0 ; i < N; ++i) {
							if (nnz_per_oc_fiber[i] > 0) ++num_of_non_zero_oc_fibers;
						}
						std::vector<int> kernel_non_zero_hist(kernel_w*kernel_h, 0);
						for (int in_channel = 0; in_channel < conv_in_channels_/group_; ++in_channel) {
							for (int i = in_channel*kernel_w*kernel_h; i < (in_channel + 1)*kernel_w*kernel_h; ++i) {
								kernel_non_zero_hist[i - in_channel*kernel_w*kernel_h] += nnz_per_oc_fiber[i];
							}
						}
					}

					for (int i = 0; i < ncolblocks; ++i) {
						msg = posix_memalign((void **)&weight_rowptr_blocked_[i], 4096, sizeof(int)*(M + 1));
						msg = posix_memalign((void **)&weight_colidx_blocked_[i], 4096, sizeof(int)*nnzs_of_col_blocks[i]);
						msg = posix_memalign((void **)&weight_values_blocked_[i], 4096, sizeof(Dtype)*nnzs_of_col_blocks[i]);
						nnzs_of_col_blocks[i] = 0;
						weight_rowptr_blocked_[i][0] = 0;
					}

					int stride_h = stride_.cpu_data()[0];
					int stride_w = stride_.cpu_data()[1];
					int dilation_h = dilation_.cpu_data()[0];
					int dilation_w = dilation_.cpu_data()[1];
					const int output_h = (height + 2 * pad_h -
							(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
					const int output_w = (width + 2 * pad_w -
							(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
					//const int SCRATCH_SIZE_PER_IC = output_h*((output_w + 16 - 1)/16*16);
					int max_col_major_ic_block = 0;
					for (int g = 0; g < group_; ++g) {
						const int *rowptr = weight_rowptr_[g];
						int *colidx = weight_colidx_[g];
						Dtype *values = weight_values_[g];
						int nnz = rowptr[M];
						int col_major_ic_block = get_col_major_ic_block(nnz, M, conv_in_channels_/group_);
						max_col_major_ic_block = std::max(max_col_major_ic_block, col_major_ic_block);
						for (int oc = 0; oc < M; ++oc) {
							for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
								int c = colidx[j];
								int ic = c/(width + pad_w)/(height + pad_h);
								int bcol = ic/col_block_size + ncolblocks/group_*g;
								weight_colidx_blocked_[bcol][nnzs_of_col_blocks[bcol]] = c;
								weight_values_blocked_[bcol][nnzs_of_col_blocks[bcol]] = values[j];
								nnzs_of_col_blocks[bcol]++;
							}
							for (int i = ncolblocks/group_*g; i < ncolblocks/group_*(g + 1); ++i) {
								weight_rowptr_blocked_[i][oc + 1] = nnzs_of_col_blocks[i];
							}
						}
					} // for each group
					msg = posix_memalign((void **)&output_scratch_, 4096, sizeof(Dtype)*OC_BLOCK*output_h*((output_w + 16 - 1)/16*16)*num_threads);
#endif // end BLOCKED_SCONV
					if(msg) printf("ERROR: memalign failed\n");
				}
				break; }
			case Caffe::GPU: {
#ifndef CPU_ONLY
				//printf("transform weight matrix from dense to sparse\n");
				int total_nonzero = 0;
				caffe_gpu_sparse_dense2csr(M, N,
					this->blobs_[0]->gpu_data() + weight_offset * g,
					nz_per_row_.mutable_gpu_data() + M * g,
					nz_weight_values_.mutable_gpu_data()+ weight_offset * g,
					nz_weight_index_pointers_.mutable_gpu_data() + row_offset * g,
					nz_weight_indices_.mutable_gpu_data()+ weight_offset * g,
					&total_nonzero);
				nz_num_[g] = total_nonzero;
				if(Caffe::conv_mode() == Caffe::SCONV || Caffe::conv_mode() == Caffe::SCONV_PAR) {
					int height = conv_input_shape_.cpu_data()[1];
					int width = conv_input_shape_.cpu_data()[2];
					int pad_h = pad_.cpu_data()[0];
					int pad_w = pad_.cpu_data()[1];
					int kernel_h = kernel_shape_.cpu_data()[0];
					int kernel_w = kernel_shape_.cpu_data()[1];
					int num_of_ifmaps = 1;
					if(Caffe::conv_mode() == Caffe::SCONV_PAR) num_of_ifmaps = num_;
					int length = num_of_ifmaps * (conv_in_channels_ * (height + pad_h) * (width + pad_w) + pad_h * (width + 2 * pad_w));
					CUDA_CHECK(cudaMalloc((void **)&d_input_padded_, sizeof(Dtype) * length));
					CUDA_CHECK(cudaMemset(d_input_padded_, 0, length * sizeof(Dtype)));
					// transform the indices for direct sparse convolution
					const int *rowptr = nz_weight_index_pointers_.gpu_data() + row_offset * g;
					int *colidx = nz_weight_indices_.mutable_gpu_data() + weight_offset * g;
					caffe_gpu_stretch(rowptr, colidx, M, height, width, pad_h, pad_w, kernel_h, kernel_w);
				}
#else
				printf("WARNNING: CPU_ONLY mode is selected\n");
#endif
				break; }
			default:
				printf("Oops, something wrong\n");
		}
	}
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);

  // cxh
  if(!reverse_dimensions()) {
//#ifndef BLOCKED_SCONV
	  nz_weight_values_.Reshape(1, 1, 1, this->blobs_[0]->count());//nonzero elements
	  nz_weight_indices_.Reshape(1,1,1,nz_weight_values_.count());//index of nonzero
	  nz_weight_index_pointers_.Reshape(1,1,1,this->blobs_[0]->shape(0)+group_);//pointer(index) of indices
	  nz_per_row_.Reshape(1,1,1,this->blobs_[0]->shape(0));
	  nz_num_.resize(group_);
//#endif
  }
  transposed_output_buffer_.Reshape(1,1,conv_out_spatial_dim_,conv_out_channels_/group_);

  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
		const Dtype* weights, Dtype* output, bool skip_im2col) {
	const Dtype* col_buff = input;
	if (!is_1x1_) {
		if (!skip_im2col) {
			conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
		}
		col_buff = col_buffer_.cpu_data();
	}

	for (int g = 0; g < group_; ++g) {
		const int M = conv_out_channels_ / group_;
		const int row_offset = conv_out_channels_ / group_ + 1;
		const int *row_offsets = nz_weight_index_pointers_.cpu_data() + row_offset * g;
		Dtype sparsity = (Dtype)1.0 - (Dtype)row_offsets[M] / (Dtype)(conv_out_channels_ / group_ * kernel_dim_);
		if(Caffe::conv_mode() == Caffe::LOWERED_SPARSE && sparsity > 0.5) {
			const int N = conv_out_spatial_dim_;
			const int K = kernel_dim_;
			caffe_cpu_sparse_csrmm(M, N, K, (Dtype)1.,
				nz_weight_values_.cpu_data() + weight_offset_ * g,
				nz_weight_indices_.cpu_data() + weight_offset_ * g,
				nz_weight_index_pointers_.cpu_data() + row_offset * g,
				nz_weight_index_pointers_.cpu_data() + row_offset * g + 1,
				col_buff + col_offset_ * g,
				(Dtype)0., output + output_offset_ * g);
		//} else if(Caffe::conv_mode() == Caffe::LOWERED_GEMM) {
		} else {
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
				group_, conv_out_spatial_dim_, kernel_dim_,
				(Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
				(Dtype)0., output + output_offset_ * g);
		}
	}
}


template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_sconv(const Dtype* input,
		const Dtype* weights, Dtype* output) {
	const int M = conv_out_channels_ / group_;
	const int *row_offsets = nz_weight_index_pointers_.cpu_data();
	if((Dtype)row_offsets[M] / (Dtype)(conv_out_channels_ / group_ * kernel_dim_) > 0.5) {
		forward_cpu_gemm(input, weights, output);
		return;
	}

	int height = 0, width = 0, kernel_h = 0, kernel_w = 0, pad_h = 0, pad_w = 0;
	int stride_h = 0, stride_w = 0, dilation_h = 0, dilation_w = 0;
	int input_padded_len = 0, cbegin = 0, cend = 0;
	Dtype *input_padded = NULL;
#ifdef BLOCKED_SCONV
	int tid = 0, gid = 0;
#endif
	height = conv_input_shape_.cpu_data()[1];
	width = conv_input_shape_.cpu_data()[2];
	kernel_h = kernel_shape_.cpu_data()[0];
	kernel_w = kernel_shape_.cpu_data()[1];
	pad_h = pad_.cpu_data()[0];
	pad_w = pad_.cpu_data()[1];
	stride_h = stride_.cpu_data()[0];
	stride_w = stride_.cpu_data()[1];
	dilation_h = dilation_.cpu_data()[0];
	dilation_w = dilation_.cpu_data()[1];
	input_padded_len = conv_in_channels_ * (height + pad_h) * (width + pad_w) + pad_h * (width + 2 * pad_w);
#ifdef BLOCKED_SCONV
	tid = omp_get_thread_num();
	input_padded_len += (VLEN - 1);
#endif
	if(pad_h == 0 && pad_w == 0)
		input_padded = (Dtype *)input;
	else {
#ifdef BLOCKED_SCONV
		gid = cpu::OpenMpManager::getThreadGroupNum(num_);
		input_padded = input_padded_ + input_padded_len * gid;
		cpu::OpenMpManager::getSimpleGroupedThreadPartition(
				&cbegin, &cend, conv_in_channels_, num_);
		//printf("tid=%d, gid=%d, bid=%d, cbegin=%d, cend=%d, num_=%d, conv_in_channels_=%d\n", 
		//		tid, gid, cbegin, cend, num_, conv_in_channels_);
#else
		cbegin = 0; cend = conv_in_channels_;
		input_padded = input_padded_;
#endif
		for (int in_channel = cbegin; in_channel < cend; ++in_channel) {
			for (int input_row = 0; input_row < height; ++input_row) {
				memcpy(input_padded + (in_channel * (height + pad_h) + input_row + pad_h) * (width + pad_w) + pad_w,
						input + (in_channel * height + input_row) * width, sizeof(Dtype) * width);
			}
		}
#ifdef BLOCKED_SCONV
		cpu::OpenMpManager::barrierGroup(num_);
#endif
	}

	// start computation
	for (int g = 0; g < group_; ++g) {
		const int M = conv_out_channels_ / group_;
		const int row_offset = conv_out_channels_ /group_ + 1;
		const int *row_offsets = nz_weight_index_pointers_.cpu_data() + row_offset * g;
		Dtype sparsity = (Dtype)1.0 - (Dtype)row_offsets[M]/ (Dtype)(conv_out_channels_ / group_ * kernel_dim_);
		LOG(INFO)<<"Sparsity of "<< Layer<Dtype>::layer_param().name() << ": "<< sparsity;
		const Dtype *in_temp = input_padded + conv_in_channels_/group_ * g * (height + pad_h) * (width + pad_w);
#ifdef BLOCKED_SCONV
		const int output_h = this->output_shape_[0];
		const int output_w = this->output_shape_[1];
		assert(output_h*output_w == conv_out_spatial_dim_);
		int ncolblock = weight_rowptr_blocked_.size()/group_;
		const int *rowptr = weight_rowptr_[g];
		const Dtype *values = weight_values_[g];
		const int *colidx = weight_colidx_[g];
		caffe_cpu_blocked_sconv<Dtype,false>(in_temp, conv_in_channels_/group_, 
			height, width, pad_h, pad_w, stride_h, stride_w, dilation_h, 
			dilation_w, rowptr, colidx, values, kernel_h, kernel_w,
			(const int **)(&weight_rowptr_blocked_[0] + g*ncolblock),
			(const int **)(&weight_colidx_blocked_[0] + g*ncolblock),
			(const Dtype **)(&weight_values_blocked_[0] + g*ncolblock),
			ncolblock, this->blobs_[1]->cpu_data(), output + output_offset_ * g, M,
			output_scratch_ + tid*OC_BLOCK*output_h*((output_w + 16 - 1)/16*16), num_);
#else
		const int *rowptr = nz_weight_index_pointers_.cpu_data() + row_offset * g;
		const Dtype *values = nz_weight_values_.cpu_data()+ weight_offset_ * g;
		const int *colidx = nz_weight_indices_.cpu_data()+ weight_offset_ * g;
		caffe_cpu_sconv<Dtype>(in_temp, conv_in_channels_/group_, height, width, 
			pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
			rowptr, colidx, values, kernel_h, kernel_w,
			this->blobs_[1]->cpu_data(),
			output + output_offset_ * g, M, input_padded_len);
#endif
	}
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
		const Dtype* weights, Dtype* output, bool skip_im2col) {
	const Dtype* col_buff = input;
	if (!is_1x1_) {
		if (!skip_im2col) {
			conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
		}
		col_buff = col_buffer_.gpu_data();
	} 

	// start computation
	for (int g = 0; g < group_; ++g) {
		Dtype sparsity = (Dtype)1.0 - (Dtype)nz_num_[g] / (Dtype)(conv_out_channels_ / group_ * kernel_dim_);
		if(Caffe::conv_mode() == Caffe::LOWERED_SPARSE && sparsity > 0.5) {
			//printf("sparse weight matrix multi. dense feature map matrix, sparsity=%f\n", sparsity);
			caffe_gpu_sparse_csrmm(conv_out_channels_ /group_,
				conv_out_spatial_dim_, kernel_dim_, nz_num_[g], (Dtype)1., 
				nz_weight_values_.gpu_data()+ weight_offset_ * g,
				nz_weight_index_pointers_.gpu_data() + (conv_out_channels_ / group_ + 1) * g,
				nz_weight_indices_.gpu_data()+ weight_offset_ * g,
				col_buff + col_offset_ * g, (Dtype)0., 
				output + output_offset_ * g,
				transposed_output_buffer_.mutable_gpu_data());
		//} else if(Caffe::conv_mode() == Caffe::LOWERED_GEMM) {
		} else {
			//printf("dense weight matrix multi. dense feature map matrix\n");
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
					group_, conv_out_spatial_dim_, kernel_dim_,
					(Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
					(Dtype)0., output + output_offset_ * g);
		}
	}
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_sconv(const Dtype* input, const Dtype* weights, Dtype* output) {
	if((Dtype)nz_num_[0] / (Dtype)(conv_out_channels_ / group_ * kernel_dim_) > 0.5) {
		forward_gpu_gemm(input, weights, output);
		return;
	}
	int height = 0, width = 0, kernel_h = 0, kernel_w = 0, pad_h = 0, pad_w = 0;
	int stride_h = 0, stride_w = 0, dilation_h = 0, dilation_w = 0;
	Dtype *d_input_padded = NULL;
	height = conv_input_shape_.cpu_data()[1];
	width = conv_input_shape_.cpu_data()[2];
	kernel_h = kernel_shape_.cpu_data()[0];
	kernel_w = kernel_shape_.cpu_data()[1];
	pad_h = pad_.cpu_data()[0];
	pad_w = pad_.cpu_data()[1];
	stride_h = stride_.cpu_data()[0];
	stride_w = stride_.cpu_data()[1];
	dilation_h = dilation_.cpu_data()[0];
	dilation_w = dilation_.cpu_data()[1];
	if(pad_h == 0 && pad_w == 0)
		d_input_padded = (Dtype *)input;
	else {
		d_input_padded = d_input_padded_;
		copy_input_data<Dtype>(d_input_padded, input, conv_in_channels_, height, width, pad_h, pad_w);
	}

	// start computation
	for (int g = 0; g < group_; ++g) {
		const Dtype *in_temp = d_input_padded + conv_in_channels_ / group_ * g * (height + pad_h) * (width + pad_w);
		const int row_offset = conv_out_channels_ / group_ + 1;
		const int M = conv_out_channels_ / group_;
		const int *rowptr = nz_weight_index_pointers_.gpu_data() + row_offset * g;
		const Dtype *values = nz_weight_values_.gpu_data()+ weight_offset_ * g;
		const int *colidx = nz_weight_indices_.gpu_data()+ weight_offset_ * g;
		if (std::string(this->type()) == "ConvolutionReLU") {
			const Dtype *bias = NULL;
			if (bias_term_)
				bias = this->blobs_[1]->gpu_data();
			assert(bias != NULL);
			caffe_gpu_sconv<Dtype>(true, in_temp, rowptr, colidx, values, bias,
				height, width, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
				kernel_h, kernel_w, output + output_offset_ * g, M);
			//} else if (std::string(this->type()) == "Convolution") {
		} else {
			caffe_gpu_sconv<Dtype>(false, in_temp, rowptr, colidx, values, NULL,
				height, width, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
				kernel_h, kernel_w, output + output_offset_ * g, M);
		}
	}
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_sconv_par(const Dtype* input, const Dtype* weights, Dtype* output) {
	//if((Dtype)nz_num_[0] / (Dtype)(conv_out_channels_ / group_ * kernel_dim_) > 0.5) {
	if(1) {
		for (int n = 0; n < num_; ++n) {
			forward_gpu_gemm(input + n * bottom_dim_, weights, output + n * top_dim_);
			if (bias_term_) {
				const Dtype* bias = this->blobs_[1]->gpu_data();
				forward_gpu_bias(output + n * top_dim_, bias);
			}
		}
    	return;
	}
	int height = 0, width = 0, kernel_h = 0, kernel_w = 0, pad_h = 0, pad_w = 0;
	int stride_h = 0, stride_w = 0, dilation_h = 0, dilation_w = 0;
	Dtype *d_input_padded = NULL;
	height = conv_input_shape_.cpu_data()[1];
	width = conv_input_shape_.cpu_data()[2];
	kernel_h = kernel_shape_.cpu_data()[0];
	kernel_w = kernel_shape_.cpu_data()[1];
	pad_h = pad_.cpu_data()[0];
	pad_w = pad_.cpu_data()[1];
	stride_h = stride_.cpu_data()[0];
	stride_w = stride_.cpu_data()[1];
	dilation_h = dilation_.cpu_data()[0];
	dilation_w = dilation_.cpu_data()[1];
	if(pad_h == 0 && pad_w == 0)
		d_input_padded = (Dtype *)input;
	else {
		d_input_padded = d_input_padded_;
		for (int n = 0; n < num_; ++n)
			copy_input_data<Dtype>(d_input_padded + n * conv_in_channels_ * (height + pad_h) * (width + pad_w), 
				input + n * bottom_dim_, conv_in_channels_, height, width, pad_h, pad_w);
	}

	// start computation
	for (int g = 0; g < group_; ++g) {
		const Dtype *in_temp = d_input_padded + conv_in_channels_ / group_ * g * (height + pad_h) * (width + pad_w);
		const int row_offset = conv_out_channels_ / group_ + 1;
		const int M = conv_out_channels_ / group_;
		const int *rowptr = nz_weight_index_pointers_.gpu_data() + row_offset * g;
		const Dtype *values = nz_weight_values_.gpu_data()+ weight_offset_ * g;
		const int *colidx = nz_weight_indices_.gpu_data()+ weight_offset_ * g;
		caffe_gpu_sconv<Dtype>(false, in_temp, rowptr, colidx, values, NULL,
			height, width, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
			kernel_h, kernel_w, output + output_offset_ * g, M);
	}
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
