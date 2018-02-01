#include <vector>

#include "caffe/layers/conv_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Timer timer;
  timer.Start();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* h_bottom_data = bottom[i]->cpu_data();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    if(Caffe::conv_mode() == Caffe::SCONV_PAR) {
      printf("ERROR: not implemented yet\n");
    } else if(Caffe::conv_mode() == Caffe::SCONV) {
	  for (int n = 0; n < this->num_; ++n) {
        this->forward_gpu_sconv(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      }
    } else { 
      printf("ERROR: conv_mode not supported for ConvolutionReLULayer\n");
    }
  }
  timer.Stop();
  printf("[cxh] %s: %.2f ms\n", this->layer_param().name().c_str(), timer.MicroSeconds()/1000);
}

template <typename Dtype>
void ConvolutionReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionReLULayer);
REGISTER_LAYER_CLASS(ConvolutionReLU);
}  // namespace caffe
