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

