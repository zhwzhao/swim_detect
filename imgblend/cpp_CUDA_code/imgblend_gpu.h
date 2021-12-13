#ifndef _BALL_QUERY_GPU_H
#define _BALL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int imgblend_wrapper_cpp(int width, int height, int left, int right,
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void imgblend_kernel_launcher_cuda(int width, int height, int left, int right,
	const int *xyz, const int *new_xyz, int *idx);

#endif
