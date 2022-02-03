// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#pragma warning(push, 0)
#include <tensorflow/core/util/gpu_device_functions.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#pragma warning(pop)

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector>

#include "utils_kernel.h"
#include "utils_common.h"

#include "kernel_predict_conv_2d_op.h"

namespace tensorflow { namespace functor {

template<typename T>
__global__ void k_KernelPredictConv2D_nhwc(
	const	T *const	input_ptr,
	const	T *const	filter_ptr,
			T*			output_ptr,
	const	int64		channels,
	const	int64		width,
	const	int64		height,
	const	int64		kernel_r,
	const	int64		stride,
	const	int64		num_threads
)
{
	for (auto index = threadIdx.x + blockDim.x * blockIdx.x; index < num_threads; index += blockDim.x * gridDim.x)
	{
		const auto n = index / channels / width / height;
		const auto h = index / channels / width % height;
		const auto w = index / channels % width;
		const auto c = index % channels;
		const auto k_stride = kernel_r * kernel_r;
		const auto input_width = width * stride;
		const auto input_height = height * stride;

		auto filter_base_ptr = filter_ptr + ((n * height + h) * width + w) * k_stride;

		T res = T(0);
		for (auto i = 0; i < k_stride; ++i) 
		{
			auto o_w = i % kernel_r - kernel_r / 2;
			auto o_h = i / kernel_r - kernel_r / 2;
			auto r_w = w * stride + o_w;
			auto r_h = h * stride + o_h;

			auto input_value = (r_w >= 0 && r_w < input_width && r_h >= 0 && r_h < input_height) ?
				(*(input_ptr + ((n * input_height + r_h) * input_width + r_w) * channels + c)) : T(0);
			res += input_value * filter_base_ptr[i];
		}
		output_ptr[index] = res;
	}
}

template<typename T>
__global__ void k_KernelPredictConv2DGradInput_nhwc(
	const	T* const	filter_ptr,
	const	T* const	output_grad_ptr,
			T*			input_grad_ptr,
	const	int64		channels,
	const	int64		width,
	const	int64		height,
	const	int64		kernel_r,
	const	int64		stride,
	const	int64		num_threads
)
{
	for (auto index = threadIdx.x + blockDim.x * blockIdx.x; index < num_threads; index += blockDim.x * gridDim.x)
	{
		const auto n = index / channels / width / height;
		const auto h = index / channels / width % height;
		const auto w = index / channels % width;
		const auto c = index % channels;
		const auto k_stride = kernel_r * kernel_r;

		const auto output_width = width / stride;
		const auto output_height = height / stride;

		T res = T(0);
		for (auto i = 0; i < k_stride; ++i)
		{
			auto o_w = i % kernel_r - kernel_r / 2;
			auto o_h = i / kernel_r - kernel_r / 2;
			auto r_w = w + o_w;
			auto r_h = h + o_h;

			if (r_w < 0 || r_w >= output_width || r_h < 0 || r_h >= output_height)
				continue;
			if (r_w % stride != 0 || r_h % stride != 0)
				continue;
			r_w /= stride;
			r_h /= stride;

			auto out_backprop_value = *(output_grad_ptr + ((n * height + r_h) * width + r_w) * channels + c);
			auto filter_value = *(filter_ptr + ((n * height + r_h) * width + r_w) * k_stride + k_stride - i - 1);
			res += out_backprop_value * filter_value;
		}
		input_grad_ptr[index] = res;
	}
}

template<typename T>
__global__ void k_KernelPredictConv2DGradFilter_nhwc(
	const	T* const	input_ptr,
	const	T* const	output_grad_ptr,
			T*			filter_grad_ptr,
	const	int64		channels,
	const	int64		width,
	const	int64		height,
	const	int64		kernel_r,
	const	int64		stride,
	const	int64		num_threads
)
{
	for (auto index = threadIdx.x + blockDim.x * blockIdx.x; index < num_threads; index += blockDim.x * gridDim.x)
	{
		const auto k_stride = kernel_r * kernel_r;
		const auto n = index / k_stride / width / height;
		const auto h = index / k_stride / width % height;
		const auto w = index / k_stride % width;
		const auto k = index % k_stride;
		const auto input_width = width * stride;
		const auto input_height = height * stride;

		auto o_w = k % kernel_r - kernel_r / 2;
		auto o_h = k / kernel_r - kernel_r / 2;

		auto r_w = w * stride + o_w;
		auto r_h = h * stride + o_h;

		T res = T(0);
		for (auto i = 0; i < channels; ++i)
		{
			auto out_backprop_value = *(output_grad_ptr + (((n * height + h) * width + w) * channels) + i);
			auto input_value = (r_w >= 0 && r_w < input_width&& r_h >= 0 && r_h < input_height) ? 
				(*(input_ptr + ((n * input_height + r_h) * input_width + r_w) * channels + i)) : T(0);
			res += out_backprop_value * input_value;
		}

		filter_grad_ptr[index] = res;
	}
}

template<typename T>
void KernelPredictConv2D<T, GpuDevice>::operator()(const GpuDevice& d, const Tensor* input, const Tensor* filter, 
	Tensor* output, const int64 stride, TensorFormat data_format)
{
	auto output_shape = output->shape();
	auto output_height = GetTensorDim(output_shape, data_format, 'H');
	auto output_width = GetTensorDim(output_shape, data_format, 'W');
	auto output_channel = GetTensorDim(output_shape, data_format, 'C');

	auto filter_shape = filter->shape();
	auto filter_length = GetTensorDim(filter_shape, data_format, 'C');
	auto filter_radius = static_cast<int64>(std::sqrt(filter_length));
	assert(filter_radius * filter_radius == filter_length);

	switch (data_format)
	{
	case FORMAT_NHWC:
		k_KernelPredictConv2D_nhwc<T> <<<BLOCKS, THREADS, 0, d.stream() >>> (
			input->unaligned_flat<T>().data(),
			filter->unaligned_flat<T>().data(),
			output->unaligned_flat<T>().data(),
			output_channel,
			output_width,
			output_height,
			filter_radius,
			stride,
			output_shape.num_elements());
		break;
	case FORMAT_NCHW:
		break;
	default:
		break;
	}
}

template<typename T>
void KernelPredictConv2DGradInput<T, GpuDevice>::operator()(const GpuDevice& d, const Tensor* filter, const Tensor* out_backprop, Tensor* output, const int64 stride, TensorFormat data_format)
{
	auto output_shape = output->shape();
	auto output_height = GetTensorDim(output_shape, data_format, 'H');
	auto output_width = GetTensorDim(output_shape, data_format, 'W');
	auto output_channel = GetTensorDim(output_shape, data_format, 'C');

	auto filter_shape = filter->shape();
	auto filter_length = GetTensorDim(filter_shape, data_format, 'C');
	auto filter_radius = static_cast<int64>(std::sqrt(filter_length));
	assert(filter_radius * filter_radius == filter_length);

	switch (data_format)
	{
	case FORMAT_NHWC:
		k_KernelPredictConv2DGradInput_nhwc<T><<<BLOCKS, THREADS, 0, d.stream() >>>(
			filter->unaligned_flat<T>().data(),
			out_backprop->unaligned_flat<T>().data(),
			output->unaligned_flat<T>().data(),
			output_channel,
			output_width,
			output_height,
			filter_radius,
			stride,
			output_shape.num_elements());
		break;
	case FORMAT_NCHW:
		break;
	default:
		break;
	}
}

template<typename T>
void KernelPredictConv2DGradFilter<T, GpuDevice>::operator()(const GpuDevice& d, const Tensor* input, const Tensor* out_backprop, Tensor* output, const int64 stride, TensorFormat data_format)
{
	auto output_shape = output->shape();
	auto output_height = GetTensorDim(output_shape, data_format, 'H');
	auto output_width = GetTensorDim(output_shape, data_format, 'W');
	auto output_channel = GetTensorDim(output_shape, data_format, 'C');
	auto filter_radius = static_cast<int64>(std::sqrt(output_channel));
	assert(filter_radius * filter_radius == output_channel);

	auto input_shape = input->shape();
	auto input_channel = GetTensorDim(input_shape, data_format, 'C');

	switch (data_format)
	{
	case FORMAT_NHWC:
		k_KernelPredictConv2DGradFilter_nhwc<T><<<BLOCKS, THREADS, 0, d.stream()>>>(
			input->unaligned_flat<T>().data(),
			out_backprop->unaligned_flat<T>().data(),
			output->unaligned_flat<T>().data(),
			input_channel,
			output_width,
			output_height,
			filter_radius,
			stride,
			output_shape.num_elements());
		break;
	case FORMAT_NCHW:
		break;
	default:
		break;
	}
}

}
}

#endif