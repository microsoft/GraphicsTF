// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef KERNEL_PREDICT_CONV_2D_OP_H
#define KERNEL_PREDICT_CONV_2D_OP_H

#pragma warning(push, 0)
#include <unsupported/Eigen/CXX11/Tensor>
#include <tensorflow/core/framework/tensor.h>
#pragma warning(pop)

#include "utils_common.h"

#pragma warning(disable : 4661)

namespace tensorflow {
namespace functor {

template <typename T, typename Device>
struct KernelPredictConv2D {
	virtual void operator()(const Device& d, const Tensor* input, const Tensor* filter, Tensor* output, TensorFormat data_format) = 0;
};

template <typename T, typename Device>
struct KernelPredictConv2DGradInput {
	virtual void operator()(const Device& d, const Tensor* filter, const Tensor* out_backprop, Tensor* output, TensorFormat data_format) = 0;
};

template <typename T, typename Device>
struct KernelPredictConv2DGradFilter {
	virtual void operator()(const Device& d, const Tensor* input, const Tensor* out_backprop, Tensor* output, TensorFormat data_format) = 0;
};

template <typename T>
struct KernelPredictConv2D<T, GpuDevice> {
	virtual void operator()(const GpuDevice& d, const Tensor* input, const Tensor* filter, Tensor* output, TensorFormat data_format);
};

template <typename T>
struct KernelPredictConv2DGradInput<T, GpuDevice> {
	virtual void operator()(const GpuDevice& d, const Tensor* filter, const Tensor* out_backprop, Tensor* output, TensorFormat data_format);
};

template <typename T>
struct KernelPredictConv2DGradFilter<T, GpuDevice> {
	virtual void operator()(const GpuDevice& d, const Tensor* input, const Tensor* out_backprop, Tensor* output, TensorFormat data_format);
};

template struct KernelPredictConv2D<float, GpuDevice>;
template struct KernelPredictConv2DGradInput<float, GpuDevice>;
template struct KernelPredictConv2DGradFilter<float, GpuDevice>;
template struct KernelPredictConv2D<Eigen::half, GpuDevice>;
template struct KernelPredictConv2DGradInput<Eigen::half, GpuDevice>;
template struct KernelPredictConv2DGradFilter<Eigen::half, GpuDevice>;

}}

#endif  // KERNEL_PREDICT_CONV_2D_OP_H