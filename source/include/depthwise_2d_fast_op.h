// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef DEPTHWISE_2D_FAST_OPS_H_
#define DEPTHWISE_2D_FAST_OPS_H_

#pragma warning(push, 0)
#include <unsupported/Eigen/CXX11/Tensor>
#include <tensorflow/core/framework/tensor.h>
#pragma warning(pop)

#include "utils_common.h"

#pragma warning(disable : 4661)

namespace tensorflow{

namespace functor {

template <typename T, typename Device>
struct Depthwise2DFast {
    virtual void operator()(const Device& d, const Tensor* input, const Tensor* filter, Tensor* output)=0;
};

template <typename T, typename Device>
struct Depthwise2DFastGradFilter {
    virtual void operator()(const GpuDevice& d, const Tensor* output_diff, const Tensor* input, Tensor* filter_diff)=0;
};

template <typename T, typename Device>
struct Depthwise2DFastGradInput {
    virtual void operator()(const GpuDevice& d, const Tensor* output_diff, const Tensor* filter, Tensor* input_diff)=0;
};

template <typename T>
struct Depthwise2DFast<T, GpuDevice> {
    virtual void operator()(const GpuDevice& d, const Tensor* input, const Tensor* filter, Tensor* output);
};

template <typename T>
struct Depthwise2DFastGradFilter<T, GpuDevice> {
    virtual void operator()(const GpuDevice& d, const Tensor* output_diff, const Tensor* input, Tensor* filter_diff);
};

template <typename T>
struct Depthwise2DFastGradInput<T, GpuDevice> {
    virtual void operator()(const GpuDevice& d, const Tensor* output_diff, const Tensor* filter, Tensor* input_diff);
};

template struct Depthwise2DFast<float, GpuDevice>;
template struct Depthwise2DFastGradFilter<float, GpuDevice>;
template struct Depthwise2DFastGradInput<float, GpuDevice>;
template struct Depthwise2DFast<Eigen::half, GpuDevice>;
template struct Depthwise2DFastGradFilter<Eigen::half, GpuDevice>;
template struct Depthwise2DFastGradInput<Eigen::half, GpuDevice>;

}
}

#endif