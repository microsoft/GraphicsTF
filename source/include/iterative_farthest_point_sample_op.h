// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef ITERATIVE_FARTHEST_POINT_SAMPLE_H_
#define ITERATIVE_FARTHEST_POINT_SAMPLE_H_

#pragma warning(push, 0)
#include <unsupported/Eigen/CXX11/Tensor>
#include <tensorflow/core/framework/tensor.h>
#pragma warning(pop)

#include "utils_common.h"

#pragma warning(disable : 4661)

namespace tensorflow {

namespace functor {

template <typename T, typename Device>
struct IterativeFarthestPointSample {
    virtual void operator()(const Device& d, const Tensor* in_points, Tensor* tmp_points, Tensor* out_points) = 0;
};

template <typename T>
struct IterativeFarthestPointSample<T, GpuDevice> {
    virtual void operator()(const GpuDevice& d, const Tensor* in_points, Tensor* tmp_points, Tensor* out_points);
};

template struct IterativeFarthestPointSample<float, GpuDevice>;
template struct IterativeFarthestPointSample<Eigen::half, GpuDevice>;
}
}

#endif