// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef QUERY_BALL_POINT_H_
#define QUERY_BALL_POINT_H_

#pragma warning(push, 0)
#include <unsupported/Eigen/CXX11/Tensor>
#include <tensorflow/core/framework/tensor.h>
#pragma warning(pop)

#include "utils_common.h"

#pragma warning(disable : 4661)

namespace tensorflow {

namespace functor {

template <typename T, typename Device>
struct QueryBallPoint {
    virtual void operator()(const Device& d, const Tensor* in_points, const Tensor* query_points, Tensor* queried_indices, Tensor* valid_counts, T radius) = 0;
};

template <typename T>
struct QueryBallPoint<T, GpuDevice> {
    virtual void operator()(const GpuDevice& d, const Tensor* in_points, const Tensor* query_points, Tensor* queried_indices, Tensor* valid_counts, T radius);
};

template struct QueryBallPoint<float, GpuDevice>;
template struct QueryBallPoint<Eigen::half, GpuDevice>;
}
}

#endif