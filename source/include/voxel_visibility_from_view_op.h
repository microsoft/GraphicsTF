// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef VOXEL_VISIBILITY_FROM_VIEW_OP_
#define VOXEL_VISIBILITY_FROM_VIEW_OP_

#pragma warning(push, 0)
#include <unsupported/Eigen/CXX11/Tensor>
#include <tensorflow/core/framework/tensor.h>
#pragma warning(pop)

#include "utils_common.h"

#pragma warning(disable : 4661)

namespace tensorflow {
namespace functor {

template <typename T, typename Device>
struct VoxelVisibilityFromView
{
	virtual void operator()(const Device& d, const Tensor* mask_voxel, const Tensor* cam_pos, Tensor* visibility_voxel) = 0;
};

template <typename T>
struct VoxelVisibilityFromView<T, GpuDevice>
{
	virtual void operator()(const GpuDevice& d, const Tensor* mask_voxel, const Tensor* cam_pos, Tensor* visibility_voxel);
};

template struct VoxelVisibilityFromView<int32, GpuDevice>;
template struct VoxelVisibilityFromView<int64, GpuDevice>;
}

}

#endif