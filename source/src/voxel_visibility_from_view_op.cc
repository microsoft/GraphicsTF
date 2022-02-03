// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma warning(push, 0)
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#pragma warning(pop)

#include "voxel_visibility_from_view_op.h"

namespace tensorflow {

REGISTER_OP("VoxelVisibilityFromView")
.Input("mask_voxel: T")
.Input("cam_pose: float")
.Attr("T: {int32, int64}")
.Output("visibility_voxel: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
	shape_inference::ShapeHandle mask_voxel_shape;
	c->WithRank(c->input(0), 4, &mask_voxel_shape);
	c->set_output(0, mask_voxel_shape);
	return Status::OK();
})
.Doc(R"doc(Voxel Visibility From Viewpoint)doc");

template <typename T, typename Device>
class VoxelVisibilityFromViewOp : public OpKernel {
public:
	explicit VoxelVisibilityFromViewOp(OpKernelConstruction *context) : OpKernel(context) {

	}

	void Compute(OpKernelContext* context) override {
		auto mask_voxel = context->input(0);
		auto cam_pose = context->input(1);
		auto mask_voxel_shape = mask_voxel.shape();

		Tensor* visibility_voxel = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, mask_voxel_shape, &visibility_voxel));

		functor::VoxelVisibilityFromView<T, Device> voxelVisibilityFromView;
		voxelVisibilityFromView(context->eigen_device<Device>(), &mask_voxel, &cam_pose, visibility_voxel);
	}
};

REGISTER_KERNEL_BUILDER(Name("VoxelVisibilityFromView").Device(DEVICE_GPU).TypeConstraint<int32>("T"), VoxelVisibilityFromViewOp<int32, GpuDevice>);
REGISTER_KERNEL_BUILDER(Name("VoxelVisibilityFromView").Device(DEVICE_GPU).TypeConstraint<int64>("T"), VoxelVisibilityFromViewOp<int64, GpuDevice>);
}
