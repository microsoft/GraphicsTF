// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma warning(push, 0)
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#pragma warning(pop)

#include "utils_common.h"

namespace tensorflow {

REGISTER_OP("GridSample")
.Input("x: T")
.Input("grid: T")
.Output("y: T")
.Attr("mode: string")
.Attr("norm: bool")
.Attr("data_format: string")
.Attr("T: {float, float16}")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	shape_inference::ShapeHandle x_shape;
	c->WithRank(c->input(0), 4, &x_shape);
	shape_inference::ShapeHandle grid_shape;
	c->WithRank(c->input(1), 4, &grid_shape);

	shape_inference::ShapeHandle y_shape = c->MakeShape({ c->Dim(grid_shape, 0), c->Dim(grid_shape, 1), c->Dim(grid_shape, 2), c->Dim(x_shape, 3) });
	c->set_output(0, y_shape);
	return Status::OK();
})
.Doc(R"doc(Grid sample implementation, similar to PyTroch function)doc");

template <typename T, typename Device>
class GridSampleOp : public OpKernel {
public:
	explicit GridSampleOp(OpKernelConstruction* context) : OpKernel(context), mode("bilinear"), norm(false) 
	{
		OP_REQUIRES_OK(context, context->GetAttr("mode", &this->mode));
		OP_REQUIRES_OK(context, context->GetAttr("norm", &this->norm));
	}

	void Compute(OpKernelContext* context) override 
	{
		VLOG(0) << "Grid sample function is for the debug-only usage currently!";

		const Tensor& x = context->input(0);
		const Tensor& grid = context->input(1);

		auto b = grid.dim_size(0);
		auto w = grid.dim_size(1);
		auto h = grid.dim_size(2);
		auto c = x.dim_size(3);

		Tensor* y = nullptr;
		
		OP_REQUIRES_OK(context, context->allocate_output("y", TensorShape{ b, w, h, c }, &y));
	}

protected:
	std::string mode;
	bool norm;
};

REGISTER_KERNEL_BUILDER(Name("GridSample").Device(DEVICE_GPU).TypeConstraint<float>("T"), GridSampleOp<float, GpuDevice>);
}