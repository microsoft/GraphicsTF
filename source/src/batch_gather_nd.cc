// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma warning(push, 0)
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#pragma warning(pop)

#include "utils_common.h"
#include <vector>

namespace tensorflow {

REGISTER_OP("BatchGatherNd")
.Input("input: T")
.Input("index: idxT")
.Output("output: T")
.Attr("batch_dims: int")
.Attr("T: {float, float16}")
.Attr("idxT: {int32, int64}")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	int batch_dim;
	TF_RETURN_IF_ERROR(c->GetAttr("batch_dims", &batch_dim));

	shape_inference::ShapeHandle input_shape = c->input(0);
	shape_inference::ShapeHandle index_shape = c->input(1);

	std::vector<shape_inference::DimensionHandle> output_ref;

	for (int r = 0; r < c->Rank(index_shape) - 1; ++r)
		output_ref.push_back(c->Dim(index_shape, r));
	auto offset = c->Value(c->Dim(index_shape, (int64)c->Rank(index_shape) - 1));
	for (int r = batch_dim + offset; r < c->Rank(index_shape); ++r)
		output_ref.push_back(c->Dim(input_shape, r));
	shape_inference::ShapeHandle output_shape = c->MakeShape(output_ref);
	c->set_output(0, output_shape);
	return Status::OK();
})
.Doc(R"doc(Batch Gather Nd with custom implementation)doc");

template <typename T, typename Device>
class BatchGatherNdOp : public OpKernel {
public:
	explicit BatchGatherNdOp(OpKernelConstruction *context) : OpKernel(context), batch_dim(0) 
	{
		OP_REQUIRES_OK(context, context->GetAttr("batch_dims", &this->batch_dim));
	}

	void Compute(OpKernelContext* context) override
	{
		VLOG(0) << "Batch gather nd function is for the debug-only usage currently!";

		const Tensor& input = context->input(0);
		const Tensor& index = context->input(1);

		auto output_shape = TensorShape();
		for (int r = 0; r < index.shape().dims() - 1; ++r)
			output_shape.AddDim(index.dim_size(r));
		auto offset = index.shape().dim_size(index.shape().dims() - 1);
		for (int r = this->batch_dim + offset; r < input.shape().dims(); ++r)
			output_shape.AddDim(input.dim_size(r));
		
		Tensor* output = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output("output", output_shape, &output));
	}

protected:
	int batch_dim;
};

REGISTER_KERNEL_BUILDER(Name("BatchGatherNd").Device(DEVICE_GPU).TypeConstraint<float>("T").TypeConstraint<int32>("idxT"), BatchGatherNdOp<float, GpuDevice>)
REGISTER_KERNEL_BUILDER(Name("BatchGatherNd").Device(DEVICE_GPU).TypeConstraint<float>("T").TypeConstraint<int64>("idxT"), BatchGatherNdOp<float, GpuDevice>)
}