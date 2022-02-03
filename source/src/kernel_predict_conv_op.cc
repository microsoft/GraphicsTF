// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma warning(push, 0)
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#pragma warning(pop)

#include "kernel_predict_conv_2d_op.h"

namespace tensorflow {

REGISTER_OP("KernelPredictConv2D")
.Input("input: T")
.Input("filter: T")
.Output("output: T")
.Attr("stride: int")
.Attr(GetConvnetDataFormatAttrString())
.Attr("T: {float, float16}")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	std::string data_format_str;
	TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format_str));
	TensorFormat data_format;
	FormatFromString(data_format_str, &data_format);
	int64 stride;
	TF_RETURN_IF_ERROR(c->GetAttr("stride", &stride));

	shape_inference::ShapeHandle conv_input_shape;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &conv_input_shape));
	auto H = c->Dim(conv_input_shape, GetTensorDimIndex(data_format, 'H'));
	auto W = c->Dim(conv_input_shape, GetTensorDimIndex(data_format, 'W'));
	c->Divide(H, c->MakeDim(stride), true, &H);
	c->Divide(W, c->MakeDim(stride), true, &W);
	c->set_output(0, conv_input_shape);
	return Status::OK();
})
.Doc(R"doc(Kernel predict convolution layer)doc");

REGISTER_OP("KernelPredictConv2DGradFilter")
.Input("input: T")
.Input("filter_size: int32")
.Input("out_backprop: T")
.Output("output: T")
.Attr("stride: int")
.Attr(GetConvnetDataFormatAttrString())
.Attr("T: {float, float16}")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	shape_inference::ShapeHandle filter_shape;
	TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &filter_shape));
	TF_RETURN_IF_ERROR(c->WithRank(filter_shape, 4, &filter_shape));
	c->set_output(0, filter_shape);
	return Status::OK();
})
.Doc(R"doc(Kernel predict convolution filter backprop layer)doc");

REGISTER_OP("KernelPredictConv2DGradInput")
.Input("input_size: int32")
.Input("filter: T")
.Input("out_backprop: T")
.Output("output: T")
.Attr("stride: int")
.Attr(GetConvnetDataFormatAttrString())
.Attr("T: {float, float16}")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	shape_inference::ShapeHandle input_shape;
	TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &input_shape));
	TF_RETURN_IF_ERROR(c->WithRank(input_shape, 4, &input_shape));
	c->set_output(0, input_shape);
	return Status::OK();
})
.Doc(R"doc(Kernel predict convolution data backprop layer)doc");

template <typename T, typename Device>
class KernelPredictConv2D : public OpKernel
{
public:
	explicit KernelPredictConv2D(OpKernelConstruction* context) : OpKernel(context), 
		stride_(1), data_format_(FORMAT_NHWC)
	{
		std::string data_format;
		OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
		OP_REQUIRES(context, FormatFromString(data_format, &this->data_format_), 
			errors::InvalidArgument("Invalid data format: ", data_format));

		OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
	}

	void Compute(OpKernelContext* context) override
	{
		const Tensor& input = context->input(0);
		const Tensor& filter = context->input(1);

		Tensor* output = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output("output", input.shape(), &output));

		functor::KernelPredictConv2D<T, Device> kernelPredictConv2D;
		kernelPredictConv2D(context->eigen_device<Device>(), &input, &filter, output, this->stride_, this->data_format_);
	}

protected:
	TensorFormat data_format_;
	int64 stride_;
};

template <typename T, typename Device>
class KernelPredictConv2DGradFilter : public OpKernel
{
public:
	explicit KernelPredictConv2DGradFilter(OpKernelConstruction* context) : OpKernel(context),
		stride_(1), data_format_(FORMAT_NHWC)
	{
		std::string data_format;
		OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
		OP_REQUIRES(context, FormatFromString(data_format, &this->data_format_),
			errors::InvalidArgument("Invalid data format: ", data_format));

		OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
	}

	void Compute(OpKernelContext* context) override
	{
		const Tensor& input = context->input(0);
		const Tensor& filter_sizes = context->input(1);
		const Tensor& out_backprop = context->input(2);

		TensorShape filter_shape;
		OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(filter_sizes.vec<int32>(), &filter_shape));

		Tensor* output = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output("output", filter_shape, &output));

		functor::KernelPredictConv2DGradFilter<T, Device> kernelPredictConv2DGradFilter;
		kernelPredictConv2DGradFilter(context->eigen_device<Device>(), &input, &out_backprop, output, this->stride_, this->data_format_);
	}

protected:
	TensorFormat data_format_;
	int64 stride_;
};

template <typename T, typename Device>
class KernelPredictConv2DGradInput : public OpKernel
{
public:
	explicit KernelPredictConv2DGradInput(OpKernelConstruction* context) : OpKernel(context),
		stride_(1), data_format_(FORMAT_NHWC)
	{
		std::string data_format;
		OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
		OP_REQUIRES(context, FormatFromString(data_format, &this->data_format_),
			errors::InvalidArgument("Invalid data format: ", data_format));

		OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
	}

	void Compute(OpKernelContext* context) override
	{
		const Tensor& input_sizes = context->input(0);
		const Tensor& filter = context->input(1);
		const Tensor& out_backprop = context->input(2);

		TensorShape input_shape;
		OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(input_sizes.vec<int32>(), &input_shape));


		Tensor* output = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output("output", input_shape, &output));

		functor::KernelPredictConv2DGradInput<T, Device> kernelPredictConv2DGradInput;
		kernelPredictConv2DGradInput(context->eigen_device<Device>(), &filter, &out_backprop, output, this->stride_, this->data_format_);
	}

protected:
	TensorFormat data_format_;
	int64 stride_;
};

REGISTER_KERNEL_BUILDER(Name("KernelPredictConv2D").Device(DEVICE_GPU).TypeConstraint<float>("T"), KernelPredictConv2D<float, GpuDevice>)
REGISTER_KERNEL_BUILDER(Name("KernelPredictConv2D").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), KernelPredictConv2D<Eigen::half, GpuDevice>)
REGISTER_KERNEL_BUILDER(Name("KernelPredictConv2DGradFilter").Device(DEVICE_GPU).HostMemory("filter_size").TypeConstraint<float>("T"), KernelPredictConv2DGradFilter<float, GpuDevice>)
REGISTER_KERNEL_BUILDER(Name("KernelPredictConv2DGradFilter").Device(DEVICE_GPU).HostMemory("filter_size").TypeConstraint<Eigen::half>("T"), KernelPredictConv2DGradFilter<Eigen::half, GpuDevice>)
REGISTER_KERNEL_BUILDER(Name("KernelPredictConv2DGradInput").Device(DEVICE_GPU).HostMemory("input_size").TypeConstraint<float>("T"), KernelPredictConv2DGradInput<float, GpuDevice>)
REGISTER_KERNEL_BUILDER(Name("KernelPredictConv2DGradInput").Device(DEVICE_GPU).HostMemory("input_size").TypeConstraint<Eigen::half>("T"), KernelPredictConv2DGradInput<Eigen::half, GpuDevice>)
}
