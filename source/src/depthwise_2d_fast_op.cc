// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma warning(push, 0)
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#pragma warning(pop)

#include <vector>

#include "depthwise_2d_fast_op.h"

namespace tensorflow {

REGISTER_OP("DepthwiseConvFast")
.Input("input: T")
.Input("filter: T")
.Attr("kernel_size: int")
.Attr("T: {float, float16}")
.Output("output: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
    c->set_output(0, input_shape);
    return Status::OK();
})
.Doc(R"doc(Depthwise 2D convolution with custom implementation)doc");

REGISTER_OP("DepthwiseConvFastGradFilter")
.Input("output_diff: T")
.Input("input: T")
.Attr("kernel_size: int")
.Attr("T: {float, float16}")
.Output("filter_diff: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    shape_inference::ShapeHandle indices_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &indices_shape));

    int kernel_size;
    TF_RETURN_IF_ERROR(c->GetAttr("kernel_size", &kernel_size));

    auto channels = c->Dim(indices_shape, 3);
    shape_inference::ShapeHandle grad_shape = c->MakeShape(
            std::vector<shape_inference::DimensionHandle>{c->MakeDim(kernel_size), c->MakeDim(kernel_size), channels, c->MakeDim(1)});
    c->set_output(0, grad_shape);
    return Status::OK();
})
.Doc(R"doc(Depthwise 2D convolution with custom implementation for filter gradient)doc");

REGISTER_OP("DepthwiseConvFastGradInput")
.Input("output_diff: T")
.Input("filter: T")
.Attr("T: {float, float16}")
.Output("input_diff: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
    c->set_output(0, input_shape);
    return Status::OK();
})
.Doc(R"doc(Depthwise 2D convolution with custom implementation for input gradient)doc");

template <typename T, typename Device>
class Depthwise2DFastOp : public OpKernel {
public:
    explicit Depthwise2DFastOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext * context) override
    {
        const Tensor& input_i = context->input(0);
        const Tensor& filter_i = context->input(1);

        Tensor* output_o = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("output", input_i.shape(), &output_o));

        functor::Depthwise2DFast<T, Device> depth2DFast;
        depth2DFast(context->eigen_device<Device>(), &input_i, &filter_i, output_o);
    }
};

template <typename T, typename Device>
class Depthwise2DFastGradFilterOp : public OpKernel
{
public:
    explicit Depthwise2DFastGradFilterOp(OpKernelConstruction *context) : filter_size(3), OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &this->filter_size));
    }

    void Compute(OpKernelContext * context) override
    {
        const Tensor& output_diff_i = context->input(0);
        const Tensor& input_i = context->input(1);

        auto batch_size = output_diff_i.shape().dim_size(0);
        auto channel_size = output_diff_i.shape().dim_size(3);

        auto filter_diff_shape = TensorShape(gtl::ArraySlice<int64>{filter_size, filter_size, channel_size, 1});

        Tensor* filter_diff_o = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("filter_diff", filter_diff_shape, &filter_diff_o));

        functor::Depthwise2DFastGradFilter<T, Device> depth2DFastGradFilter;
        depth2DFastGradFilter(context->eigen_device<Device>(), &output_diff_i, &input_i, filter_diff_o);
    }

protected:
    int filter_size;
};

template <typename T, typename Device>
class Depthwise2DFastGradInputOp : public OpKernel
{
public:
    explicit Depthwise2DFastGradInputOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext * context) override
    {
        const Tensor& output_diff_i = context->input(0);
        const Tensor& filter_i = context->input(1);

        Tensor* input_diff_o = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("input_diff", output_diff_i.shape(), &input_diff_o));

        functor::Depthwise2DFastGradInput<T, Device> depth2DFastGradInput;
        depth2DFastGradInput(context->eigen_device<Device>(), &output_diff_i, &filter_i, input_diff_o);
    }
};

REGISTER_KERNEL_BUILDER(Name("DepthwiseConvFast").Device(DEVICE_GPU).TypeConstraint<float>("T"), Depthwise2DFastOp<float, GpuDevice>)
REGISTER_KERNEL_BUILDER(Name("DepthwiseConvFastGradFilter").Device(DEVICE_GPU).TypeConstraint<float>("T"), Depthwise2DFastGradFilterOp<float, GpuDevice>)
REGISTER_KERNEL_BUILDER(Name("DepthwiseConvFastGradInput").Device(DEVICE_GPU).TypeConstraint<float>("T"), Depthwise2DFastGradInputOp<float, GpuDevice>)
REGISTER_KERNEL_BUILDER(Name("DepthwiseConvFast").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), Depthwise2DFastOp<Eigen::half, GpuDevice>)
REGISTER_KERNEL_BUILDER(Name("DepthwiseConvFastGradFilter").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), Depthwise2DFastGradFilterOp<Eigen::half, GpuDevice>)
REGISTER_KERNEL_BUILDER(Name("DepthwiseConvFastGradInput").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), Depthwise2DFastGradInputOp<Eigen::half, GpuDevice>)
}