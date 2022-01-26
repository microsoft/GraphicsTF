// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
/*
 * Microsoft Research Asia, Internet Graphics
 */

#pragma warning(push, 0)
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#pragma warning(pop)

#include "scatter_operator_op.h"

namespace tensorflow {

REGISTER_OP("ScatterOperator")
.Input("indices: int32")
.Input("feat: float")
.Input("length: int32")
.Attr("type: int")
.Output("scattered: float")
.Output("statistics: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    shape_inference::ShapeHandle feat_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &feat_shape));
    auto batch_size = c->Dim(feat_shape, 0);
    auto channel_size = c->Dim(feat_shape, 2);

    shape_inference::ShapeHandle scatter_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &scatter_shape));
    auto scatter_length = c->Dim(scatter_shape, 0);

    shape_inference::ShapeHandle scattered_shape = c->MakeShape(
            std::vector<shape_inference::DimensionHandle>{batch_size, c->MakeDim(scatter_length), channel_size});
    c->set_output(0, scattered_shape);

    shape_inference::ShapeHandle statistics_shape = c->MakeShape(
            std::vector<shape_inference::DimensionHandle>{batch_size, c->MakeDim(scatter_length), c->MakeDim(1)});
    c->set_output(1, statistics_shape);
    return Status::OK();
})
.Doc(R"doc(Scatter operator to insert sparse 1-D array to dense 1-D array)doc");

REGISTER_OP("ScatterOperatorGrad")
.Input("indices: int32")
.Input("scattered_feat: float")
.Input("scattered_grad: float")
.Attr("type:int")
.Output("grad: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    shape_inference::ShapeHandle indices_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &indices_shape));
    shape_inference::ShapeHandle scattered_grad_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &scattered_grad_shape));

    auto batch_size = c->Dim(indices_shape, 0);
    auto length = c->Dim(indices_shape, 1);
    auto channel_size = c->Dim(scattered_grad_shape, 2);

    shape_inference::ShapeHandle grad_shape = c->MakeShape(
            std::vector<shape_inference::DimensionHandle>{batch_size, length, channel_size});
    c->set_output(0, grad_shape);
    return Status::OK();
})
.Doc(R"doc(Scatter operator gradient)doc");

template <typename Device>
class ScatterOperatorOp : public OpKernel {
public:
    explicit ScatterOperatorOp(OpKernelConstruction *context) : type(ScatterType::SUM), OpKernel(context)
    {
        int32 type_code;
        OP_REQUIRES_OK(context, context->GetAttr("type", &type_code));
        this->type = static_cast<ScatterType >(type_code);
    }

    void Compute(OpKernelContext * context) override
    {
        const Tensor& indices_input = context->input(0);
        const Tensor& feat_input = context->input(1);

        const Tensor& shape_tensor = context->input(2);
        auto length = shape_tensor.shape().dim_size(0);

        auto batch_size = indices_input.shape().dim_size(0);
        auto channel_size = feat_input.shape().dim_size(2);

        auto scattered_shape = TensorShape(gtl::ArraySlice<int64>{batch_size, length, channel_size});
        auto statistics_shape = TensorShape(gtl::ArraySlice<int64>{batch_size, length, 1});

        Tensor* scattered_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("scattered", scattered_shape, &scattered_output));
        Tensor* statistics_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("statistics", statistics_shape, &statistics_output));

        functor::ScatterOperator<Device> scatterOperator;
        scatterOperator(context->eigen_device<Device>(), &indices_input, &feat_input, scattered_output,
                        statistics_output, this->type);
    }

protected:
    ScatterType type;
};

template <typename Device>
class ScatterOperatorGradOp : public OpKernel {
public:
    explicit ScatterOperatorGradOp(OpKernelConstruction *context) : type(ScatterType::SUM), OpKernel(context)
    {
        int32 type_code;
        OP_REQUIRES_OK(context, context->GetAttr("type", &type_code));
        this->type = static_cast<ScatterType >(type_code);
    }

    void Compute(OpKernelContext * context) override
    {
        const Tensor& indices_input = context->input(0);
        const Tensor& scattered_feat_input = context->input(1);
        const Tensor& scattered_grad_input = context->input(2);

        auto batch_size = indices_input.shape().dim_size(0);
        auto length = indices_input.shape().dim_size(1);
        auto channel_size = scattered_grad_input.shape().dim_size(2);
        auto grad_shape = TensorShape(gtl::ArraySlice<int64>{batch_size, length, channel_size});

        Tensor* grad_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("grad", grad_shape, &grad_output));

        functor::ScatterOperatorGrad<Device> scatterOperatorGrad;
        scatterOperatorGrad(context->eigen_device<Device>(), &indices_input, &scattered_feat_input, &scattered_grad_input,
                            grad_output, this->type);
    }

protected:
    ScatterType type;
};

REGISTER_KERNEL_BUILDER(Name("ScatterOperator").Device(DEVICE_GPU), ScatterOperatorOp<GpuDevice>);
REGISTER_KERNEL_BUILDER(Name("ScatterOperatorGrad").Device(DEVICE_GPU), ScatterOperatorGradOp<GpuDevice>);
}