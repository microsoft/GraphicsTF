// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma warning(push, 0)
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#pragma warning(pop)

#include <vector>

#include "query_ball_point_op.h"

namespace tensorflow {

REGISTER_OP("QueryBallPoint")
.Input("in_points: T")
.Input("query_points: T")
.Attr("T: {float, float16}")
.Attr("radius: float")
.Attr("max_sample: int")
.Output("queried_indices: int32")
.Output("valid_counts: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    int max_sample;
    TF_RETURN_IF_ERROR(c->GetAttr("max_sample", &max_sample));
    shape_inference::ShapeHandle queried_points_shape;
    c->WithRank(c->input(1), 3, &queried_points_shape);
    shape_inference::ShapeHandle queried_indices_shape = c->MakeShape({ c->Dim(queried_points_shape, 0), c->Dim(queried_points_shape, 1), max_sample });
    c->set_output(0, queried_indices_shape);
    shape_inference::ShapeHandle valid_counts_shape = c->MakeShape({ c->Dim(queried_points_shape, 0), c->Dim(queried_points_shape, 1) });
    c->set_output(1, valid_counts_shape);
    return Status::OK();
})
.Doc(R"doc(Query Ball Point implementation)doc");

template <typename T, typename Device>
class QueryBallPointOp : public OpKernel {
public:
    explicit QueryBallPointOp(OpKernelConstruction *context) : OpKernel(context) {
        float radius_;
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
        this->radius = T(radius_);
        OP_REQUIRES(context, this->radius > T(0), errors::InvalidArgument("Received ball radius should be positive"));
        OP_REQUIRES_OK(context, context->GetAttr("max_sample", &this->max_sample));
        OP_REQUIRES(context, this->max_sample > 0, errors::InvalidArgument("Out number of samples should be positive"));
    }

    void Compute(OpKernelContext* context) override {
        auto in_points = context->input(0);
        auto query_points = context->input(1);

        int batch_size = static_cast<int>(query_points.shape().dim_size(0));
        int query_size = static_cast<int>(query_points.shape().dim_size(1));
        Tensor *queried_indices = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{ batch_size, query_size, this->max_sample }, &queried_indices));
        Tensor *valid_counts = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{ batch_size, query_size }, &valid_counts));

        functor::QueryBallPoint<T, Device> queryBallPoint;
        queryBallPoint(context->eigen_device<Device>(), &in_points, &query_points, queried_indices, valid_counts, this->radius);
    }
protected:
    T radius;
    int max_sample;
};

REGISTER_KERNEL_BUILDER(Name("QueryBallPoint").Device(DEVICE_GPU).TypeConstraint<float>("T"), QueryBallPointOp<float, GpuDevice>);
REGISTER_KERNEL_BUILDER(Name("QueryBallPoint").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), QueryBallPointOp<Eigen::half, GpuDevice>);
}
