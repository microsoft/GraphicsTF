// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma warning(push, 0)
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#pragma warning(pop)

#include <vector>

#include "iterative_farthest_point_sample_op.h"

namespace tensorflow {

REGISTER_OP("IterativeFarthestPointSample")
.Input("in_points: T")
.Attr("m: int")
.Attr("T: {float, float16}")
.Output("out_points: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    int m;
    TF_RETURN_IF_ERROR(c->GetAttr("m", &m));
    shape_inference::ShapeHandle in_points_shape;
    c->WithRank(c->input(0), 3, &in_points_shape);
    shape_inference::ShapeHandle out_points_shape = c->MakeShape({ c->Dim(in_points_shape, 0), m });
    c->set_output(0, out_points_shape);
    return Status::OK();
})
.Doc(R"doc(Iterative Farthest Point Sample implementation)doc");

template <typename T, typename Device>
class IterativeFarthestPointSampleOp : public OpKernel {
public:
    explicit IterativeFarthestPointSampleOp(OpKernelConstruction *context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("m", &this->m));
        OP_REQUIRES(context, this->m > 0, errors::InvalidArgument("Out points size should be positive"));
    }

    void Compute(OpKernelContext* context) override {
        auto in_points = context->input(0);
        int batch_size = static_cast<int>(in_points.shape().dim_size(0));
        int in_num_points = static_cast<int>(in_points.shape().dim_size(1));
        Tensor *out_points = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{ batch_size, this->m }, &out_points));
        Tensor tmp_points;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape{ batch_size, in_num_points }, &tmp_points));

        functor::IterativeFarthestPointSample<T, Device> iterativeFarthestPointSample;
        iterativeFarthestPointSample(context->eigen_device<Device>(), &in_points, &tmp_points, out_points);
    }

protected:
    int m;
};

REGISTER_KERNEL_BUILDER(Name("IterativeFarthestPointSample").Device(DEVICE_GPU).TypeConstraint<float>("T"), IterativeFarthestPointSampleOp<float, GpuDevice>);
REGISTER_KERNEL_BUILDER(Name("IterativeFarthestPointSample").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), IterativeFarthestPointSampleOp<Eigen::half, GpuDevice>);
}