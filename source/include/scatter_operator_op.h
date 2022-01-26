// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef SCATTER_OPERATOR_OPS_H_
#define SCATTER_OPERATOR_OPS_H_

#pragma warning(push, 0)
#include <unsupported/Eigen/CXX11/Tensor>
#include <tensorflow/core/framework/tensor.h>
#pragma warning(pop)

#include "utils_common.h"

namespace tensorflow{

enum ScatterType{
    SUM=0,
    MAX=1
};

namespace functor {

template <typename Device>
struct ScatterOperator {
    virtual void operator()(const Device& d, const Tensor* indices, const Tensor* feat, Tensor* scattered,
                            Tensor* statistics, ScatterType type)=0;
};

template <typename Device>
struct ScatterOperatorGrad {
    virtual void operator()(const Device& d, const Tensor* indices, const Tensor* scattered_feat,
                            const Tensor* scattered_grad, Tensor* grad, ScatterType type)=0;
};

template <>
struct ScatterOperator<GpuDevice >{
    virtual void operator()(const GpuDevice &d, const Tensor* indices, const Tensor* feat, Tensor* scattered,
                            Tensor* statistics, ScatterType type);
};

template <>
struct ScatterOperatorGrad<GpuDevice >{
    virtual void operator()(const GpuDevice& d, const Tensor* indices, const Tensor* scattered_feat,
                            const Tensor* scattered_grad, Tensor* grad, ScatterType type);
};

}
}

#endif