// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "utils_kernel.h"
#include "utils_common.h"

#include "scatter_operator_op.h"

namespace tensorflow { namespace functor {

__global__ void k_scatterOperatorSum(
        const int32* indices_ptr,
        const float* feat_ptr,
        float* scattered_ptr,
        float* statistics_ptr,
        int32 length, int32 channel_size,
        int32 threads)
{
    for (auto idx = threadIdx.x + blockDim.x * blockIdx.x; idx < threads; idx += blockDim.x * gridDim.x) {
        auto index = indices_ptr[idx];

        if (index < 0 || index >= length) continue;

        auto feat_based = feat_ptr + channel_size * idx;
        auto scattered_based = scattered_ptr + channel_size * index;

        atomicAdd(statistics_ptr + index, 1);
        for (auto c = 0; c < channel_size; c++) {
            atomicAdd(scattered_based + c, feat_based[c]);
        }
    }
}

__global__ void k_scatterOperatorSumGrad(
        const int32* indices_ptr,
        const float* scattered_grad_ptr,
        float* grad_ptr,
        int32 length, int32 channel_size,
        int32 threads)
{
    for (auto idx = threadIdx.x + blockDim.x * blockIdx.x; idx < threads; idx += blockDim.x * gridDim.x) {
        auto index = indices_ptr[idx];

        if (index < 0 || index >= length) continue;

        auto feat_based = grad_ptr + channel_size * idx;
        auto scattered_based = scattered_grad_ptr + channel_size * index;

        for (auto c = 0; c < channel_size; c++)
            feat_based[c] = scattered_based[c];
    }
}

void ScatterOperator<GpuDevice >::operator()(const GpuDevice &d, const Tensor *indices, const Tensor *feat,
                                             Tensor *scattered, Tensor *statistics, ScatterType type)
{
    auto channel_size = static_cast<int32>(feat->dim_size(2));
    auto threads = static_cast<int32>(feat->dim_size(1));
    auto length = static_cast<int32>(scattered->dim_size(1));

    for (auto batch_idx = 0; batch_idx < indices->dim_size(0); batch_idx++) {
        auto indices_ptr = SliceTensorPtr<int32>(indices, batch_idx);
        auto feat_ptr = SliceTensorPtr<float>(feat, batch_idx);
        auto scattered_ptr = SliceTensorPtr<float>(scattered, batch_idx);
        auto statistics_ptr = SliceTensorPtr<float>(statistics, batch_idx);

        cudaMemset(scattered_ptr, 0, sizeof(float) * length * channel_size);
        cudaMemset(statistics_ptr, 0, sizeof(float) * length);

        switch(type) {
            case ScatterType::SUM:
                k_scatterOperatorSum<<<BLOCKS, THREADS, 0, d.stream()>>>(indices_ptr, feat_ptr, scattered_ptr,
                        statistics_ptr, length, channel_size, threads);
                break;
            default:
                break;
        }
    }
}

void ScatterOperatorGrad<GpuDevice >::operator()(const GpuDevice &d, const Tensor *indices, const Tensor *scattered_feat,
                                                 const Tensor *scattered_grad, Tensor *grad, ScatterType type)
{
    auto channel_size = static_cast<int32>(grad->dim_size(2));
    auto threads = static_cast<int32>(grad->dim_size(1));
    auto length = static_cast<int32>(scattered_grad->dim_size(1));

    for (auto batch_idx = 0; batch_idx < indices->dim_size(0); batch_idx++) {
        auto indices_ptr = SliceTensorPtr<int32>(indices, batch_idx);
        auto scattered_grad_ptr = SliceTensorPtr<float>(scattered_grad, batch_idx);
        auto grad_ptr = SliceTensorPtr<float>(grad, batch_idx);

        cudaMemset(grad_ptr, 0, sizeof(float) * threads * channel_size);

        switch(type) {
            case ScatterType::SUM:
                k_scatterOperatorSumGrad<<<BLOCKS, THREADS, 0, d.stream()>>>(indices_ptr, scattered_grad_ptr,
                        grad_ptr, length, channel_size, threads);
                break;
            default:
                break;
        }
    }
}

};
};

#endif