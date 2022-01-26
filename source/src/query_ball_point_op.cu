// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#pragma warning(push, 0)
#include <tensorflow/core/util/gpu_device_functions.h>
#pragma warning(pop)

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector>

#include "utils_kernel.h"
#include "utils_common.h"
#include "query_ball_point_op.h"

namespace tensorflow {

template<typename T>
__global__ void queryBallPointKernel(
    const   T*      inPointsPtr,
    const   T*      queryPointsPtr,
            int32*  queriedIndicesPtr,
            int32*  validCountsPtr,
            T       radius,
            int32   b,
            int32   n,
            int32   m,
            int32   nSample
)
{
    auto batch_index = blockIdx.x;
    auto inPointsOffsetPtr = inPointsPtr + batch_index * n * 3;
    auto queryPointsOffsetPtr = queryPointsPtr + batch_index * m * 3;
    auto queriedIndicesOffsetPtr = queriedIndicesPtr + batch_index * m * nSample;
    auto validCountOffsetPtr = validCountsPtr + batch_index * m;
    for (auto i = threadIdx.x; i < m; i += gridDim.x)
    {
        int32 count = 0;
        for (auto k = 0; k < n; k++) {
            if (count == nSample)
                break;
            auto x0 = queryPointsOffsetPtr[i * 3];
            auto y0 = queryPointsOffsetPtr[i * 3 + 1];
            auto z0 = queryPointsOffsetPtr[i * 3 + 2];
            auto xk = inPointsOffsetPtr[k * 3];
            auto yk = inPointsOffsetPtr[k * 3 + 1];
            auto zk = inPointsOffsetPtr[k * 3 + 2];
            auto d = T(sqrtf(float((x0 - xk) * (x0 - xk) + (y0 - yk) * (y0 - yk) + (z0 - zk) * (z0 - zk))));
            if (T(d) >= radius)
                continue;
            if (count == 0)
            {
                for (int l = 0; l < nSample; l++)
                    queriedIndicesOffsetPtr[i * nSample + l] = k;
            }
            queriedIndicesOffsetPtr[i * nSample + count] = k;
            count += 1;
        }
        validCountOffsetPtr[i] = count;
    }
}

template<typename T>
void functor::QueryBallPoint<T, GpuDevice>::operator()(const GpuDevice & d, const Tensor* in_points, const Tensor* query_points, Tensor* queried_indices, Tensor* valid_counts, T radius)
{
    auto batch_size = static_cast<int32>(in_points->dim_size(0));
    auto in_points_size = static_cast<int32>(in_points->dim_size(1));
    auto query_points_size = static_cast<int32>(query_points->dim_size(1));
    auto queried_buffer = static_cast<int32>(queried_indices->dim_size(2));

    auto in_points_ptr = in_points->unaligned_flat<T>().data();
    auto query_points_ptr = query_points->unaligned_flat<T>().data();
    auto queried_indices_ptr = queried_indices->unaligned_flat<int32>().data();
    auto valid_counts_ptr = valid_counts->unaligned_flat<int32>().data();

    queryBallPointKernel<T><<<batch_size, 512>>>(in_points_ptr, query_points_ptr, queried_indices_ptr, valid_counts_ptr, radius, batch_size, in_points_size, query_points_size, queried_buffer);
}

}
#endif