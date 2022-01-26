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
#include "iterative_farthest_point_sample_op.h"

namespace tensorflow { namespace functor {

template<typename T, int blockSize, int bufferSize>
__global__ void iterativeFarthestPointSampleKernel(
    const   T*      inPointsPtr,
            T*      tmpPointsPtr,
            int32*  outPointsPtr,
            int     b,
            int     n,
            int     m
)
{
    __shared__ T        dists[blockSize];
    __shared__ T        buf[bufferSize * 3];
    __shared__ int      distsIndex[blockSize];

    for (auto i = blockIdx.x; i < b; i += gridDim.x) {
        int32 old = 0;
        if (threadIdx.x == 0)
            outPointsPtr[i * m] = old;
        for (auto j = threadIdx.x; j < n; j += blockDim.x)
            tmpPointsPtr[blockIdx.x * n + j] = T(1e38);
        for (auto j = threadIdx.x; j < min(bufferSize, n) * 3; j += blockDim.x)
            buf[j] = inPointsPtr[i * n * 3 + j];
        __syncthreads();
        for (auto j = 1; j < m; j++) {
            int bestIdx = 0;
            T best = T(-1);
            auto p0Index = i * n * 3 + old * 3;
            T x0 = inPointsPtr[p0Index++];
            T y0 = inPointsPtr[p0Index++];
            T z0 = inPointsPtr[p0Index];
            for (auto k = threadIdx.x; k < n; k += blockDim.x) {
                T td = tmpPointsPtr[blockIdx.x * n + k];
                T xk, yk, zk;
                if (k < bufferSize) {
                    xk = buf[k * 3 + 0];
                    yk = buf[k * 3 + 1];
                    zk = buf[k * 3 + 2];
                }
                else {
                    xk = inPointsPtr[i * n * 3 + k * 3 + 0];
                    yk = inPointsPtr[i * n * 3 + k * 3 + 1];
                    zk = inPointsPtr[i * n * 3 + k * 3 + 2];
                }
                T d = (xk - x0) * (xk - x0) + (yk - y0) * (yk - y0) + (zk - z0) * (zk - z0);
                T d2 = min(d, td);
                if (d2 != td)
                    tmpPointsPtr[blockIdx.x * n + k] = d2;
                if (d2 > best) {
                    best = d2;
                    bestIdx = k;
                }
            }
            dists[threadIdx.x] = best;
            distsIndex[threadIdx.x] = bestIdx;
            __syncthreads();
            for (int u = 0; (1 << u) < blockDim.x; u++) {
                if (threadIdx.x < (blockDim.x >> (u + 1))) {
                    int i1 = (threadIdx.x * 2) << u;
                    int i2 = (threadIdx.x * 2 + 1) << u;
                    if (dists[i1] < dists[i2]) {
                        dists[i1] = dists[i2];
                        distsIndex[i1] = distsIndex[i2];
                    }
                }
                __syncthreads();
            }
            __syncthreads();
            old = distsIndex[0];
            if (threadIdx.x == 0)
                outPointsPtr[i * m + j] = old;
        }
    }
}

template<typename T>
void IterativeFarthestPointSample<T, GpuDevice>::operator()(const GpuDevice & d, const Tensor * in_points, Tensor * tmp_points, Tensor * out_points)
{
    auto batch_size = static_cast<int>(in_points->dim_size(0));
    auto in_points_size = static_cast<int>(in_points->dim_size(1));
    auto out_points_size = static_cast<int>(out_points->dim_size(1));

    auto in_points_ptr = in_points->unaligned_flat<T>().data();
    auto tmp_points_ptr = tmp_points->unaligned_flat<T>().data();
    auto out_points_ptr = out_points->unaligned_flat<int32>().data();
    iterativeFarthestPointSampleKernel<T, 512, 3072><<<32, 512, 0, d.stream()>>>(in_points_ptr, tmp_points_ptr, out_points_ptr, batch_size, in_points_size, out_points_size);
}
} 
}

#endif