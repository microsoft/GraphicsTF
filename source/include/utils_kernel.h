// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef KERNEL_UTILS_H_
#define KERNEL_UTILS_H_
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#pragma warning(push, 0)
#include <tensorflow/core/framework/tensor.h>
#pragma warning(pop)

#define THREADS 256
#define BLOCKS 256

template <typename T>
__device__ bool k_within_clip(T target, T maximum, T minimum)
{
    return target >= minimum && target < maximum;
}

template <typename T>
__device__ T k_distance(const T* a, const T* b, int size)
{
    T distance = T(0);
    for (auto idx = 0; idx < size; idx++)
        distance += (a[idx] - b[idx]) * (a[idx] - b[idx]);
    distance = sqrtf(distance);
    return distance;
}

template <typename T>
__device__ T k_l2(const T* a, int size)
{
    T distance = T(0);
    for (auto idx = 0; idx < size; idx++)
        distance += powf(a[idx], 2);
    distance = sqrtf(distance);
    return distance;
}


template <typename T>
__device__ T k_sum(const T* a, int size)
{
    T sum = T(0);
    for (auto idx = 0; idx < size; idx++)
        sum += a[idx];
    return sum;
}

template <typename T>
__device__ void k_assign(T* a, T value, int size)
{
    for (auto idx = 0; idx < size; idx++)
        a[idx] = value;
}

template <typename T>
__device__ void k_assign_weight_add(T* a, const T* b, const T c, int size)
{
    for (auto idx = 0; idx < size; idx++)
        a[idx] += b[idx] * c;
}

template <typename T>
__device__ void k_assign_weight_add_locked(T* a, const T* b, const T c, int size)
{
    for (auto idx = 0; idx < size; idx++)
        atomicAdd(a + idx, b[idx] * c);
}

template <typename T>
__global__ void k_memset(T* array, T value, int threads)
{
    for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < threads; idx += blockDim.x * gridDim.x)
        array[idx] = value;
}

template <typename T, unsigned int blockSize>
__device__ void warpReduce(volatile T *s_data, unsigned int tid)
{
    if (blockSize >=  64) s_data[tid] = s_data[tid] + s_data[tid + 32];
    if (blockSize >=  32) s_data[tid] = s_data[tid] + s_data[tid + 16];
    if (blockSize >=  16) s_data[tid] = s_data[tid] + s_data[tid +  8];
    if (blockSize >=   8) s_data[tid] = s_data[tid] + s_data[tid +  4];
    if (blockSize >=   4) s_data[tid] = s_data[tid] + s_data[tid +  2];
    if (blockSize >=   2) s_data[tid] = s_data[tid] + s_data[tid +  1];
}

template <typename T, unsigned int blockSize>
__device__ void reduce6(T *s_data)
{
    int tid = threadIdx.x;
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            s_data[tid] += s_data[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128)
        {
            s_data[tid] += s_data[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid <  64)
        {
            s_data[tid] += s_data[tid +  64];
        }
        __syncthreads();
    }
    if (blockSize >= 64) {
        if (tid < 32)
        {
            s_data[tid] += s_data[tid + 32];
        }
        __syncthreads();
    }
    //if (tid < 32)
    //    warpReduce<T, blockSize>(s_data, tid);
    if (blockSize >= 32) {
        if (tid < 16)
        {
            s_data[tid] += s_data[tid + 16];
        }
        __syncthreads();
    }
    if (blockSize >= 16) {
        if (tid < 8)
        {
            s_data[tid] += s_data[tid + 8];
        }
        __syncthreads();
    }
    if (blockSize >= 8) {
        if (tid < 4)
        {
            s_data[tid] += s_data[tid + 4];
        }
        __syncthreads();
    }
    if (blockSize >= 4) {
        if (tid < 2)
        {
            s_data[tid] += s_data[tid + 2];
        }
        __syncthreads();
    }
    if (blockSize >= 2) {
        if (tid < 1)
        {
            s_data[tid] += s_data[tid + 1];
        }
        __syncthreads();
    }
}

template __global__ void k_memset<float>(float* a, float c, int size);
template __global__ void k_memset<Eigen::half>(Eigen::half* a, Eigen::half c, int size);

#endif