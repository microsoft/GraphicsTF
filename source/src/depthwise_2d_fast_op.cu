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
#include "depthwise_2d_fast_op.h"

namespace tensorflow { namespace functor {

template<typename Dtype>
__global__ void k_ConvolutionDepthWiseForward(
        const Dtype* bottom_data,
        const Dtype* weight_data,
        const int channels,
        const int top_height, const int top_width,
        const int bottom_height, const int bottom_width,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        Dtype *top_data,
        const int threads
)
{
    for (auto index = threadIdx.x + blockDim.x * blockIdx.x;
         index < threads; index += blockDim.x * gridDim.x) {
        const int n = (index / channels / top_height / top_width);
        const int h = (index / channels / top_width) % top_height;
        const int w = (index / channels % top_width);
        const int c = (index % channels);
        //const Dtype *weight = weight_data + c * kernel_h * kernel_w;
        Dtype value = Dtype(0);
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int h_in = -pad_h + h + kh;
                const int w_in = -pad_w + w + kw;
                if ((h_in >= 0) && (h_in < bottom_height)
                    && (w_in >= 0) && (w_in < bottom_width)) {
                    const int d_offset = ((n * bottom_height + h_in) * bottom_width + w_in)
                                       * channels + c;
                    const int w_offset = (kh * kernel_w + kw) * channels + c;
                    value += weight_data[w_offset] * bottom_data[d_offset];
                }
            }
        }
        top_data[index] = value;
    }
}

template<typename Dtype>
__global__ void k_ConvolutionDepthWiseBackwardFilter(
        const Dtype *const top_diff,
        const Dtype *const bottom_data,
        const int channels,
        const int top_height, const int top_width,
        const int bottom_height, const int bottom_width,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        Dtype* filter_diff,
        const int threads
)
{
    for (auto index = threadIdx.x + blockDim.x * blockIdx.x; index < threads; index += blockDim.x * gridDim.x) {
        const int n = index / channels / bottom_height / bottom_width;
        const int c = (index / bottom_height / bottom_width) % channels;
        const int h = (index / bottom_width) % bottom_height;
        const int w = index % bottom_width;
        __shared__ Dtype reduce_weights[THREADS];
        const int b_offset = ((n * bottom_height + h) * bottom_width + w) * channels + c;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int h_out_s = h + pad_h - kh;
                const int w_out_s = w + pad_w - kw;
                const int h_out = h_out_s;
                const int w_out = w_out_s;
                if ((h_out >= 0) && (h_out < top_height)
                    && (w_out >= 0) && (w_out < top_width)) {
                    const int t_offset = ((n * top_height + h_out) * top_width + w_out)
                                       * channels + c;
                    reduce_weights[threadIdx.x] = bottom_data[b_offset] * top_diff[t_offset];
                }
                else{
                    reduce_weights[threadIdx.x] = Dtype(0);
                }
                __syncthreads();
                reduce6<Dtype, THREADS>(reduce_weights);
                if (threadIdx.x == 0)
                    GpuAtomicAdd(filter_diff + (kh * kernel_w + kw) * channels + c, reduce_weights[0]);
                __syncthreads();
            }
        }
    }
}

template<typename Dtype>
__global__ void k_ConvolutionDepthWiseBackwardInput(
        const Dtype *const top_diff,
        const Dtype *const weight_data,
        const int channels,
        const int top_height, const int top_width,
        const int bottom_height, const int bottom_width,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        Dtype* bottom_diff,
        const int threads
)
{
    for (auto index = threadIdx.x + blockDim.x * blockIdx.x;
         index < threads; index += blockDim.x * gridDim.x) {
        const int n = (index / channels / bottom_height / bottom_width);
        const int h = (index / channels / bottom_width) % bottom_height;
        const int w = (index / channels % bottom_width);
        const int c = (index % channels);
        Dtype value = Dtype(0);
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int h_out_s = h + pad_h - kh;
                const int w_out_s = w + pad_w - kw;
                const int h_out = h_out_s;
                const int w_out = w_out_s;
                if ((h_out >= 0) && (h_out < top_height)
                    && (w_out >= 0) && (w_out < top_width)) {
                    const int d_offset = ((n * top_height + h_out) * top_width + w_out)
                                       * channels + c;
                    const int w_offset = (kh * kernel_w + kw) * channels + c;
                    value += weight_data[w_offset] * top_diff[d_offset];
                }
            }
        }
        bottom_diff[index] += value;
    }
}

template <typename T>
void Depthwise2DFast<T, GpuDevice>::operator()(const GpuDevice &d, const Tensor* input, const Tensor* filter,
        Tensor* output) {
    auto batch_size = static_cast<int32>(input->dim_size(0));
    auto input_channels = static_cast<int32>(input->dim_size(3));
    auto input_height = static_cast<int32>(input->dim_size(1));
    auto input_width = static_cast<int32>(input->dim_size(2));
    auto filter_height = static_cast<int32>(filter->dim_size(0));
    auto filter_width = static_cast<int32>(filter->dim_size(1));
    auto pad_height = filter_height / 2;
    auto pad_width = filter_width / 2;
    auto threads = batch_size * input_channels * input_height * input_width;
    k_ConvolutionDepthWiseForward<T><<<BLOCKS, THREADS, 0, d.stream()>>>(
            input->unaligned_flat<T>().data(), filter->unaligned_flat<T>().data(),
            input_channels, input_height, input_width, input_height, input_width, filter_height, filter_width,
            pad_height, pad_width, output->unaligned_flat<T>().data(), threads);
}

template<typename T>
void Depthwise2DFastGradFilter<T, GpuDevice>::operator()(const GpuDevice& d, const Tensor* output_diff,
        const Tensor* input, Tensor* filter_diff)
{
    auto batch_size = static_cast<int32>(output_diff->dim_size(0));
    auto input_channels = static_cast<int32>(output_diff->dim_size(3));
    auto input_height = static_cast<int32>(output_diff->dim_size(1));
    auto input_width = static_cast<int32>(output_diff->dim_size(2));
    auto filter_height = static_cast<int32>(filter_diff->dim_size(0));
    auto filter_width = static_cast<int32>(filter_diff->dim_size(1));
    auto pad_height = filter_height / 2;
    auto pad_width = filter_width / 2;
    auto threads = batch_size * input_channels * input_height * input_width;
//    VLOG(0) << "Threads: [" << threads << "], "
//            << "Output Diff: [" << batch_size << ", " << input_channels << ", " << input_height << ", " << input_width << "], "
//            << "Pad: [" << pad_height << ", " << pad_width << "], "
//            << "Filter: [" << static_cast<int32>(filter_diff->dim_size(1)) << ", " << filter_height << ", " << filter_width << "]";
    auto filter_size = filter_height * filter_width * input_channels;
    k_memset<T><<<BLOCKS, THREADS, 0, d.stream()>>>(filter_diff->unaligned_flat<T>().data(), T(0), filter_size);
    k_ConvolutionDepthWiseBackwardFilter<T><<<BLOCKS, THREADS, 0, d.stream()>>>(
            output_diff->unaligned_flat<T>().data(), input->unaligned_flat<T>().data(),
            input_channels, input_height, input_width, input_height, input_width, filter_height, filter_width,
            pad_height, pad_width, filter_diff->unaligned_flat<T>().data(), threads);
}

template <typename T>
void Depthwise2DFastGradInput<T, GpuDevice>::operator()(const GpuDevice& d, const Tensor* output_diff,
        const Tensor* filter, Tensor* input_diff)
{
    auto batch_size = static_cast<int32>(output_diff->dim_size(0));
    auto input_channels = static_cast<int32>(output_diff->dim_size(3));
    auto input_height = static_cast<int32>(output_diff->dim_size(1));
    auto input_width = static_cast<int32>(output_diff->dim_size(2));
    auto filter_height = static_cast<int32>(filter->dim_size(0));
    auto filter_width = static_cast<int32>(filter->dim_size(1));
    auto pad_height = filter_height / 2;
    auto pad_width = filter_width / 2;
    auto threads = batch_size * input_channels * input_height * input_width;
//    VLOG(0) << "Threads: [" << threads << "], "
//            << "Output Diff: [" << batch_size << ", " << input_channels << ", " << input_height << ", " << input_width << "], "
//            << "Pad: [" << pad_height << ", " << pad_width << "], "
//            << "Filter: [" << filter_height << ", " << filter_width << "]";
    k_memset<T><<<BLOCKS, THREADS, 0, d.stream()>>>(input_diff->unaligned_flat<T>().data(), T(0), threads);
    k_ConvolutionDepthWiseBackwardInput<T><<<BLOCKS, THREADS, 0, d.stream()>>>(
            output_diff->unaligned_flat<T>().data(), filter->unaligned_flat<T>().data(),
            input_channels, input_height, input_width, input_height, input_width, filter_height, filter_width,
            pad_height, pad_width, input_diff->unaligned_flat<T>().data(), threads);
}

}
}

#endif
