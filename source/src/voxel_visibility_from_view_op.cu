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
#include "voxel_visibility_from_view_op.h"

namespace tensorflow { namespace functor
{
template <typename T>
__global__ void voxelVisibilityFromViewKernel(
	const	T*		maskVoxelPtr,
	const	float*	camPosePtr,
			T*		visibilityVoxelPtr,
			int		batch,
			int		width,
			int		height,
			int		depth,
			int		numThreads
)
{
	for (auto tid = threadIdx.x + blockIdx.x * blockDim.x; tid < numThreads; tid += blockDim.x * gridDim.x)
	{
		visibilityVoxelPtr[tid] = 0;
		int d = tid % depth;
		int h_s = depth;
		int h = (tid / h_s) % height;
		auto w_s = depth * height;
		int w = (tid / w_s) % width;
		auto b_s = depth * height * width;
		int b = (tid / b_s) % batch;  // just for aligment, no necessary to floor divide batch size

		auto &cam_x = camPosePtr[b * 3];
		auto &cam_y = camPosePtr[b * 3 + 1];
		auto &cam_z = camPosePtr[b * 3 + 2];

		float sample_x[8] = {0.01, 0.01, 0.01, 0.01, 0.99, 0.99, 0.99, 0.99};
		float sample_y[8] = {0.01, 0.01, 0.99, 0.99, 0.01, 0.01, 0.99, 0.99};
		float sample_z[8] = {0.01, 0.99, 0.01, 0.99, 0.01, 0.99, 0.01, 0.99};
		for (auto s = 0; s < 8; s++)
		{
			auto v_x = (float)w + sample_x[s];
			auto v_y = (float)h + sample_y[s];
			auto v_z = (float)d + sample_z[s];

			auto step_x = cam_x - v_x;
			auto step_y = cam_y - v_y;
			auto step_z = cam_z - v_z;
			auto step_dist = sqrtf(step_x * step_x + step_y * step_y + step_z * step_z);
			if (step_dist < 1) {
				auto tmp_mask = maskVoxelPtr[tid] == 1 ? 1 : 2;
				visibilityVoxelPtr[tid] = visibilityVoxelPtr[tid] != 1 ? tmp_mask : 1;
				continue;
			}
			step_x = step_x / step_dist;
			step_y = step_y / step_dist;
			step_z = step_z / step_dist;
			auto pass = true;
			for (int s = 0; s < (int)step_dist; s++)
			{
				v_x += step_x;
				v_y += step_y;
				v_z += step_z;
				auto v_w = (int)floorf(v_x);
				auto v_h = (int)floorf(v_y);
				auto v_d = (int)floorf(v_z);
				auto vid = b * b_s + v_w * w_s + v_h * h_s + v_d;
				if (maskVoxelPtr[vid] == 1) {
					pass = false;
					break;
				}
			}
			if (pass) {
				auto tmp_mask = maskVoxelPtr[tid] == 1 ? 1 : 2;
				visibilityVoxelPtr[tid] = visibilityVoxelPtr[tid] != 1 ? tmp_mask : 1;
			}
		}
		return;
	}
}

template <typename T>
void VoxelVisibilityFromView<T, GpuDevice>::operator()(const GpuDevice & d, const Tensor * mask_voxel, const Tensor * cam_pos, Tensor * visibility_voxel)
{
	auto batch_size = static_cast<int>(mask_voxel->dim_size(0));
	auto voxel_w = static_cast<int>(mask_voxel->dim_size(1));
	auto voxel_h = static_cast<int>(mask_voxel->dim_size(2));
	auto voxel_d = static_cast<int>(mask_voxel->dim_size(3));
	auto threads = batch_size * voxel_w * voxel_h * voxel_d;

	auto mask_voxel_ptr = mask_voxel->unaligned_flat<T>().data();
	auto cam_pose_ptr = cam_pos->unaligned_flat<float>().data();
	auto visibility_voxel_ptr = visibility_voxel->unaligned_flat<T>().data();

	voxelVisibilityFromViewKernel<T><<<BLOCKS, THREADS, 0, d.stream()>>>(mask_voxel_ptr, cam_pose_ptr, visibility_voxel_ptr,
		batch_size, voxel_w, voxel_h, voxel_d, threads);
}
} 
}

#endif