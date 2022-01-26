# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
from typing import List, Optional, Tuple
from abc import abstractmethod

import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow_graphics.geometry.representation import grid as tfg_grid
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d as tfg_3dr
from tensorflow_graphics.rendering.camera import perspective as tfg_prespective
from ..nn.interpolate import batch_interpolate


class ResampleVoxel(layers.Layer):
    def __init__(self,
                 sample_size: List,
                 sample_cnts: int,
                 near: float,
                 far: float,
                 trainable: bool = False,
                 name: str = None,
                 dtype: tf.dtypes = tf.float32,
                 **kwargs):
        """
        Resample voxel by a specified camera type. This is an abstract class, needed reimplemented by a method --
            generate_mesh_grad
        Args:
            sample_size: the sampled 2D image size
            sample_cnts: the total samples from nearest to farthest
            near: the nearest distance to sample
            far: the farthest distance to sample
            trainable: the flag to make the variable trainable or not
            name: the name of the ops
            dtype: the data type used in tf.Tensor
            **kwargs: None
        """
        super().__init__(trainable, name, dtype, **kwargs)
        self.sample_size = sample_size
        self.sample_cnts = sample_cnts
        self.near = near
        self.far = far

    @abstractmethod
    def generate_mesh_grid(self) -> tf.Tensor:
        pass

    def call(self, inputs=None, **kwargs) -> tf.Tensor:
        mesh_grid = self.generate_mesh_grid()       # 4D-NHW3, the basis sample direction in 3D space
        mesh_grid = mesh_grid[..., tf.newaxis, :]   # 5D-NHW13, the expanded sample direction for multiplier
        sample_stride = tf.linspace(self.near, self.far, self.sample_cnts)  # 1D-C, the basis step for each sample
        sample_stride = sample_stride[tf.newaxis, tf.newaxis, tf.newaxis, ..., tf.newaxis]  # 5D-111C1
        sample_points = mesh_grid * tf.cast(sample_stride, dtype=self.dtype)    # 5D-NHWC3
        return sample_points


class PerspectiveResampleVoxel(ResampleVoxel):
    def __init__(self,
                 sample_size: List,
                 sample_cnts: int,
                 focal_length: List,
                 near: float,
                 far: float,
                 trainable: bool = False,
                 name: str = None,
                 dtype: tf.dtypes = tf.float32,
                 **kwargs):
        """
        Resample voxel for Perspective camera. The coordinate system is Y-top.

        Args:
            sample_size: the sampled 2D image size
            sample_cnts: the total samples from nearest to farthest
            focal_length: focal length of camera
            near: the nearest distance to sample
            far: the farthest distance to sample
            trainable: the flag to make the variable trainable or not
            name: the name of the ops
            dtype: the data type used in tf.Tensor
            **kwargs: None
        """
        super().__init__(sample_size=sample_size, sample_cnts=sample_cnts, near=near, far=far, trainable=trainable,
                         name=name, dtype=dtype, **kwargs)
        self.focal_length = focal_length

    def generate_mesh_grid(self) -> tf.Tensor:
        sensor_coord = tfg_grid.generate([self.sample_size[1] - 1.0, self.sample_size[0] - 1.0],
                                         [0.0, 0.0], [self.sample_size[1], self.sample_size[0]])
        sensor_coord = tf.expand_dims(sensor_coord, axis=0)
        focal = tf.reshape(tf.cast(self.focal_length, dtype=tf.float32), [1, 1, 1, 2])
        principal_point = tf.constant([[[[self.sample_size[1], self.sample_size[0]]]]], dtype=tf.float32) / 2
        local_coord = tfg_prespective.ray(sensor_coord, focal, principal_point)
        return local_coord


class ResampleVoxelBasedOnCamera(layers.Layer):
    def __init__(self,
                 sample_size: List,
                 sample_cnts: int,
                 focal_length: List,
                 viewport: Optional[List],
                 near: float,
                 far: float,
                 trainable: bool = False,
                 name: str = None,
                 dtype: tf.dtypes = tf.float32,
                 **kwargs):
        """
           Resample voxel for rendering

        Args:
            sample_size: the sampled 2D image size
            sample_cnts: the total samples from nearest to farthest
            focal_length: focal length of camera
            viewport: viewport of camera
            near: the nearest distance to sample
            far: the farthest distance to sample
            trainable: the flag to make the variable trainable or not
            name: the name of the ops
            dtype: the data type used in tf.Tensor
            **kwargs: None
        """
        super().__init__(trainable, name, dtype, **kwargs)
        self.focal_length, self.viewport = focal_length, viewport
        self.voxel_sample = PerspectiveResampleVoxel(sample_size, sample_cnts, focal_length, near, far)

    @staticmethod
    def apply_rotation(points, rotation_matrix):
        with tf.compat.v1.name_scope('3DRotation', values=[points, rotation_matrix]):
            # flatten shapes
            flattened_input_points = tf.reshape(points, [points.shape.dims[0].value, -1, 3])

            transposed_points = tf.transpose(flattened_input_points, perm=[0, 2, 1])  # [1, 3, k]
            rotated_points = tf.matmul(rotation_matrix, transposed_points)  # [n, 3, k]
            out_flattened_points = tf.transpose(rotated_points, perm=[0, 2, 1])  # [n, k, 3]

            batch_size = max(rotation_matrix.shape.dims[0].value, points.shape.dims[0].value)
            out_points = tf.reshape(out_flattened_points, [batch_size, *points.shape.as_list()[1:]])
            return out_points

    @staticmethod
    def xyz_obj2vol_mapping(points_xyz, cube_size, vol_shape):
        points_xyz_vol = points_xyz * cube_size
        points_x, points_y, points_z = tf.unstack(points_xyz, axis=-1)
        x_mask = tf.logical_and(tf.greater_equal(points_x, 0), tf.less_equal(points_x, vol_shape[0] / cube_size))
        y_mask = tf.logical_and(tf.greater_equal(points_y, 0), tf.less_equal(points_y, vol_shape[1] / cube_size))
        z_mask = tf.logical_and(tf.greater_equal(points_z, 0), tf.less_equal(points_z, vol_shape[2] / cube_size))
        points_mask = tf.expand_dims(tf.logical_and(tf.logical_and(x_mask, y_mask), z_mask), axis=-1)

        return points_xyz_vol, points_mask

    def sample_points(self, vol_shape, obj_r, obj_t, cam_r, cam_t):
        sampled_points_cam = self.voxel_sample(inputs=None)  # 1whd3

        if cam_r is not None:
            cam_r = tf.matmul(cam_r, tf.convert_to_tensor([[[0, 0, 1], [0, 1, 0], [-1, 0, 0]]], tf.float32))
            sampled_points_cam = self.apply_rotation(sampled_points_cam, cam_r)
        cam_t = cam_t if cam_t is None else cam_t[:, tf.newaxis, tf.newaxis, tf.newaxis, :]  # 5D-N1113
        sampled_points_world = sampled_points_cam if cam_t is None else sampled_points_cam + cam_t

        obj_t = obj_t if obj_t is None else obj_t[:, tf.newaxis, tf.newaxis, tf.newaxis, :]  # 5D-N1113
        sampled_points_obj = sampled_points_world if obj_t is None else sampled_points_world - obj_t
        obj_r = obj_r if obj_r is None else tfg_3dr.inverse(obj_r)
        sampled_points_obj = sampled_points_obj if obj_r is None else self.apply_rotation(sampled_points_obj, obj_r)

        trans_perm = [0, 3, 1, 2, 4] if self.focal_length is None and self.viewport is None else [0, 3, 2, 1, 4]
        sampled_points_obj = tf.transpose(sampled_points_obj, perm=trans_perm)

        sampled_points_vol, points_mask = self.xyz_obj2vol_mapping(sampled_points_obj, vol_shape[1], vol_shape[1:4])

        return sampled_points_vol, points_mask

    def call(self, inputs=None, **kwargs) -> tf.Tensor:
        if len(inputs) == 3:
            vol_data, cam_r, cam_t = tuple(inputs)
            obj_r, obj_t = (None, None)
        elif len(inputs) == 4:
            vol_data, cam_r, cam_t, _ = tuple(inputs)
            obj_r, obj_t = (None, None)
        elif len(inputs) == 5:
            vol_data, obj_r, obj_t, cam_r, cam_t = tuple(inputs)
        else:
            raise NotImplementedError

        vol_shape = vol_data.shape.as_list()
        sampled_points_vol, points_mask = self.sample_points(vol_shape, obj_r, obj_t, cam_r, cam_t)

        vol_data_pad = tf.pad(vol_data, tf.constant([[0, 0], *[[1, 1]] * len(vol_shape[1:-1]), [0, 0]]), 'REFLECT')
        pts_value_range = [[0] * len(vol_shape[1:-1]), vol_shape[1:-1]]
        sampled_vol = batch_interpolate(vol_data_pad, sampled_points_vol, pts_value_range, 0.5, 'nearest')
        # sampled_vol = batch_interpolate(vol_data_pad, sampled_points_vol, pts_value_range, 0.5, 'linear')

        empty_vol = tf.cast(tf.one_hot(tf.zeros(sampled_vol.shape[:-1], tf.uint8), vol_shape[-1]), tf.float32)
        vol_mask = tf.tile(points_mask, [*[1] * (sampled_vol.shape.ndims - 1), sampled_vol.shape[-1]])
        masked_vol = tf.where(vol_mask, sampled_vol, empty_vol)

        return masked_vol
