# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Union, Callable, Dict, TypeVar, Optional, List

import itertools
import tensorflow as tf
import numpy as np

TF_EXECUTABLE_ND_ARRAY = TypeVar('TF_EXECUTABLE_ND_ARRAY', tf.Tensor,
                                 np.ndarray)


def interpolate(x: tf.Tensor, y: tf.Tensor, beta: Union[tf.Tensor,
                                                        float]) -> tf.Tensor:
    beta = tf.convert_to_tensor(beta)
    with tf.compat.v1.name_scope('Interpolation', values=[x, y, beta]):
        return x + (y - x) * beta


def regular_grid_sample(feat: TF_EXECUTABLE_ND_ARRAY,
                        pts: TF_EXECUTABLE_ND_ARRAY,
                        name: str = 'regular_grid_sample'):
    """
    This is a regular grid sampling method to receive the N-D indices of points, returning the corresponding features on
        these points

    Args:
        feat: a `float32` Tensor with shape [1, ..., N, C], the sampled feature map
        pts: a `int32` Tensor with shape [M, N], the indices map to sample on `feat`
        name: operation name

    Returns:
        a `float32` Tensor with shape [M, C]
    """
    feat = tf.convert_to_tensor(feat)
    pts = tf.convert_to_tensor(pts)
    batch_dims = len(feat.shape) - pts.shape.as_list()[-1] - 1
    with tf.name_scope(name=name):
        return tf.gather_nd(feat, pts, batch_dims=batch_dims)


def floor_and_ceiling(x):
    floor = tf.floor(x)
    ceiling = floor + 1
    return floor, ceiling


def expand_stack_indices(indices):
    expand_list = list([[]])
    for idx in indices:
        tmp_list = list()
        tmp_floor, tmp_ceil = floor_and_ceiling(idx)
        for e_l in expand_list:
            tmp_list.append(e_l + [tmp_floor])
            tmp_list.append(e_l + [tmp_ceil])
        expand_list = list(tmp_list)
    expand_list = [tf.stack(e, axis=-1) for e in expand_list]
    return expand_list


def linear_nd_interpolate(x: TF_EXECUTABLE_ND_ARRAY,
                          s_pts: TF_EXECUTABLE_ND_ARRAY,
                          s_pts_scale: int = 1,
                          s_func: Optional[Callable[..., tf.Tensor]] = None,
                          s_func_kargs: Optional[Dict] = None,
                          name: str = 'linear_nd_interpolate'):
    """
    This is a nd-linear interpolation function to support regular grid (with default `s_func`) and other irregular
        ND-structure (with customize `s_func`).

    Args:
        x: a `float32` Tensor with an arbitrary shape (should be parsed by `s_func`) but given channels [..., C]
        s_pts: a `float32` Tensor with shape [M, N + 1], where M is the total points size and 3 indicates the batch id,
            ND-positions, respectively.
        s_pts_scale: scale of s_pts
        s_func: a function receives a feature map `x` and a set of points location as inputs, returning the exactly
            features in these points
        s_func_kargs: a dict to save the extra parameters of `s_func`
        name: operation name

    Returns:
        a `float32` Tensor with shape [M, C]
    """
    if s_func is None:
        s_func = regular_grid_sample
    if s_func_kargs is None:
        s_func_kargs = dict()
    x = tf.convert_to_tensor(x)
    s_pts = tf.convert_to_tensor(s_pts)
    with tf.name_scope(name):
        indices = tf.unstack(s_pts, axis=-1)
        indices[1:] = [pts * s_pts_scale for pts in indices[1:]]
        expanded_indices = expand_stack_indices(indices[1:])
        center = tf.stack(indices[1:], axis=-1)

        def _acquire_feat(sample_):
            feat_ = s_func(x, tf.cast(tf.concat([indices[0][..., tf.newaxis], sample_], axis=-1), tf.int32),
                           **s_func_kargs)
            return feat_

        def _acquire_weight(sample_):
            offset_ = center - sample_
            weight_ = tf.abs(tf.reduce_prod(offset_, axis=-1, keepdims=True))
            return weight_

        intrp_shape = tf.concat([tf.shape(s_pts)[:-1], [tf.shape(x)[-1]]],
                                axis=0)
        intrp_value = tf.zeros(intrp_shape, dtype=x.dtype)
        for s_f, s_w in zip(expanded_indices, expanded_indices[::-1]):
            intrp_value += _acquire_feat(s_f) * _acquire_weight(s_w)
        return intrp_value


def batch_interpolate(x: tf.Tensor,
                      s_pts: tf.Tensor,
                      pts_value_range: List,
                      pts_offset: float,
                      interpolate_method: str,
                      name: str = None):
    """
    batch nearest interpolation function

    Args:
        x: a `float32` Tensor with an arbitrary shape (should be parsed by `s_func`) but given channels [N, ..., C]
        s_pts: a `float32` Tensor with shape [N, ..., D], where D = x.shape.ndims - 2
        pts_value_range: points position range
        pts_offset: points position offset
        interpolate_method: interpolate method, 'linear' or 'nearest'
        name: operation name

    Returns:
        a `float32` Tensor with shape [..., C]
    """
    with tf.compat.v1.name_scope(name, default_name='BatchInterpolation', values=[x, s_pts]):
        x_shape = x.shape.as_list()
        clipped_pts = tf.clip_by_value(s_pts, pts_value_range[0], pts_value_range[1])
        batch_range = tf.reshape(tf.range(x_shape[0], dtype=tf.int32), [x_shape[0], *[1]*(len(x_shape) - 1)])
        batch_indices = tf.tile(batch_range, [1, *s_pts.shape.as_list()[1:-1], 1])
        if interpolate_method == 'linear':
            batch_pts = tf.concat((tf.cast(batch_indices, tf.float32), clipped_pts + pts_offset), axis=-1)
            interpolated_label = linear_nd_interpolate(x, tf.reshape(batch_pts, [-1, len(x_shape)-1]))
            interpolated = tf.reshape(interpolated_label, [*s_pts.shape.as_list()[:-1], x_shape[-1]])
        elif interpolate_method == 'nearest':
            rounded_pts = tf.cast(tf.round(clipped_pts + pts_offset), tf.int32)
            batch_pts = tf.concat((batch_indices, rounded_pts), axis=-1)
            interpolated = tf.gather_nd(x, batch_pts)
        else:
            raise NotImplementedError
        return interpolated


class TestInterpolation(tf.test.TestCase):
    def __init__(self, method_name='TestInterpolation'):
        super().__init__(methodName=method_name)

    @staticmethod
    def gen_test_pairs(test_shape, channel_num=4, batch_size=6, offset=0.5):
        test_size = np.asarray(test_shape)
        feat_map = np.random.rand(
            *[batch_size, *test_size.tolist(), channel_num]).astype(np.float32)
        interp_nd = [
            np.arange(0, t_s - 1).astype(np.float32) + offset
            for t_s in test_size
        ]
        interp = np.meshgrid(
            *[np.arange(batch_size).astype(np.float32), *interp_nd],
            indexing='ij')
        interp = np.stack(interp, axis=-1)
        interp_shape = interp.shape
        interp = np.reshape(interp, [-1, len(test_shape) + 1])
        return feat_map, interp, interp_shape

    @staticmethod
    def mean_nd_tensor(arr, axis):
        sls = [slice(0, -1), slice(1, None)]
        sls_list = [i for i in itertools.product(*([sls] * len(axis)))]
        new_arr_shape = np.array(arr.shape)
        new_arr_shape[1:-1] = new_arr_shape[1:-1] - 1
        new_arr = np.zeros(new_arr_shape)
        for s_l in sls_list:
            assert len(s_l) == 2 or len(s_l) == 3
            x, y = s_l[:2]
            new_arr += arr[:, x, y, ...] if len(s_l) == 2 else arr[:, x, y,
                                                                   s_l[2], ...]
        return new_arr / len(sls_list)

    def testLinearInterpolate(self):
        offset = 0.25
        feat_map, interp, interp_shape = self.gen_test_pairs([8],
                                                             offset=offset)
        interp_native = tf.reshape(linear_nd_interpolate(feat_map, interp), [*interp_shape[:-1], -1])
        interp_ref = np.array(feat_map[:, :-1, ...]) * (1 - offset)
        interp_ref += feat_map[:, 1:, ...] * offset
        self.assertAllClose(interp_ref, interp_native.numpy())

    def testBilinearInterpolate(self):
        feat_map, interp, interp_shape = self.gen_test_pairs([8, 8])
        interp_native = tf.reshape(linear_nd_interpolate(feat_map, interp), [*interp_shape[:-1], -1])
        interp_ref = self.mean_nd_tensor(feat_map, [1, 2])
        self.assertAllClose(interp_ref, interp_native.numpy())

    def testTrilinearInterpolate(self):
        feat_map, interp, interp_shape = self.gen_test_pairs([8, 8, 8])
        interp_native = tf.reshape(linear_nd_interpolate(feat_map, interp), [*interp_shape[:-1], -1])
        interp_ref = self.mean_nd_tensor(feat_map, [1, 2, 3])
        self.assertAllClose(interp_ref, interp_native.numpy())
