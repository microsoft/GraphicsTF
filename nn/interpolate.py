# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from functools import partial
import sys
from typing import Callable, Optional

import tensorflow as tf

from .. import parser
from ..parser import batch_gather_nd, ExportLevel, get_export_level


def regular_grid_sample(feat: tf.Tensor,
                        pts: tf.Tensor,
                        name: str = 'RegularGridSample'):
    """
    This is a regular grid sampling method to receive the N-D indices of points, returning
        the corresponding features on these points

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
        if get_export_level() == ExportLevel.SIMPLIFY:
            return batch_gather_nd(feat, pts, batch_dims=batch_dims)
        else:
            return tf.gather_nd(feat, pts, batch_dims=batch_dims)


def _compute_linear_weight(sample: tf.Tensor, center: tf.Tensor):
    offset_ = center - sample
    weight_ = tf.abs(tf.reduce_prod(offset_, axis=-1, keepdims=True))
    return weight_


def _compute_gaussian_weight(sample: tf.Tensor, center: tf.Tensor,
    sigma: float = 1.):
    offset = center - sample
    dist = tf.reduce_sum(offset * offset, axis=-1, keepdims=True)
    weight = tf.exp(-dist / (2 * sigma * sigma))
    return weight


def _expand_stack_indices(indices):
    expand_list = list([[]])
    for idx in indices:
        tmp_list = list()
        tmp_floor = tf.floor(idx)
        tmp_ceil = tmp_floor + 1
        for e_l in expand_list:
            tmp_list.append(e_l + [tmp_floor])
            tmp_list.append(e_l + [tmp_ceil])
        expand_list = list(tmp_list)
    expand_list = [tf.stack(e, axis=-1) for e in expand_list]
    return expand_list


def interpolate_base(x_input: tf.Tensor,
                     s_pts: tf.Tensor,
                     s_func: Optional[Callable[..., tf.Tensor]],
                     w_func: Optional[Callable[..., tf.Tensor]],
                     name: str = 'LinearNDInterpolation'):
    """
    This is a nd-linear interpolation function to support regular grid (with default `s_func`) and
        other irregular ND-structure (with customize `s_func`).

    Args:
        x: a `float32` Tensor with an arbitrary shape (should be parsed by `s_func`) but given
            channels [..., C]
        s_pts: a `float32` Tensor with shape [M, N + 1], where M is the total points size and 3
            indicates the batch id, ND-positions, respectively.
        s_pts_scale: scale of s_pts
        s_func: a function receives a feature map `x` and a set of points location as inputs,
            returning the exactly features in these points
        s_func_kargs: a dict to save the extra parameters of `s_func`
        name: operation name

    Returns:
        a `float32` Tensor with shape [M, C]
    """
    x_input = tf.convert_to_tensor(x_input)
    s_pts = tf.convert_to_tensor(s_pts)
    with tf.name_scope(name):
        indices = tf.unstack(s_pts, axis=-1)
        expanded_indices = _expand_stack_indices(indices)
        center = tf.stack(indices, axis=-1)

        def _acquire_feat(sample_):
            feat_ = s_func(x_input, tf.cast(sample_, tf.int32))
            return feat_

        intrp_shape = tf.concat([tf.shape(s_pts)[:-1], [tf.shape(x_input)[-1]]],  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                                axis=0)
        intrp_value = tf.zeros(intrp_shape, dtype=x_input.dtype)
        intrp_weight = tf.zeros(intrp_shape[:-1], dtype=x_input.dtype)[..., tf.newaxis]
        for s_f, s_w in zip(expanded_indices, expanded_indices[::-1]):
            weight = w_func(center, s_w)
            intrp_weight += weight
            intrp_value += _acquire_feat(s_f) * weight
        return intrp_value / intrp_weight


def nearest_nd_interpolate(x_input: tf.Tensor,
                           s_pts: tf.Tensor,
                           name: str = 'NearestNDInterpolation'):
    """
    Nearest interpolation
    """
    with tf.name_scope(name):
        s_pts = tf.cast(tf.round(s_pts), tf.int32)
        return regular_grid_sample(x_input, s_pts)


def bicubic_2d_interpolate(x_input: tf.Tensor, s_pts: tf.Tensor, name: str = 'Bicubic'):
    """
    Args :
    input_ : Input tensor. Its shape should be
        [batch_size, height, width, channel].
        In this implementation, the shape should be fixed for speed.
    new_size : The output size [new_height, new_width]
    ref :
    http://blog.demofox.org/2015/08/15/resizing-images-with-bicubic-interpolation/
    """
    with tf.name_scope(name):
        def _hermite(A, B, C, D, t):
            a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
            b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
            c = A * (-0.5) + C * 0.5
            d = B

            return a * t * t * t + b * t * t + c * t + d

        def _apply_offset(_base, _offset):
            return _base + tf.cast(tf.reshape(_offset, [1, 1, 1, -1]), _base.dtype)

        s_dist = s_pts - tf.floor(s_pts)
        i_dist, j_dist = s_dist[..., :1], s_dist[..., 1:]
        s_base = tf.cast(tf.floor(s_pts), tf.int32)

        p_00 = regular_grid_sample(x_input, _apply_offset(s_base, [-1, -1]))
        p_10 = regular_grid_sample(x_input, _apply_offset(s_base, [-1,  0]))
        p_20 = regular_grid_sample(x_input, _apply_offset(s_base, [-1,  1]))
        p_30 = regular_grid_sample(x_input, _apply_offset(s_base, [-1,  2]))

        p_01 = regular_grid_sample(x_input, _apply_offset(s_base, [ 0, -1]))
        p_11 = regular_grid_sample(x_input, _apply_offset(s_base, [ 0,  0]))
        p_21 = regular_grid_sample(x_input, _apply_offset(s_base, [ 0,  1]))
        p_31 = regular_grid_sample(x_input, _apply_offset(s_base, [ 0,  2]))

        p_02 = regular_grid_sample(x_input, _apply_offset(s_base, [ 1, -1]))
        p_12 = regular_grid_sample(x_input, _apply_offset(s_base, [ 1,  0]))
        p_22 = regular_grid_sample(x_input, _apply_offset(s_base, [ 1,  1]))
        p_32 = regular_grid_sample(x_input, _apply_offset(s_base, [ 1,  2]))

        p_03 = regular_grid_sample(x_input, _apply_offset(s_base, [ 2, -1]))
        p_13 = regular_grid_sample(x_input, _apply_offset(s_base, [ 2,  0]))
        p_23 = regular_grid_sample(x_input, _apply_offset(s_base, [ 2,  1]))
        p_33 = regular_grid_sample(x_input, _apply_offset(s_base, [ 2,  2]))

        col0 = _hermite(p_00, p_10, p_20, p_30, j_dist)
        col1 = _hermite(p_01, p_11, p_21, p_31, j_dist)
        col2 = _hermite(p_02, p_12, p_22, p_32, j_dist)
        col3 = _hermite(p_03, p_13, p_23, p_33, j_dist)
        value = _hermite(col0, col1, col2, col3, i_dist)

        return value


linear_interpolate = partial(interpolate_base,
                             s_func=regular_grid_sample,
                             w_func=_compute_linear_weight)
bilinear_interpolate = linear_interpolate
trilinear_interpolate = linear_interpolate

bicubic_interpolate = bicubic_2d_interpolate

gaussian_interpolate = partial(interpolate_base,
                                 s_func=regular_grid_sample,
                                 w_func=_compute_gaussian_weight)

nearest_interpolate = nearest_nd_interpolate


def grid_sample(x_input,
                grid,
                mode='linear',
                padding_mode='zeros',
                align_corners=False):
    """
    Grid sampler
    """
    assert padding_mode == 'zeros'
    assert align_corners is False

    if get_export_level() == ExportLevel.SIMPLIFY:
        return parser.grid_sample(x_input, grid, mode)
    else:
        func_name = f'{mode.lower()}_interpolate'
        if hasattr(sys.modules[__name__], func_name):
            return getattr(sys.modules[__name__], func_name)(x_input, grid)
        else:
            raise NotImplementedError
