# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Optional

import numpy as np
import tensorflow as tf

from ..nn import interpolate


def compute_gradient_magnitude(gradient_input: tf.Tensor,
                               eps=1e-8,
                               data_format='channels_last',
                               name='GradMagn'):
    with tf.name_scope(name):
        if data_format == 'channels_first':
            gradient_input = tf.transpose(gradient_input, [0, 2, 3, 1])
        # gradient_input = tf.image.rgb_to_grayscale(gradient_input)
        gray_map = np.reshape([0.299, 0.587, 0.114], [1] * (gradient_input.shape.ndims - 1) + [3])
        gray_map = tf.convert_to_tensor(gray_map, dtype=tf.float32)
        gradient_input = tf.reduce_sum(gray_map * gradient_input, axis=-1, keepdims=True)
        gx, gy = tf.image.image_gradients(gradient_input)
        gradient_magnitude = tf.sqrt(tf.square(gx) + tf.square(gy) + eps)
        if data_format == 'channels_first':
            gradient_magnitude = tf.transpose(gradient_input, [0, 3, 1, 2])
        return gradient_magnitude

def generate_image_grid(x_input, scale, bias, name='GenerateGrid'):
    with tf.name_scope(name):
        ndims = x_input.shape.ndims - 2
        b, s, _ = tf.split(tf.shape(x_input), [1, ndims, 1])

        s_up = tf.cast(tf.cast(s, tf.float32) * scale, b.dtype)

        b_dims = bias.shape.ndims
        if b_dims == 1:
            bias = bias[tf.newaxis, tf.newaxis, tf.newaxis, :]
        elif b_dims == 2:
            bias = bias[:, tf.newaxis, tf.newaxis, :]
        elif b_dims == 4:
            pass
        else:
            raise NotImplementedError

        s_axis = [tf.range(a_, dtype=tf.float32) for a_ in tf.unstack(s_up)]
        s_grid = tf.stack(tf.meshgrid(*s_axis, indexing='ij'), axis=-1)
        s_grid = s_grid[tf.newaxis, ...]

        if scale > 1:
            g_bias = 0.5
        elif scale < 1:
            g_bias = -0.25
        else:
            g_bias = 0

        s_grid = (s_grid - g_bias) / scale + bias
        return s_grid

def resize_image(x: tf.Tensor,
                 scale: int,
                 bias: Optional[tf.Tensor] = None,
                 method: str = 'linear',
                 pad: bool = False,
                 name: str = 'ResizeImage'):
    """
    This is a ND-grid uniform linear interpolation function.

    Args:
        x: a `float32` Tensor with shape [N, D1, ..., DM, C]
        scale: a `float32` scalar for the up/down-sample time
        bias: a `float32` Tensor with shape [M] for the sampling offset
        pad: a `bool` scalar whether pads the external points
        name: operator name

    Returns:
        a `float32` Tensor with shape [N, D1 * scale, ..., DN * scale, C]
    """
    assert pad is False
    # scale = tf.convert_to_tensor(scale, tf.float32)
    if bias is None:
        bias = tf.zeros([tf.shape(x)[0], 2], tf.float32)
    else:
        bias = tf.convert_to_tensor(bias, tf.float32)
    with tf.name_scope(name):
        s_grid = generate_image_grid(x, scale, bias)
        return interpolate.grid_sample(x, s_grid, method)
