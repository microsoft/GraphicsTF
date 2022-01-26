# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Optional, List, Callable, Dict

import logging
import tensorflow as tf
import numpy as np


def softmax(x, axis=-1):
    x = x - tf.stop_gradient(tf.reduce_max(x, axis=[r for r in range(1, x.shape.ndims)], keepdims=True))
    x = tf.nn.softmax(x, axis=axis)
    return x


def format_check(t: Optional[tf.Tensor],
                 data_format: str,
                 allow_all: bool = False) -> str:
    if t is not None:
        if t.shape.ndims == 5 and len(data_format) != 5:
            data_format = data_format.replace('HW', 'DHW')
        if t.shape.ndims == 4 and len(data_format) != 4:
            data_format = data_format.replace('DHW', 'HW')
    if not allow_all:
        assert data_format.find('C') != 1
    return data_format


def expand_dims(x: tf.Tensor,
                axis: List,
                name: Optional[str] = None) -> tf.Tensor:
    with tf.name_scope(name, default_name='ExpandDims', values=[x]):
        for a in axis:
            x = tf.expand_dims(x, a)
        return x


def convert_to_channel_first(x: tf.Tensor,
                             name: Optional[str] = None) -> tf.Tensor:
    with tf.name_scope(name, default_name='ConvertToChannelFirst', values=[x]):
        feat_dims = np.arange(x.shape.ndims - 2) + 1
        x_trans = tf.transpose(x, [0, x.shape.ndims - 1, *feat_dims])
        return x_trans


def convert_to_channel_last(x: tf.Tensor,
                            name: Optional[str] = None) -> tf.Tensor:
    with tf.name_scope(name, default_name='ConvertToChannelLast', values=[x]):
        feat_dims = np.arange(x.shape.ndims - 2) + 2
        x_trans = tf.transpose(x, [0, *feat_dims, 1])
        return x_trans


def group_kernel(x: tf.Tensor,
                 w: tf.Tensor,
                 group: int,
                 conv_func: Callable,
                 conv_kargs: Dict):
    xs_g = tf.split(x, group, axis=-1)
    ws_g = tf.split(w, group, axis=-1)
    x = tf.concat([conv_func(xs, ws, **conv_kargs) for xs, ws in zip(xs_g, ws_g)], axis=-1)
    return x


def group_conv(x: tf.Tensor,
               w: tf.Tensor,
               strides: List,
               padding: str = 'SAME',
               data_format: str = 'NHWC',
               disable_3d: bool = True,
               name: Optional[str] = None):
    data_format = format_check(x, data_format)
    with tf.name_scope(name, default_name='GroupConv', values=[x, w]):
        conv_kargs = dict(data_format=data_format, strides=strides, padding=padding)
        conv_func = getattr(tf.nn, f'conv{x.shape.ndims - 2}d')
        group = x.shape[-1].value // w.shape[-2].value
        if group > 1 and disable_3d and x.shape.ndims == 5:
            logging.info('Unsupported official group convolution, using native implementation')
            x = group_kernel(x, w, group, conv_func, conv_kargs)
        else:
            x = conv_func(x, w, **conv_kargs)
        return x


def group_transpose_conv(x: tf.Tensor,
                         w: tf.Tensor,
                         strides: List,
                         out_shape: List,
                         padding: str = 'SAME',
                         data_format: str = 'NHWC',
                         disable_3d: bool = True,
                         name: Optional[str] = None):
    data_format = format_check(x, data_format)
    with tf.name_scope(name, default_name='GroupTransConv', values=[x, w]):
        conv_kargs = dict(data_format=data_format, strides=strides, padding=padding, output_shape=out_shape)
        conv_func = getattr(tf.nn, f'conv{x.shape.ndims - 2}d_transpose')
        group = out_shape[-1] // w.shape[-2].value
        if group > 1 and disable_3d and x.shape.ndims == 5:
            logging.info('Unsupported official group transpose convolution, using native implementation')
            conv_kargs['output_shape'][-1] = conv_kargs['output_shape'][-1] // group
            x = group_kernel(x, w, group, conv_func, conv_kargs)
        else:
            x = conv_func(x, w, **conv_kargs)
        return x
