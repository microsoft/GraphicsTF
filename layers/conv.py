# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf

from typing import List, Tuple
from tensorflow.keras import layers, backend
from tensorflow.python.keras.layers import convolutional as conv_base
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops, nn
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from ..parser import depthwise_2d_fast, kernel_predict_conv2d


class FastDepthwiseConv2D(conv_base.DepthwiseConv2D):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 use_bias=True,
                 **kwargs):
        super(FastDepthwiseConv2D,
              self).__init__(kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             depth_multiplier=depth_multiplier,
                             data_format=data_format,
                             use_bias=use_bias,
                             **kwargs)
        self.use_native = None

    def call(self, inputs):
        if self.use_native is None:
            try:
                outputs = depthwise_2d_fast(inputs, self.depthwise_kernel)
                self.use_native = True
            except:
                outputs = backend.depthwise_conv2d(
                    inputs,
                    self.depthwise_kernel,
                    strides=self.strides,
                    padding=self.padding,
                    dilation_rate=self.dilation_rate,
                    data_format=self.data_format)
        elif self.use_native:
            outputs = depthwise_2d_fast(inputs, self.depthwise_kernel)
        else:
            outputs = backend.depthwise_conv2d(
                inputs,
                self.depthwise_kernel,
                strides=self.strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate,
                data_format=self.data_format)

        if self.use_bias:
            outputs = backend.bias_add(outputs,
                                       self.bias,
                                       data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs


class KernelPredictConv2D(layers.Layer):
    def __init__(self,
                 data_format=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 norm=None,
                 dtype=tf.float32,
                 name=None,
                 **kwargs):
        super(KernelPredictConv2D, self).__init__(trainable=True,
                                                  name=name,
                                                  dtype=dtype)
        self.data_format = data_format
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self._ndim = 4
        self._data_format = conv_utils.convert_data_format(
            data_format, self._ndim)

        self.bias = None
        self.norm = norm

    def build(self, input_shape: List[tf.TensorShape]):
        assert len(input_shape) == 2
        input_shape: Tuple[tf.TensorShape, tf.TensorShape] = tuple(input_shape)
        feat_shape, _ = input_shape

        input_channel = feat_shape[
            1] if self.data_format == 'channels_first' else feat_shape[-1]
        self.bias = self.add_weight(name='bias',
                                    shape=(input_channel, ),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    trainable=True,
                                    dtype=self.dtype)
        self.built = True

    def call(self, inputs: List[tf.Tensor]):
        assert len(inputs) == 2
        inputs: Tuple[tf.Tensor, tf.Tensor] = tuple(inputs)
        feat, kernel = inputs

        output = kernel_predict_conv2d(feat, kernel, self._data_format)

        if self.norm is not None:
            output = self.norm(output)

        if self.bias is not None:
            output = tf.nn.bias_add(output, self.bias, self._data_format)

        return output
