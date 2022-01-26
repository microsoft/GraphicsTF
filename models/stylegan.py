# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Optional

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.python.keras import activations


class ZMapping(layers.Layer):
    def __init__(self,
                 out_channel: int,
                 std_dev: float = 0.02,
                 activation: Optional[layers.Layer] = None,
                 trainable: bool = True,
                 dtype: tf.DType = tf.float32,
                 name: str = None,
                 **kwargs):
        super().__init__(trainable, name, dtype, **kwargs)
        self.out_channel = out_channel
        self.std_dev = std_dev
        self.activation = activations.get(activation)

        self.w = None
        self.b = None

    def build(self, input_shape: tf.TensorShape):
        assert K.image_data_format() == 'channels_last'
        input_channel = input_shape.dims[-1].value
        self.w = self.add_weight('w', [input_channel, self.out_channel * 2], dtype=self.dtype, trainable=self.trainable,
                                 initializer=tf.random_normal_initializer(stddev=self.std_dev))
        self.b = self.add_weight('b', [self.out_channel * 2], dtype=self.dtype, trainable=self.trainable,
                                 initializer=tf.constant_initializer(0.))
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):
        m_s, m_b = tf.split(self.activation(tf.matmul(inputs, self.w) + self.b), 2, axis=-1)
        return m_s, m_b

