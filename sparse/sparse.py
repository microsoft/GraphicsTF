# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import numpy as np
import tensorflow as tf

from typing import List
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class SparseConv(layers.Layer):
    """
    * `__init__()`: Save configuration in member variables
    * `build()`: to initialize the weights
    * `call()`: require two inputs -- a feat tensor and a non-empty mask tensor
    """
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        assert K.image_data_format() == 'channels_last'
        super().__init__(trainable, name, dtype, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

        self.w = None
        self.b = None

    def build(self, input_shape: List[tf.TensorShape]):
        assert len(input_shape) == 2
        feat_shape, _ = tuple(input_shape)
        k_size = self.kernel_size ** (feat_shape.ndims - 2)
        c_in = feat_shape.dims[-1].value
        self.w = self.add_weight('w', shape=[c_in * k_size, self.filters], dtype=self.dtype, trainable=self.trainable,
                                 initializer=tf.initializers.glorot_normal())
        self.b = self.add_weight('b', shape=[self.filters], dtype=self.dtype, trainable=self.trainable,
                                 initializer=tf.initializers.zeros)
        self.built = True

    def gen_conv_index_map(self, k_dim):
        index_map = np.arange(self.kernel_size) - self.kernel_size // 2
        index_grid = np.stack(np.meshgrid(*([index_map] * k_dim), indexing='ij'), axis=0)
        index_grid = np.reshape(index_grid, [3, -1])
        return tf.convert_to_tensor(index_grid)

    def call(self, inputs: List[tf.Tensor], **kwargs):
        assert len(inputs) == 2
        feat, index = tuple(inputs)
        k_dim = feat.shape.ndims - 2
        k_size = self.kernel_size ** k_dim
        valid_index = tf.where(tf.not_equal(index, 0))
        batch_index, nd_index = tf.split(valid_index, [1, k_dim], axis=-1)
        batch_index_exp = tf.tile(batch_index[..., tf.newaxis], [*([1] * batch_index.shape.ndims), k_size])
        nd_index_exp = tf.tile(nd_index[..., tf.newaxis], [*([1] * nd_index.shape.ndims), k_size])
        nd_index_exp = nd_index_exp + tf.cast(self.gen_conv_index_map(k_dim), dtype=nd_index_exp.dtype)
        index_exp = tf.transpose(tf.concat([batch_index_exp, nd_index_exp], axis=-2), [0, 2, 1])
        feat_exp = tf.gather_nd(feat, index_exp)
        # feat_exp = tf.reshape(feat_exp, [feat_exp.shape.as_list()[0], -1])
        feat_exp = tf.reshape(feat_exp, [-1, feat_exp.shape.as_list()[1] * feat_exp.shape.as_list()[2]])
        feat_mul = tf.matmul(feat_exp, self.w) + self.b
        out_shape = [*feat.shape.as_list()[:-1], self.filters]
        out = tf.scatter_nd(valid_index, feat_mul, out_shape)
        return out


class TestSparse(tf.test.TestCase):
    def __init__(self, method_name='TestSparseConv'):
        super().__init__(methodName=method_name)
        self.update_gt = True
        self.resource_dir = f'{os.path.split(__file__)[0]}/../unittest_resource'

    def testSparseConv(self):
        sparse_conv = SparseConv(16, 3, trainable=True, name='test_sparse_conv', dtype=tf.float32)
        feat_in = np.random.rand(4, 8, 8, 8, 6)
        index_in = np.random.rand(4, 8, 8, 8)
        index_in = np.where(index_in > 0.5, np.ones_like(index_in), np.zeros_like(index_in)).astype(np.int32)
        feat_conv = sparse_conv([feat_in, index_in])
        pass
