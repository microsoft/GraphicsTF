# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf
import numpy as np

from typing import Optional, Tuple

from tensorflow.keras import layers
from tensorflow.compat import v1 as tf_v1
from tensorflow import nn
from tensorflow.python.ops import array_ops, standard_ops, sparse_ops, math_ops, gen_math_ops
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers


class InstanceNorm(layers.Layer):
    def __init__(self,
                 scale: bool = True,
                 bias: bool = True,
                 gamma_initializer: str = 'ones',
                 beta_initializer: str = 'zeros',
                 epsilon: float = 1e-12,
                 data_format: str = None,
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        super().__init__(trainable, name, dtype, **kwargs)
        self.data_format = K.image_data_format() if data_format is None else data_format
        assert self.data_format == 'channels_last'
        self.scale = scale
        self.bias = bias
        self.gamma_initializer = gamma_initializer
        self.beta_initializer = beta_initializer
        self.epsilon = epsilon

        self.gamma = None
        self.beta = None

    def build(self, input_shape: tf.TensorShape):
        num_channels = input_shape.dims[-1].value
        if self.scale:
            self.gamma = self.add_weight(name='gamma',
                                         shape=(num_channels,),
                                         initializer=initializers.get(self.gamma_initializer),
                                         trainable=self.trainable,
                                         dtype=self.dtype)
        if self.bias:
            self.beta = self.add_weight(name='beta',
                                        shape=(num_channels,),
                                        initializer=initializers.get(self.beta_initializer),
                                        trainable=self.trainable,
                                        dtype=self.dtype)
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):
        m, v = tf.nn.moments(inputs, axes=tf.range(1, inputs.shape.ndims - 1), keepdims=True)
        v_t = tf.math.rsqrt(v + self.epsilon)
        n: tf.Tensor = (inputs - m) * v_t
        n_shape = [1] * n.shape.ndims
        n_shape[-1] = inputs.shape.dims[-1].value
        if self.gamma is not None:
            n = n * tf.reshape(self.gamma, n_shape)
        if self.beta is not None:
            n = n + tf.reshape(self.beta, n_shape)
        return n


class AdaptiveInstanceNorm(layers.Layer):
    def __init__(self,
                 epsilon: float = 1e-12,
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        super().__init__(trainable, name, dtype, **kwargs)
        self.epsilon = epsilon
        self.instance_norm = InstanceNorm(False, False, epsilon=epsilon, trainable=trainable, dtype=dtype)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], **kwargs):
        assert len(inputs) == 3
        feat, scale, bias = inputs
        norm_feat = self.instance_norm(feat)
        f_dims = feat.shape.ndims - 2
        scale_b = tf.reshape(scale, [scale.shape.dims[0].value, *([1] * f_dims), scale.shape.dims[-1].value])
        bias_b = tf.reshape(bias, [bias.shape.dims[0].value, *([1] * f_dims), bias.shape.dims[-1].value])
        return scale_b * norm_feat + bias_b


class SpectrumNorm(layers.Layer):
    def __init__(self,
                 p_iter: int = 1,
                 epsilon: float = 1e-12,
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        super().__init__(trainable, name, dtype, **kwargs)
        self.p_iter = p_iter
        self.epsilon = epsilon
        self.move_u: Optional[tf.Tensor] = None

    def build(self, input_shape: tf.TensorShape):
        move_u_shape = [np.prod(input_shape.as_list()[:-1]), 1]
        self.move_u = self.add_weight('move_u', move_u_shape, self.dtype, tf.random_normal_initializer, trainable=False)
        self.built = True

    def call(self, inputs: tf.Tensor, as_master=True, **kwargs):
        w = tf.reshape(inputs, [-1, inputs.shape.as_list()[-1]])
        u = tf.identity(self.move_u)
        v = None
        for _ in range(self.p_iter):
            v = tf.nn.l2_normalize(tf.matmul(tf.transpose(w), u), axis=None, epsilon=self.epsilon)
            u = tf.nn.l2_normalize(tf.matmul(w, v), axis=None, epsilon=self.epsilon)

        if as_master and K.learning_phase():
            with tf.control_dependencies([tf_v1.assign(self.move_u, u, name="update_u")]):
                u = tf.identity(u)

        u = tf.stop_gradient(u)
        v = tf.stop_gradient(v)

        norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
        norm_value.shape.assert_is_fully_defined()
        norm_value.shape.assert_is_compatible_with([1, 1])

        w_normalized = w / norm_value

        w_tensor_normalized = tf.reshape(w_normalized, inputs.shape)
        return w_tensor_normalized


class SpectrumConvolution(Conv):
    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        super().__init__(rank, filters, kernel_size, trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.spectrum_norm = SpectrumNorm(trainable=trainable, dtype=dtype)

    def call(self, inputs: tf.Tensor, as_master=True):
        k = self.spectrum_norm(self.kernel, as_master)
        outputs = self._convolution_op(inputs, k)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class SpectrumConvolution2D(SpectrumConvolution):
    def __init__(self,
                 filters,
                 kernel_size,
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        super().__init__(2, filters, kernel_size, trainable=trainable, name=name, dtype=dtype, **kwargs)


class SpectrumConvolution3D(SpectrumConvolution):
    def __init__(self,
                 filters,
                 kernel_size,
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.dtypes = tf.float32,
                 **kwargs):
        super().__init__(3, filters, kernel_size, trainable=trainable, name=name, dtype=dtype, **kwargs)


class SpectrumLinear(layers.Dense):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self.spectrum_norm = SpectrumNorm()

    def call(self, inputs, as_master=True):
        k = self.spectrum_norm(self.kernel, as_master)
        rank = len(inputs.shape)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, k, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            if K.is_sparse(inputs):
                outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, k)
            else:
                outputs = gen_math_ops.mat_mul(inputs, k)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


class OrthogonalProjection(layers.Layer):
    def __init__(self,
                 axis: int,
                 name: str = None,
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor, **kwargs):
        assert inputs.dtype in [tf.int32, tf.int64, tf.uint8]
        vol_mesh_grid = tf.meshgrid(*[tf.range(a, dtype=inputs.dtype) for a in inputs.shape.as_list()],
                                    indexing='ij')[self.axis]
        vol_v_index = tf.argmax(tf.where(tf.equal(inputs, 0), tf.zeros_like(inputs), vol_mesh_grid), axis=self.axis)
        plane_mesh_grid = tf.meshgrid(*[tf.range(a, dtype=vol_v_index.dtype) for a in vol_v_index.shape.as_list()],
                                      indexing='ij')
        plane_a_axis = tf.stack([*plane_mesh_grid[:self.axis], vol_v_index, *plane_mesh_grid[self.axis:]], axis=-1)
        plane_label = tf.stop_gradient(tf.gather_nd(inputs, plane_a_axis))
        return plane_label
