# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import gradient_checker_v2, gradient_checker

graphics_tf_module = None
try:
    import sys
    lib_name = 'libgraphicstf.so' if not sys.platform.startswith('win') else 'graphicstf.dll'
    _lib_path = os.path.dirname(__file__)
    graphics_tf_module = tf.load_op_library(os.path.join(_lib_path, 'source', lib_name))
except tf.errors.NotFoundError:
    logging.warning('Fail to load GraphicsTF runtime library support!')


def scatter_operator(indices, feat, scattered, scatter_type):
    """
    """
    if indices.dtype == tf.int64:
        indices = tf.cast(indices, tf.int32)
    if scattered.dtype == tf.int64:
        scattered = tf.cast(scattered, tf.int32)
    return graphics_tf_module.scatter_operator(indices, feat, scattered, scatter_type.value)


@ops.RegisterGradient('ScatterOperator')
def scatter_operator_grad(op, scattered_grad, statistics_grad):
    """
    """
    del statistics_grad
    indices, _, _ = op.inputs
    scattered_feat, _ = op.outputs
    # indices = tf.expand_dims(indices, axis=-1)
    grad = graphics_tf_module.scatter_operator_grad(indices, scattered_feat, scattered_grad, op.get_attr('type'))
    return None, grad, None


def depthwise_2d_fast(feat, weights):
    """
    """
    feat = tf.convert_to_tensor(feat)
    weights = tf.convert_to_tensor(weights)
    kernel_size = weights.shape.as_list()[0]
    return graphics_tf_module.depthwise_conv_fast(feat, weights, kernel_size)


@ops.RegisterGradient('DepthwiseConvFast')
def depthwise_conv_fast_grad(op: tf.Operation, out_feat_diff):
    """
    """
    in_feat, in_filter = op.inputs
    kernel_size = op.get_attr('kernel_size')
    grad_filter_diff = graphics_tf_module.depthwise_conv_fast_grad_filter(out_feat_diff, in_feat, kernel_size)
    grad_in_diff = graphics_tf_module.depthwise_conv_fast_grad_input(out_feat_diff, in_filter)
    return grad_in_diff, grad_filter_diff


@ops.RegisterStatistics("DepthwiseConvFast", "flops")
def _calc_depthwise_conv_fast_flops(graph, node):
    input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    input_shape.assert_is_fully_defined()
    filter_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[1])
    filter_shape.assert_is_fully_defined()
    output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    output_shape.assert_is_fully_defined()
    filter_height = int(filter_shape[2])
    filter_width = int(filter_shape[3])
    output_count = np.prod(output_shape.as_list(), dtype=np.int64)
    return ops.OpStats("flops", (output_count * filter_height * filter_width * 2))


def kernel_predict_conv2d(feat, kernel, data_format='NHWC'):
    feat = tf.convert_to_tensor(feat)
    kernel = tf.convert_to_tensor(kernel)
    return graphics_tf_module.kernel_predict_conv2d(feat, kernel, data_format)

@ops.RegisterGradient('KernelPredictConv2D')
def kernel_predict_conv2d_grad(op: tf.Operation, out_feat_diff: tf.Tensor, data_format='NHWC'):
    in_feat, in_filter = op.inputs
    grad_input_diff = graphics_tf_module.kernel_predict_conv2d_grad_input(
        in_feat.shape, in_filter, out_feat_diff, data_format)
    grad_filter_diff = graphics_tf_module.kernel_predict_conv2d_grad_filter(
        in_feat, tf.shape(in_filter), out_feat_diff, data_format)
    return grad_input_diff, grad_filter_diff


class CustomizeOpsTest(tf.test.TestCase):
    @staticmethod
    def _generate_single_kernel_predict_conv_case():
        feat_value = np.array([[[[1.], [2.]], [[3.], [4.]]]], dtype=np.float32)
        kernel_value = np.arange(9, dtype=np.float32)[np.newaxis, np.newaxis, np.newaxis, ...]
        kernel_value = np.repeat(np.repeat(kernel_value, 2, axis=1), 2, axis=2)
        kernel_value = kernel_value * np.array([[[[1.], [2.]], [[3.], [4.]]]], dtype=np.float32)
        predict_value = np.array([[[[67.], [114.]], [[111.], [108.]]]], dtype=np.float32)
        return feat_value, kernel_value, predict_value

    @staticmethod
    def _generate_batch_kernel_predict_conv_case():
        feat_value, kernel_value, predict_value = CustomizeOpsTest._generate_single_kernel_predict_conv_case()
        feat_value = np.repeat(feat_value, 3, axis=-1) * np.array([[[[1., 2., 3.]]]], dtype=np.float32)
        feat_value = np.repeat(feat_value, 2, axis=0) * np.array([[[[1.]]], [[[2.]]]], dtype=np.float32)
        kernel_value = np.repeat(kernel_value, 2, axis=0)  # * np.array([[[[1.]]], [[[2.]]]], dtype=np.float32)
        predict_value = np.repeat(predict_value, 3, axis=-1) * np.array([[[[1., 2., 3.]]]], dtype=np.float32)
        predict_value = np.repeat(predict_value, 2, axis=0) * np.array([[[[1.]]], [[[2.]]]], dtype=np.float32)
        return feat_value, kernel_value, predict_value

    def testKernelPredictConv2D(self):
        feat_value, kernel_value, predict_value = self._generate_single_kernel_predict_conv_case()
        output_value = kernel_predict_conv2d(feat_value, kernel_value).numpy()
        self.assertAllEqual(predict_value, output_value)

        feat_value, kernel_value, predict_value = self._generate_batch_kernel_predict_conv_case()
        output_value = kernel_predict_conv2d(feat_value, kernel_value).numpy()
        self.assertAllEqual(predict_value, output_value)

    def testKernelPredictConv2DBackprop(self):
        feat_value, kernel_value, _ = self._generate_batch_kernel_predict_conv_case()
        theorical, numerical = gradient_checker_v2.compute_gradient(kernel_predict_conv2d, 
                                                                    [feat_value, kernel_value], 
                                                                    delta=1e-1)
        self.assertLessEqual(gradient_checker_v2.max_error(theorical, numerical), 1e-3)
