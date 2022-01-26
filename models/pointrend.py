# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import numpy as np
from typing import List, Optional

import tensorflow as tf
from tensorflow.compat import v1 as tf_v1

from ..nn import sampling, interpolate, nn
from .. import layers
from ..layers import STR_TF_FUNC_VARS
from .stylegan import PGGANOps


class PointRend(layers.BaseNet):
    def __init__(self,
                 is_training: bool,
                 num_channels: int,
                 points_num: int,
                 data_source: str = 'fake',
                 act: STR_TF_FUNC_VARS = tf.nn.leaky_relu,
                 norm: STR_TF_FUNC_VARS = tf.identity,
                 dtype: str = 'float32',
                 name: str = None
                 ):
        super().__init__(is_training, act, norm, dtype)
        self.name = 'PointRend' if name is None else name

        self.data_source = data_source
        self.num_channels = num_channels
        self.points_num = points_num

        self.in_vol_hq: Optional[tf.Tensor] = None
        self.in_vol: Optional[tf.Tensor] = None
        self.in_pyramid_feat: List[tf.Tensor] = list()

        self.points_xyz: Optional[tf.Tensor] = None
        self.points_label: Optional[tf.Tensor] = None

    def process_inputs(self, raw_inputs: List[tf.Tensor]):
        if self.data_source == 'real':
            self.in_vol = tf.identity(raw_inputs[0], 'in_vol')
            self.in_vol_hq = tf.identity(raw_inputs[1], 'in_vol_HQ')
        elif self.data_source == 'fake':
            self.in_vol = tf.identity(raw_inputs[0], name='in_vol')
            self.in_pyramid_feat = raw_inputs[1]
        else:
            raise NotImplementedError

    def point_rend_importance_sample(self, input_, points_num):
        with tf.name_scope('PointRendSample', values=[input_]):
            vol_feat = tf.identity(input_)
            sample_shape = np.array(vol_feat.shape[1:-1].as_list()) - 1
            if self.data_source == 'real':
                vol_feat = tf.pad(vol_feat, tf.constant([[0, 0], *[[1, 1]]*len(sample_shape), [0, 0]]), 'REFLECT')
                vol_feat = PGGANOps.blur(vol_feat, padding='VALID')
            # TODO: k, beta
            sampled_pts = sampling.importance_coverage_sampling(vol_feat, vol_feat, points_num, sample_shape, k=20, beta=1.0)
            sampled_pts = sampled_pts / sample_shape[0]
            batch_index = np.arange(vol_feat.shape[0].value)[..., np.newaxis, np.newaxis]
            batch_index = tf.convert_to_tensor(np.tile(batch_index, [1, points_num, 1]), dtype=tf.float32)
            batch_sampled_pts = tf.concat((batch_index, sampled_pts), axis=-1)
            batch_sampled_pts = tf.reshape(batch_sampled_pts, (vol_feat.shape.as_list()[0] * points_num, -1))
            return sampled_pts, batch_sampled_pts

    def points_feat2label(self, input_, reuse=False, name=None):
        with tf.variable_scope(name, default_name='PointRendFeat2Label', reuse=reuse):
            pts_feat = tf.identity(input_)
            # TODO: channel
            pts_feat = self.act(layers.linear(pts_feat, 256, name='MPL0'))
            pts_feat = self.act(layers.linear(pts_feat, 128, name='MPL1'))
            pts_feat = layers.linear(pts_feat, self.num_channels, name='MPL2')
            pts_label = nn.softmax(pts_feat)
            return pts_label

    def build_network(self, reuse=tf_v1.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse):
            sampled_pts, batch_sampled_pts = self.point_rend_importance_sample(self.in_vol, self.points_num)
            self.points_xyz = tf.identity(sampled_pts, name='points_xyz')

            if self.data_source == 'real':
                pts_scale = self.in_vol_hq.shape.as_list()[1] - 1
                pts_label = interpolate.linear_nd_interpolate(self.in_vol_hq, batch_sampled_pts, pts_scale)
            elif self.data_source == 'fake':
                interp_feat_list = [tf.reshape(sampled_pts, (self.in_vol.shape.as_list()[0] * self.points_num, -1))]
                pyramid_feat = [feat for feat in self.in_pyramid_feat if feat.shape.as_list()[1] > 4]
                for p_i, feat in enumerate(pyramid_feat):
                    interp_feat = interpolate.linear_nd_interpolate(feat, batch_sampled_pts, feat.shape.as_list()[1]-1)
                    interp_feat_list.append(interp_feat)

                pts_label = self.points_feat2label(tf.concat(interp_feat_list, axis=-1))
            else:
                raise NotImplementedError

            pts_label = tf.reshape(pts_label, (self.in_vol.shape.as_list()[0], self.points_num, -1))
            self.points_label = tf.identity(pts_label, name='points_label')

    def outputs(self) -> List[tf.Tensor]:
        return [self.points_xyz, self.points_label]


class TestPointRend(tf.test.TestCase):
    def __init__(self, method_name='TestPointRend'):
        super().__init__(methodName=method_name)

    @staticmethod
    def make_one_hot(data, channel_num=21):
        return np.eye(channel_num, dtype=np.float32)[data]

    def testPointRend(self):
        beta = 1.0
        b, num_channels, points_num, points_resolution = 2, 21, 1024, 64
        resource_dir = f'{os.path.split(__file__)[0]}/../unittest_resource'
        vol_v32_data = np.load(os.path.join(resource_dir, 'unittest_input.npz'))['data']
        vol_hq_data = np.load(os.path.join(resource_dir, f'unittest_input_v{points_resolution}.npz'))['data']
        assert b == vol_v32_data.shape[0] == vol_hq_data.shape[0]

        real_point_rend = PointRend(False, num_channels, points_num, data_source='real')
        real_in = [tf.convert_to_tensor(self.make_one_hot(vol_v32_data)),
                   tf.convert_to_tensor(self.make_one_hot(vol_hq_data))]
        real_point_rend.process_inputs(real_in)
        real_point_rend.build_network()
        real_pts_xyz, real_pts_feat = real_point_rend.outputs()
        self.assertAllEqual(real_pts_xyz.shape.as_list(), [b, points_num, len(real_in[0].shape.as_list())-2])
        self.assertAllEqual(real_pts_feat.shape.as_list(), [b, points_num, num_channels])

        with tf.Session() as sess:
            real_pts_xyz_, real_pts_feat_ = sess.run([real_pts_xyz, real_pts_feat])
            real_logits_sum = np.sum(real_pts_feat_, axis=-1)
            self.assertAllClose(real_logits_sum, np.ones_like(real_logits_sum))
