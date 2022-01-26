# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf

from tensorflow.python.util import deprecation
from tensorflow.keras import layers as keras_layers
from typing import List, Optional

from ..nn import sampling
from .. import compat


class SampleAndGroup(keras_layers.Layer):
    def __init__(self,
                 m: int,
                 radius: float,
                 n_sample: int,
                 use_xyz: bool = False,
                 trainable: bool = True,
                 name: str = None,
                 **kwargs):
        super().__init__(name=name, trainable=trainable, **kwargs)
        self.m = m
        self.radius = radius
        self.n_sample = n_sample
        self.use_xyz = use_xyz
        self.data_format = tf.keras.backend.image_data_format()
        assert self.data_format == 'channels_last'

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        input_xyz, input_feat = inputs
        if self.use_xyz:
            input_feat = tf.concat((input_xyz, input_feat), axis=-1)
        cnt_indices = sampling.iterative_farthest_point_sampling(input_xyz, self.m)
        new_xyz = compat.compat_batch_gather(input_xyz, cnt_indices)
        grouped_idx, pts_cnt = sampling.query_ball_point(input_xyz, new_xyz, self.radius, self.n_sample)
        grouped_flat = tf.reshape(grouped_idx, [new_xyz.shape.dims[0].value, -1])
        grouped_feat = compat.compat_batch_gather(input_feat, grouped_flat)
        grouped_raw_shape = [new_xyz.shape.dims[0].value, self.m, self.n_sample, -1]
        grouped_feat = tf.reshape(grouped_feat, grouped_raw_shape)
        grouped_xyz = compat.compat_batch_gather(input_xyz, grouped_flat)
        grouped_xyz = tf.reshape(grouped_xyz, grouped_raw_shape)
        return new_xyz, grouped_feat, grouped_idx, grouped_xyz

    def get_config(self):
        config = {
            'm': self.m,
            'radius': self.radius,
            'n_sample': self.n_sample,
            'use_xyz': self.use_xyz,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class SampleAndGroupAll(keras_layers.Layer):
    def __init__(self,
                 use_xyz: bool = False,
                 trainable: bool = True,
                 name: str = None,
                 **kwargs):
        super().__init__(name=name, trainable=trainable, **kwargs)
        self.use_xyz = use_xyz
        self.data_format = tf.keras.backend.image_data_format()
        assert self.data_format == 'channels_last'

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        input_xyz, input_feat = inputs
        if self.use_xyz:
            input_feat = tf.concat((input_xyz, input_feat), axis=-1)
        new_xyz = tf.zeros([input_feat.shape.dims[0].value, 1, 3])
        grouped_idx = tf.tile(tf.range(0, input_feat.shape.dims[1].value)[tf.newaxis, tf.newaxis, ...],
                              [input_feat.shape.dims[0].value, 1, 1])
        new_feat = tf.expand_dims(input_feat, axis=1)
        return new_xyz, new_feat, grouped_idx, input_xyz[:, tf.newaxis, ...]

    def get_config(self):
        config = {
            'use_xyz': self.use_xyz,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class SetAbstraction(keras_layers.Layer):
    def __init__(self,
                 m: int,
                 radius: float,
                 n_sample: int,
                 num_mlp: List[int],
                 num_mlp2: Optional[List[int]],
                 pooling: str = 'max',
                 act: STR_TF_FUNC_VARS = tf.nn.relu,
                 norm: STR_TF_FUNC_VARS = tf.identity,
                 use_xyz: bool = False,
                 trainable: bool = True,
                 group_all: bool = True,
                 name: str = None,
                 **kwargs):
        super().__init__(name=name, trainable=trainable, **kwargs)
        self.m = m
        self.radius = radius
        self.n_sample = n_sample
        self.num_mlp = num_mlp
        self.num_mlp2 = num_mlp2
        self.act = act
        self.norm = norm
        self.use_xyz = use_xyz
        self.pooling = pooling
        self.group_all = group_all
        self.data_format = tf.keras.backend.image_data_format()

        if not self.group_all:
            self.sample_and_group = SampleAndGroup(self.m, self.radius, self.n_sample, self.use_xyz, trainable, 'SG')
        else:
            self.sample_and_group = SampleAndGroupAll(self.use_xyz, trainable, 'SGA')
        self.mlp_ops = list()
        self.mlp2_ops = list()
        assert self.data_format == 'channels_last'

        self.input_spec = [keras_layers.InputSpec(ndim=3), keras_layers.InputSpec(ndim=3)]

    def build(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) == 2

        for idx, n_m in enumerate(self.num_mlp):
            self.mlp_ops.append(keras_layers.Conv2D(n_m, 1, 1, data_format=self.data_format, name=f'MLP_{idx}'))
        if self.num_mlp2 is not None:
            for idx, n2_m in enumerate(self.num_mlp2):
                self.mlp2_ops.append(keras_layers.Conv2D(n2_m, 1, 1, data_format=self.data_format, name=f'MLP2_{idx}'))
        self.built = True

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        input_xyz, input_feat = inputs
        new_xyz, new_feat, grouped_idx, grouped_xyz = self.sample_and_group([input_xyz, input_feat])
        for m_o in self.mlp_ops:
            new_feat = self.act(self.norm(m_o(new_feat)))
        if self.pooling == 'max':
            new_feat = tf.reduce_max(new_feat, axis=[2], keepdims=True, name='MaxPooling')
        elif self.pooling == 'avg':
            new_feat = tf.reduce_mean(new_feat, axis=[2], keepdims=True, name='AvgPooling')
        else:
            raise NotImplementedError
        new_feat = tf.squeeze(new_feat, axis=2)
        return new_xyz, new_feat, grouped_idx

    def get_config(self):
        config = {
            'm': self.m,
            'radius': self.radius,
            'n_sample': self.n_sample,
            'num_mlp': self.num_mlp,
            'num_mlp2': self.num_mlp2,
            'pooling': self.pooling,
            'use_xyz': self.use_xyz,
            "group_all": self.group_all
        }
        base_config = super().get_config()
        return {**base_config, **config}


class PointNet2ClsSSG(keras_layers.Layer):
    def __init__(self,
                 trainable: bool = True,
                 act: STR_TF_FUNC_VARS = tf.nn.relu,
                 norm: STR_TF_FUNC_VARS = tf.identity,
                 name: str = None,
                 **kwargs):
        super().__init__(name=name, trainable=trainable, **kwargs)
        self.act = act
        self.norm = norm
        self.sa_ops = list()
        self.fc_ops = list()
        self.dr_ops = list()
        self.reduction = None

    def build(self, input_shape):
        assert len(input_shape) == 2
        if input_shape[0][1].value == 8192:
            sa_args = {'m': [2048, 512, 128, 32, 0], 'radius': [0.05, 0.1, 0.2, 0.3, 0.],
                       'n_sample': [64, 128, 256, 128, 0], 'num_mlp2': [None, None, None, None, None],
                       'num_mlp': [[32, 32, 64], [64, 64, 128], [128, 128, 128], [128, 128, 256], [256, 512, 1024]],
                       'pooling': ['max', 'max', 'max', 'max', 'avg'], 'group_all': [False, False, False, False, True]}
        elif input_shape[0][1].value == 4096:
            sa_args = {'m': [1024, 256, 64, 16, 0], 'radius': [0.05, 0.1, 0.2, 0.3, 0.],
                       'n_sample': [32, 64, 128, 64, 0], 'num_mlp2': [None, None, None, None, None],
                       'num_mlp': [[32, 32, 64], [64, 64, 128], [128, 128, 128], [128, 128, 256], [256, 512, 1024]],
                       'pooling': ['max', 'max', 'max', 'max', 'avg'], 'group_all': [False, False, False, False, True]}
        elif input_shape[0][1].value == 2048:
            sa_args = {'m': [512, 128, 32, 0], 'radius': [0.1, 0.2, 0.3, 0.],
                       'n_sample': [32, 64, 128, 0], 'num_mlp2': [None, None, None, None],
                       'num_mlp': [[32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 512, 1024]],
                       'pooling': ['max', 'max', 'max', 'avg'], 'group_all': [False, False, False, True]}
        elif input_shape[0][1].value == 1024:
            sa_args = {'m': [256, 64, 16, 0], 'radius': [0.1, 0.2, 0.3, 0.],
                       'n_sample': [64, 64, 64, 0], 'num_mlp2': [None, None, None, None],
                       'num_mlp': [[32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 512, 1024]],
                       'pooling': ['max', 'max', 'max', 'avg'], 'group_all': [False, False, False, True]}
        else:
            raise NotImplementedError

        for arg_k, arg_v in sa_args.items():
            assert len(arg_v) == len(sa_args['m'])

        for as_i in range(len(sa_args['m'])):
            args = dict()
            for arg_k, arg_v in sa_args.items():
                args[arg_k] = arg_v[as_i]
            set_abs = SetAbstraction(act=self.act, norm=self.norm, trainable=self.trainable, name=f'SA_{as_i}', **args)
            self.sa_ops.append(set_abs)

        fc_1 = keras_layers.Dense(512, activation=self.act, trainable=self.trainable, name='FC_1')
        fc_2 = keras_layers.Dense(256, activation=self.act, trainable=self.trainable, name='FC_2')
        self.fc_ops.extend([fc_1, fc_2])
        # dr_1 = keras_layers.Dropout(0.5, trainable=self.trainable, name='DR_1')
        # dr_2 = keras_layers.Dropout(0.5, trainable=self.trainable, name='DR_2')
        # self.dr_ops.extend([dr_1, dr_2])
        self.dr_ops.extend([tf.identity, tf.identity])
        self.reduction = keras_layers.Dense(1, trainable=self.trainable, name='Reduction')
        self.built = True

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        pts_xyz, pts_feat = inputs
        for s_o in self.sa_ops:
            pts_xyz, pts_feat, _ = s_o([pts_xyz, pts_feat])
        pts_feat = tf.squeeze(pts_feat, axis=1)
        for f_o, d_o in zip(self.fc_ops, self.dr_ops):
            pts_feat = d_o(f_o(pts_feat))
        pts_feat = self.reduction(pts_feat)
        return pts_feat


@deprecation.deprecated(date=None, instructions='V1 interface will be disable in the future')
class PointNet2ClsSSGLegacyV1(BaseNet):
    KERAS_INSTANCE = {}

    def __init__(self, is_training: bool, act: STR_TF_FUNC_VARS, norm: STR_TF_FUNC_VARS, name: str = None):
        super().__init__(is_training, act, norm, name=name)
        if self.name not in PointNet2ClsSSGLegacyV1.KERAS_INSTANCE.keys():
            PointNet2ClsSSGLegacyV1.KERAS_INSTANCE[self.name] = PointNet2ClsSSG(self.is_training, self.act, self.norm,
                                                                                self.name)
        self.keras_backbone = PointNet2ClsSSGLegacyV1.KERAS_INSTANCE[self.name]

        self.input_feat = None
        self.input_xyz = None
        self.output_logit = None

    def process_inputs(self, raw_inputs: List[tf.Tensor]):
        assert len(raw_inputs) == 2
        self.input_xyz, self.input_feat = raw_inputs

    def build_network(self):
        input_vars = [self.input_xyz, self.input_feat]
        self.output_logit = self.keras_backbone(input_vars)

    def outputs(self) -> List[tf.Tensor]:
        return [self.output_logit]

    def inputs(self) -> List[tf.Tensor]:
        return [self.input_xyz, self.input_feat]


class TestPointNet(tf.test.TestCase):
    def __init__(self, method_name='TestPointNet'):
        super().__init__(methodName=method_name)

    def testPointNet2ClsSSG(self):
        sa_module = PointNet2ClsSSG()
        sa_module([tf.zeros([4, 1024, 3], dtype=tf.float32), tf.zeros([4, 1024, 20], dtype=tf.float32)])

    def testPointNet2ClsSSGLegacyV1(self):
        optimizer = tf.compat.v1.train.AdamOptimizer()
        tower_grads = list()
        for i in range(2):
            with tf.device(f'/gpu:{i}'), tf.name_scope(f'gpu_{i}'):
                in_op = [tf.zeros([4, 1024, 3], dtype=tf.float32), tf.zeros([4, 1024, 20], dtype=tf.float32)]
                label_op = tf.ones([4], dtype=tf.int32)
                point_net2 = PointNet2ClsSSGLegacyV1(False, '', '', 'PointNet2')
                point_net2.process_inputs(in_op)
                point_net2.build_network()
                out_ops, = point_net2.outputs()
                out_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(label_op, out_ops)
                tower_grads.append(optimizer.compute_gradients(out_loss))
