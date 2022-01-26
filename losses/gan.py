# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Tuple, Optional
from tensorflow.compat import v1 as tf_v1
from ..summary import *

import tensorflow as tf
import numpy as np


class GLoss(object):
    @staticmethod
    def logistic_non_saturating(real_score: tf.Tensor,
                                fake_score: tf.Tensor,
                                real_in: tf.Tensor,
                                fake_in: tf.Tensor) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Standard non-saturating loss for generator training

        Args:
            real_score:
            fake_score:
            real_in:
            fake_in:

        Returns:

        """
        del real_score, real_in, fake_in
        with tf.name_scope('GLogisticNS'):
            loss = tf.squeeze(tf.nn.softplus(-fake_score), axis=-1, name='logistic_ns_g')
            return loss, None

    @staticmethod
    def logistic_non_saturating_path_reg(real_score: tf.Tensor,
                                         fake_score: tf.Tensor,
                                         real_in: tf.Tensor,
                                         fake_in: tf.Tensor,
                                         latent_in: tf.Tensor,
                                         pl_decay: float = 0.01,
                                         pl_weight: float = 2.):
        """
        Standard non-saturating loss with perceptual path regularization for generator training

        Args:
            real_score:
            fake_score:
            real_in:
            fake_in:
            latent_in:
            pl_decay:
            pl_weight:

        Returns:

        """
        loss, _ = GLoss.logistic_non_saturating(real_score, fake_score, real_in, fake_in)
        with tf.name_scope('PerceptualPathRegularization'):
            pl_decay = tf.convert_to_tensor(pl_decay)
            pl_noise = tf.random.normal(tf.shape(fake_in)) / np.sqrt(np.prod(fake_in.shape[1:-1].as_list()))
            pl_grads, = tf.gradients(tf.reduce_sum(fake_in * pl_noise), [latent_in])
            pl_length = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
            add_summary_item('Loss/D/perceptual_length', pl_length, 'scalar')

            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0., dtype=tf.float32)
            pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_length) - pl_mean_var)
            pl_update = tf_v1.assign(pl_mean_var, pl_mean)
            with tf.control_dependencies([pl_update]):
                pl_penalty = tf.square(pl_length - pl_mean)
            reg = tf.identity(pl_penalty * pl_weight, name='perceptual_path')
        return loss, reg


class DLoss(object):
    @staticmethod
    def logistic_non_saturating(real_score: tf.Tensor,
                                fake_score: tf.Tensor,
                                real_in: tf.Tensor,
                                fake_in: tf.Tensor) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Standard non-saturating loss for discriminator training

        Args:
            real_score:
            fake_score:
            real_in:
            fake_in:

        Returns:

        """
        del real_in, fake_in
        with tf.name_scope('DLogisticNS'):
            add_summary_item('Loss/D/real_score', real_score, 'scalar')
            add_summary_item('Loss/D/fake_score', fake_score, 'scalar')
            loss = tf.nn.softplus(fake_score)
            loss += tf.nn.softplus(-real_score)
            loss = tf.squeeze(loss, axis=-1, name='logistic_ns_d')
            return loss, None

    @staticmethod
    def logistic_non_saturating_r1(real_score: tf.Tensor,
                                   fake_score: tf.Tensor,
                                   real_in: tf.Tensor,
                                   fake_in: tf.Tensor,
                                   gamma: float = 10.) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Standard non-saturating loss with simple gradient penalty for discriminator training

        Args:
            real_score:
            fake_score:
            real_in:
            fake_in:
            gamma:

        Returns:

        """
        loss, _ = DLoss.logistic_non_saturating(real_score, fake_score, real_in, fake_in)
        with tf.name_scope('SimpleGradientPenaltyRegularization'):
            real_grads, = tf.gradients(tf.reduce_sum(real_score), [real_in])
            gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=np.arange(real_in.shape.ndims - 1) + 1)
            add_summary_item('Loss/D/simple_gp', gradient_penalty, 'scalar')
            reg = tf.identity(gradient_penalty * (gamma * 0.5), name='simple_gp')
        return loss, reg
