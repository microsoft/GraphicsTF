# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import tensorflow as tf
import tensorflow.keras.layers as layers

from typing import List
from abc import abstractmethod


class BasicLossProxy(layers.Layer):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self._losses_name: List[str] = list()

    @property
    def num_losses(self):
        return len(self._losses_name)

    @property
    def losses_name(self):
        if not self._losses_name:
            logging.error('Please declare the name of all loss terms to enable automatic name assignment '
                          'in TensorBoard: self._losses_name')
            raise NotImplementedError
        return self._losses_name

    @abstractmethod
    def call(self, inputs, option=None, **kwargs):
        pass


class BasicModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def call(self, inputs, training=None, as_master=True, vis_port=False, mask=None): pass
