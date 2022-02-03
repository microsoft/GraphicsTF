# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
from enum import Enum
from typing import List
from abc import abstractmethod

from tensorflow.keras import layers



class InputKeyMode(Enum):
    """ The input organized structure for the loss call
    """
    DICT = 0
    LIST = 1


class BasicLossProxy(layers.Layer):
    """ The abstract class for all implemented loss function
    """
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self._losses_name: List[str] = list()

    @property
    def num_losses(self):
        """ Get the total number of losses
        """
        if not self._losses_name:
            self._assign_loss_names()
        return len(self._losses_name)

    @property
    def losses_name(self):
        """ Get the losses name
        """
        if not self._losses_name:
            self._assign_loss_names()
        return self._losses_name

    @property
    def input_key_mode(self):
        """ The key structure of inputs for call function.
        """
        return InputKeyMode.LIST

    @abstractmethod
    def _assign_loss_names(self):
        """
        Assign the names of contained losses
        """
        return list()

    @abstractmethod
    def call_internal(self, inputs, option=None, **kwargs):
        """ The abstract method to implement the loss definition.
        """

    def call(self, inputs, option=None, **kwargs):  # pylint: disable=arguments-differ
        out = self.call_internal(inputs, option, **kwargs)
        if isinstance(out, dict):
            res_out = out['out']
            del out['out']
            return res_out, out
        return out, dict()
