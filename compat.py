# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf

from tensorflow.python.util import deprecation


@deprecation.deprecated(None, instructions='Batch gather is disable since TF2.0')
def compat_batch_gather(params, indices, name=None):
    try:
        return tf.gather(params, indices, batch_dims=-1, name=name)
    except TypeError:
        return tf.batch_gather(params, indices, name)
