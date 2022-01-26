# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Dict
from tensorflow.compat import v1 as tf_v1

import tensorflow as tf


TRAIN_SCALAR_COLLECT: Dict = dict()


def add_summary_item(name, value, summary_type, phase='train'):
    assert phase == 'train'
    global TRAIN_SCALAR_COLLECT
    if summary_type == 'scalar':
        if name in TRAIN_SCALAR_COLLECT.keys():
            TRAIN_SCALAR_COLLECT[name].append(value)
        else:
            TRAIN_SCALAR_COLLECT[name] = [value]
    else:
        raise NotImplementedError


def make_all_summary():
    pass


def make_all_scalar_summary():
    global TRAIN_SCALAR_COLLECT
    for k, v in TRAIN_SCALAR_COLLECT.items():
        with tf.name_scope('AVGScalar', values=v):
            v_m = tf.reduce_mean(v, keepdims=False)
            tf_v1.summary.scalar(k, v_m, family='train')
