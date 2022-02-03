# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from . import nn as gtf_nn
from . import layers as gtf_layers

import os
import logging
if 'ENABLE_TFG' in os.environ.keys():
    from .layers import rendering as gtf_render
else:
    logging.info('TensorFlow Graphics package is disable; Manually enable \
        it by adding `ENABLE_TFG` into the os environment variable')
