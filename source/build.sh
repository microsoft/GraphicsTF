# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#!/bin/sh
export TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

INCR_BUILD=${1-0}

CUR_DIR="$(pwd)"
# shellcheck disable=SC2039
SRC_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ -f libgraphicstf.so ] && [ $((INCR_BUILD == 1)) ]
then
echo "Ignore library building for existence"
else
cd "${SRC_DIR}" && mkdir build && cd build && cmake .. && make && make install && cd .. && rm -rf build && cd "${CUR_DIR}"
fi
