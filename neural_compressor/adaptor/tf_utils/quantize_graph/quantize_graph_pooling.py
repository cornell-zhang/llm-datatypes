#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantize the AvgPool/MaxPool."""

import tensorflow as tf
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.quantize_graph_common import QuantizeGraphHelper as helper
from neural_compressor.adaptor.tf_utils.util import version1_eq_version2, version1_gt_version2, version1_lt_version2

from .quantize_graph_base import QuantizeNodeBase


class FuseNodeStartWithPooling(QuantizeNodeBase):
    """Quantize the AvgPool and MaxPool."""

    def _add_pool_function(self, original_node, quantized_op_node):
        """Set quantized pooling node attributes."""
        pooling_type = (
            dtypes.quint8
            if version1_lt_version2(tf.version.VERSION, "2.6.0") or self._find_relu_node(original_node)
            else dtypes.qint8
        )
        helper.set_attr_dtype(quantized_op_node, "T", pooling_type)
        helper.copy_attr(quantized_op_node, "ksize", original_node.attr["ksize"])
        helper.copy_attr(quantized_op_node, "strides", original_node.attr["strides"])
        helper.copy_attr(quantized_op_node, "padding", original_node.attr["padding"])

    def _apply_pool_quantization(self):
        """Quantize AvgPool/MaxPool."""
        for _, v in self.node_name_mapping.items():
            # Tensorflow 2.5.0 enabled the s8 input for pooling op.
            # If the tf version is lower than 2.5.0, we need to confirm the input
            # data type of pooling is unsigned.
            if v.node.name == self.start_node_name and (
                version1_gt_version2(tf.version.VERSION, "2.5.0")
                or version1_lt_version2(tf.version.VERSION, "2.6.0")
                and self._find_relu_node(v.node)
            ):
                self.eightbitize_single_input_tensor_node(v.node, self._add_pool_function)
                self.quantizable_node_names.append(v.node.name)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(v.node)
                self.add_output_graph_node(new_node)

    def get_longest_fuse(self):
        """Only pooling op itself, no fusion pattern."""
        return 1

    def apply_the_transform(self):
        """Quantize AvgPool/MaxPool."""
        self.quantizable_node_names = []
        self._apply_pool_quantization()
        self._reset_output_node_maps()
        if self.remove_redundant_quant_flag:
            self.output_graph = self.remove_redundant_quantization(self.output_graph)
        return self.output_graph, self.quantizable_node_names, []
