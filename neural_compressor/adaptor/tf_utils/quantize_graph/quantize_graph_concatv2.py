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
"""Quantize ConcatV2 to int8 op."""

import re

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.quantize_graph_common import QuantizeGraphHelper as helper

from .quantize_graph_base import QuantizeNodeBase


class FuseNodeStartWithConcatV2(QuantizeNodeBase):
    """Quantize ConcatV2 to int8 op QuantizedConcatV2."""

    def _apply_concatv2_transform(self, original_node):
        """Quantize ConcatV2."""
        namespace_prefix = original_node.name + "_eightbit"
        quantized_concat_name = namespace_prefix + "_quantized_concatv2"
        reshape_dims_name, reduction_dims_name = self._add_common_quantization_nodes(
            namespace_prefix, helper.node_name_from_input(original_node.input[-1])
        )
        num_input = len(original_node.input)
        shape_input_name = original_node.input[num_input - 1]
        original_inputs = original_node.input[0 : num_input - 1]
        input_names = []
        min_names = []
        max_names = []
        for original_input_name in original_inputs:
            original_input_node = self.node_name_mapping[original_input_name].node
            input_data_type = dtypes.quint8 if self._find_relu_node(original_input_node) else dtypes.qint8
            quantize_input_name, min_input_name, max_input_name = self._eightbitize_input_to_node(
                namespace_prefix, original_input_name, reshape_dims_name, reduction_dims_name, dtype=input_data_type
            )
            input_names.append(quantize_input_name)
            min_names.append(min_input_name)
            max_names.append(max_input_name)
        all_input_names = input_names
        all_input_names.append(shape_input_name)
        all_input_names.extend(min_names)
        all_input_names.extend(max_names)
        quantized_concat_node = helper.create_node("QuantizedConcatV2", quantized_concat_name, all_input_names)
        helper.set_attr_int(quantized_concat_node, "N", len(original_inputs))
        helper.set_attr_dtype(quantized_concat_node, "T", input_data_type)
        self.add_output_graph_node(quantized_concat_node)
        self._intel_cpu_add_dequantize_result_node(quantized_concat_name, original_node.name, input_data_type)

    def _quantizable_concat(self, node):
        """Check if the ConcatV2 is quantizable."""
        deq_type = []
        for input_node_name in node.input[: node.attr["N"].i]:
            node_name = helper.node_name_from_input(input_node_name)
            if self.node_name_mapping[node_name].node.op != "Dequantize":
                return False

            deq_type.append(self.node_name_mapping[node_name].node.attr["T"].type)

        if len(set(deq_type)) != 1:
            return False

        return True

    def _apply_concatv2_quantization(self):
        """Quantize ConcatV2 if it's quantizable."""
        for _, v in self.node_name_mapping.items():
            if (
                v.node.op in ("ConcatV2",)
                and self._quantizable_concat(v.node)
                and dtypes.as_dtype(v.node.attr["T"].type) == dtypes.float32
                and not re.search(r"map(_\d+)?/while", v.node.name)
            ):
                self._apply_concatv2_transform(v.node)
                self.quantizable_node_names.append(v.node.name)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(v.node)
                self.add_output_graph_node(new_node)

    def get_longest_fuse(self):
        """Only single ConcatV2, no fusion pattern."""
        return 1

    def apply_the_transform(self):
        """Apply the quantization of ConcatV2."""
        self.quantizable_node_names = []
        self._apply_concatv2_quantization()
        self._reset_output_node_maps()
        if self.remove_redundant_quant_flag:
            self.output_graph = self.remove_redundant_quantization(self.output_graph)
        return self.output_graph, self.quantizable_node_names, []
