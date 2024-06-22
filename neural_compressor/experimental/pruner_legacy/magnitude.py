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
"""Magnitude pruner."""

import numpy as np
from deprecated import deprecated

from neural_compressor.utils import logger

from .pruner import Pruner, pruner_registry


@deprecated(version="2.0")
@pruner_registry
class BasicMagnitudePruner(Pruner):
    """Magnitude pruner class.

    Args:
        model (object): The original model (currently PyTorchModel instance).
        local_config (Conf): configs specific for this pruning instance.
        global_config (Conf): global configs which may be overwritten by local_config.
    """

    def __init__(self, model, local_config, global_config):
        """Initialize the attributes."""
        super(BasicMagnitudePruner, self).__init__(model, local_config, global_config)

    def on_epoch_begin(self, epoch):
        """Update target sparsity according to the schedule and compute mask accordingly."""
        self.sparsity = self.update_sparsity(epoch)
        logger.debug("Start pruning in epoch {} with sparsity {}.".format(str(epoch), str(self.sparsity)))
        self.is_last_epoch = epoch == self.end_epoch
        if epoch >= self.start_epoch and epoch <= self.end_epoch:
            self.compute_mask()

    def on_step_begin(self, batch_id):
        """Apply mask to the weight."""
        res = dict()

        for weight in self.weights:
            if weight in self.masks:
                new_weight = self.masks[weight] * np.array(self.model.get_weight(weight))
                self.model.update_weights(weight, new_weight)
                res[weight] = new_weight
        return res

    def compute_mask(self):
        """Compute masks according to absolute values."""
        for weight in self.weights:
            tensor = np.array(self.model.get_weight(weight))
            if len(tensor.shape) in self.tensor_dims:
                reduced_tensor = self.pattern.reduce(tensor)
                if self.method == "per_channel":
                    tensor_flat = reduced_tensor.reshape(list(tensor.shape)[:-2], -1)
                    tensor_flat.sort(axis=-1)
                    threshold = tensor_flat[..., int(self.sparsity * tensor_flat.shape[-1])]
                    threshold = np.expand_dims(np.expand_dims(threshold, -1), -1)
                    threshold = np.repeat(threshold, reduced_tensor.shape[-1], axis=-1)
                    threshold = np.repeat(threshold, reduced_tensor.shape[-2], axis=-2)
                else:
                    tensor_flat = sorted(np.abs(reduced_tensor.flatten()))
                    threshold = float(tensor_flat[int(len(tensor_flat) * self.sparsity)])
                reduced_mask = threshold < np.abs(reduced_tensor)
                self.masks[weight] = self.pattern.repeat_mask(reduced_mask, tensor.shape)

    def on_epoch_end(self):
        """Sparsity ratio summary and apply mask to the weight."""
        res = dict()
        if self.is_last_epoch:
            for weight in self.weights:
                if weight in self.masks:
                    logger.info(
                        "Set {} sparsity with mask {} {} {}.".format(
                            weight,
                            str(self.masks[weight].size),
                            str(self.masks[weight].sum()),
                            str(1 - self.masks[weight].sum() / self.masks[weight].size),
                        )
                    )
                    new_weight = self.masks[weight] * np.array(self.model.get_weight(weight))
                    self.model.update_weights(weight, new_weight)
                    res[weight] = new_weight
        return res

    def on_step_end(self):
        """Apply mask to the weight."""
        res = dict()
        for weight in self.weights:
            if weight in self.masks:
                new_weight = self.masks[weight] * np.array(self.model.get_weight(weight))
                self.model.update_weights(weight, new_weight)
                res[weight] = new_weight
        return res
