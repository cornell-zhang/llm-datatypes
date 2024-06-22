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
"""LayerHistogramCollector: save the histogram by layer."""

import numpy as np

from neural_compressor.utils.utility import combine_histogram


class LayerHistogramCollector(object):
    """The collector of the histogram by layer.

    Saves layer histogram in a dict with layer names as keys and lists of NDArrays as
    values. The collected histogram will be used for calculating the optimal thresholds for
    quantization using KL divergence.
    """

    def __init__(self, num_bins=8001, layer_tensor=None, include_layer=None, logger=None):
        """Init a LayerHistogramCollector object.

        Args:
            num_bins: Number of bins for the histogram
            layer_tensor: A dict with layer names as keys and lists of NDArrays as values
            include_layer: Layers that need to collect histograms
            logger: Output logger information if use a logger
        """
        self.hist_dict = {}
        self.num_bins = num_bins
        self.layer_tensor = layer_tensor
        self.include_layer = include_layer
        self.logger = logger

    def collect(self):
        """Collect layer output NDArrays as a callback function."""
        # name = py_str(name)
        # if name not in self.include_layer:
        #     return
        # handle = ctypes.cast(arr, NDArrayHandle)
        # arr = NDArray(handle, writable=False).copyto(cpu()).asnumpy()
        for name in self.layer_tensor:
            if name not in self.include_layer:
                continue
            for arr in self.layer_tensor[name]:
                if self.logger:
                    self.logger.debug("Collect layer {} histogram of shape {}.".format(name, arr.shape))
                min_range = np.min(arr)
                max_range = np.max(arr)
                th = max(abs(min_range), abs(max_range))
                if name in self.hist_dict:
                    self.hist_dict[name] = combine_histogram(self.hist_dict[name], arr)
                else:
                    hist, hist_edges = np.histogram(arr, bins=self.num_bins, range=(-th, th))
                    self.hist_dict[name] = (hist, hist_edges, min_range, max_range, th)
