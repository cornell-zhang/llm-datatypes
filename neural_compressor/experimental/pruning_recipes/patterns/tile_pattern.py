"""Tile pattern classes."""
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

import numpy as np
from deprecated import deprecated

from .pattern import PatternBase, pattern_registry


@deprecated(version="2.0")
class TilePatternBase(PatternBase):
    """Parent class for all NxM tile patterns."""

    def __init__(self, mask_shape):
        """Magnitude mask should be contiguous."""
        super(TilePatternBase, self).__init__(mask_shape, True)

    def compute_sparsity(self, tensor):
        """Calculate the sparsity of a tensor (weight matrix)."""
        reduced_tensor = self.reduce(tensor)
        return 1 - np.count_nonzero(reduced_tensor) / reduced_tensor.size

    def repeat_mask(self, mask, ori_shape=None):
        """Repeat mask in 2 dimensions."""
        flatten_mask = np.repeat(np.repeat(mask, self.mask_shape[0], axis=-2), self.mask_shape[1], axis=-1)
        if ori_shape:
            return flatten_mask.reshape(ori_shape)
        else:
            return flatten_mask


@deprecated(version="2.0")
@pattern_registry(pattern_type="tile_pattern_1x1")
class TilePattern_1x1(TilePatternBase):
    """1x1 tile pattern (unstructured)."""

    def __init__(self):
        """Element wise sparsity."""
        super(TilePattern_1x1, self).__init__([1, 1])


@deprecated(version="2.0")
@pattern_registry(pattern_type="tile_pattern_2x2")
class TilePattern_2x2(TilePatternBase):
    """2x2 tile pattern (unstructured)."""

    def __init__(self):
        """2x2 tile wise sparsity."""
        super(TilePattern_2x2, self).__init__([2, 2])


@deprecated(version="2.0")
@pattern_registry(pattern_type="tile_pattern_1x16")
class TilePattern_1x16(TilePatternBase):
    """1x16 tile pattern (unstructured)."""

    def __init__(self):
        """1x16 tile wise sparsity."""
        super(TilePattern_1x16, self).__init__([1, 16])


@deprecated(version="2.0")
@pattern_registry(pattern_type="tile_pattern_4x1")
class TilePattern_4x1(TilePatternBase):
    """4x1 tile pattern (unstructured)."""

    def __init__(self):
        """4x1 tile wise vnni-aware sparsity."""
        super(TilePattern_4x1, self).__init__([4, 1])


@deprecated(version="2.0")
@pattern_registry(pattern_type="tile_pattern_1x2")
class TilePattern_1x2(TilePatternBase):
    """1x2 tile pattern (unstructured)."""

    def __init__(self):
        """1x2 tile wise vnni-aware sparsity."""
        super(TilePattern_1x2, self).__init__([1, 2])
