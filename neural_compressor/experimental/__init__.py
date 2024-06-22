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
"""Intel® Neural Compressor: An open-source Python library supporting popular model compression techniques."""

from .component import Component
from .quantization import Quantization
from .pruning import Pruning
from .benchmark import Benchmark
from .graph_optimization import Graph_Optimization, GraphOptimization
from .mixed_precision import MixedPrecision
from .model_conversion import ModelConversion
from .distillation import Distillation
from .nas import NAS
from . import export
from .contrib import *

__all__ = [
    "Component",
    "Quantization",
    "Pruning",
    "Benchmark",
    "Graph_Optimization",
    "GraphOptimization",
    "ModelConversion",
    "Distillation",
    "NAS",
    "MixedPrecision",
    "export",
]
