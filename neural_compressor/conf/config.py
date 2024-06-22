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

import copy
import datetime
import itertools
import os
import re
from collections import OrderedDict

import yaml
from schema import And, Hook, Optional, Or, Schema, Use

from ..adaptor import FRAMEWORKS
from ..objective import OBJECTIVES
from ..utils import logger
from ..version import __version__
from .dotdict import DotDict, deep_set

# TODO WA for avoid circular import
# from ..experimental.strategy import EXP_STRATEGIES
EXP_STRATEGIES = ['basic', 'auto_mixed_precision', 'bayesian', 'conservative',\
    'exhaustive', 'hawq_v2', 'mse', 'mse_v2', 'random', 'sigopt', 'tpe', 'fake']

def constructor_register(cls):
    yaml_key = "!{}".format(cls.__name__)

    def constructor(loader, node):
        instance = cls.__new__(cls)
        yield instance

        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    yaml.add_constructor(
        yaml_key,
        constructor,
        yaml.SafeLoader,
    )
    return cls

@constructor_register
class Pruner():
    def __init__(self, start_epoch=None, end_epoch=None, initial_sparsity=None,
                 target_sparsity=None, update_frequency=1,
                 method='per_tensor',
                 prune_type='basic_magnitude',##for pytorch pruning, these values should be None
                 start_step=None, end_step=None, update_frequency_on_step=None, prune_domain=None,
                 sparsity_decay_type=None, pattern="tile_pattern_1x1", names=None,
                 extra_excluded_names=None, parameters=None):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.update_frequency = update_frequency
        self.target_sparsity = target_sparsity
        self.initial_sparsity = initial_sparsity
        self.update_frequency = update_frequency
        self.start_step = start_step
        self.end_step = end_step
        self.update_frequency_on_step = update_frequency_on_step
        self.prune_domain = prune_domain
        self.sparsity_decay_type = sparsity_decay_type
        self.extra_excluded_names = extra_excluded_names
        self.pattern = pattern
        ## move this to experimental/pruning to support dynamic pruning
        # assert prune_type.replace('_', '') in [i.lower() for i in PRUNERS], \
        #                                  'now only support {}'.format(PRUNERS.keys())
        self.prune_type = prune_type
        self.method = method
        self.names= names
        self.parameters = parameters


@constructor_register
class PrunerV2:
    """Similar to torch optimizer's interface."""

    def __init__(self,
                 target_sparsity=None, pruning_type=None, pattern=None, op_names=None,
                 excluded_op_names=None,
                 start_step=None, end_step=None, pruning_scope=None, pruning_frequency=None,
                 min_sparsity_ratio_per_op=None, max_sparsity_ratio_per_op=None,
                 sparsity_decay_type=None, pruning_op_types=None, reg_type=None,
                 criterion_reduce_type=None, parameters=None, resume_from_pruned_checkpoint=None):
        self.pruner_config = DotDict({
            'target_sparsity': target_sparsity,
            'pruning_type': pruning_type,
            'pattern': pattern,
            'op_names': op_names,
            'excluded_op_names': excluded_op_names,  ##global only
            'start_step': start_step,
            'end_step': end_step,
            'pruning_scope': pruning_scope,
            'pruning_frequency': pruning_frequency,
            'min_sparsity_ratio_per_op': min_sparsity_ratio_per_op,
            'max_sparsity_ratio_per_op': max_sparsity_ratio_per_op,
            'sparsity_decay_type': sparsity_decay_type,
            'pruning_op_types': pruning_op_types,
            'reg_type': reg_type,
            'criterion_reduce_type': criterion_reduce_type,
            'parameters': parameters,
            'resume_from_pruned_checkpoint': resume_from_pruned_checkpoint
        })


# Schema library has different loading sequence priorities for different
# value types.
# To make sure the fields under dataloader.transform field of yaml file
# get loaded with written sequence, this workaround is used to convert
# None to {} in yaml load().
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:null', lambda loader, node: {})
# Add python tuple support because best_configure.yaml may contain tuple
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
                                lambda loader, node: tuple(loader.construct_sequence(node)))

def _valid_accuracy_field(key, scope, error):
    assert bool(
        'relative' in scope['accuracy_criterion']) != bool(
        'absolute' in scope['accuracy_criterion'])

def _valid_prune_epoch(key, scope, error):
    if "start_epoch" in scope[key] and "end_epoch" in scope[key]:
        assert scope[key]["start_epoch"] <= scope[key]["end_epoch"]

def _valid_prune_sparsity(key, scope, error):
    if "initial_sparsity" in scope[key] and "target_sparsity" in scope[key]:
        assert scope[key]["initial_sparsity"] <= scope[key]["target_sparsity"]
    elif "initial_sparsity" in scope[key]:
        assert scope[key]["initial_sparsity"] >= 0
    elif "target_sparsity" in scope[key]:
        assert scope[key]["target_sparsity"] < 1

def _valid_multi_objectives(key, scope, error):
    if 'weight' in scope[key] and scope[key]['weight'] is not None:
        assert len(scope[key]['objective']) == len(scope[key]['weight'])

def _valid_multi_metrics(key, scope, error):
    if 'metric' in scope and 'multi_metrics' in scope:
        assert False

def _valid_metric_length(key, scope, error):
    metrics = [i for i in scope[key] if i != 'weight' and i != 'higher_is_better']
    if 'weight' in scope[key] and scope[key]['weight'] is not None:
        assert len(input_to_list_float(scope[key]['weight'])) == len(metrics)
    if 'higher_is_better' in scope[key] and scope[key]['higher_is_better'] is not None:
        assert len(input_to_list_bool(scope[key]['higher_is_better'])) == len(metrics)

# used for '123.68 116.78 103.94' style to float list
def input_to_list_float(data):
    if isinstance(data, str):
        return [float(s.strip()) for s in data.split()]

    if isinstance(data, float):
        return [data]

    assert isinstance(data, list)
    return [float(d) for d in data]

def input_to_list_bool(data):
    if isinstance(data, str):
        if ',' in data:
            return [s.strip() == 'True' for s in data.split(',')]
        else:
            return [s.strip() == 'True' for s in data.split()]

    if isinstance(data, bool):
        return [data]

    assert isinstance(data, list) and all([isinstance(i, bool) for i in data])
    return data

def input_int_to_float(data):
    if isinstance(data, str):
        # used for '123.68, 116.78, 103.94' style
        if ',' in data:
            data = data.split(',')
        # used for '123.68 116.78 103.94' style
        else:
            data = data.split()

        if len(data) == 1:
            return float(data[0].strip())
        else:
            return [float(s.strip()) for s in data]
    elif isinstance(data, list):
        return [float(s) for s in data]
    elif isinstance(data, int):
        return float(data)

def input_to_list_int(data):
    if isinstance(data, str):
        return [int(s.strip()) for s in data.split(',')]

    if isinstance(data, int):
        return [data]

    assert isinstance(data, list)
    return [int(d) for d in data]

def input_to_list(data):
    if isinstance(data, str):
        if ',' in data:
            return [s.strip() for s in data.split(',')]

        return [s.strip() for s in data.split()]

    if isinstance(data, int):
        return [data]

    assert isinstance(data, list)
    return data

def list_to_tuple(data):
    if isinstance(data, str):
        return tuple([int(s.strip()) for s in data.split(',')])

    elif isinstance(data, list):
        if isinstance(data[0], list):
            result = []
            for item in data:
                result.append(tuple([int(s) for s in item]))
            return result
        else:
            return tuple([int(s) for s in data])

def percent_to_float(data):
    if isinstance(data, str) and re.match(r'-?\d+(\.\d+)?%', data):
        data = float(data.strip('%')) / 100
    if isinstance(data, int):
        data = float(data)
    else:
        assert isinstance(data, float), 'This field should be float, int or percent string'
    return data

ops_schema = Schema({
    Optional('weight', default=None): {
        Optional('granularity'): And(
            list,
            lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
        Optional('scheme'): And(
            list,
            # asym_float are only for PyTorch framework
            lambda s: all(i in ['asym', 'sym', 'asym_float'] for i in s)),
        Optional('dtype'): And(
            list,
            lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16', 'fp16'] for i in s)),
        Optional('algorithm'): And(
            list,
            lambda s: all(i in ['minmax'] for i in s)),
        Optional('bit'):  And(
            Or(float, list),
            Use(input_to_list_float),
            lambda s: all(0.0 < i <= 7.0 for i in s))
    },
    Optional('activation', default=None): {
        Optional('granularity'): And(
            list,
            lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
        Optional('scheme'): And(
            list,
            lambda s: all(i in ['asym', 'sym'] for i in s)),
        Optional('dtype'): And(
            list,
            lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16', 'fp16'] for i in s)),
        # compute_dtypeis only for PyTorch framework
        Optional('compute_dtype', default=['uint8']): And(
            list,
            lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16', 'None'] for i in s)),
        # placeholder are only for PyTorch framework
        Optional('algorithm'): And(
            list,
            lambda s: all(i in ['minmax', 'kl', 'placeholder', 'percentile'] for i in s))
    }
})

graph_optimization_schema = Schema({

    Optional('precisions', default={'precisions': ['fp32']}): And(
        Or(str, list),
        Use(input_to_list),
        lambda s: all(i in [ 'fp32', 'bf16'] for i in s)),

    Optional('op_wise', default={'weight': {}, 'activation': {}}): {
        Optional('weight', default=None): {
            Optional('dtype', default=None): And(
                Or(str, list),
                Use(input_to_list),
                lambda s: all(i in ['fp32', 'bf16', 'fp16'] for i in s)),
        },
        Optional('activation', default=None): {
            Optional('dtype', default=None): And(
                Or(str, list),
                Use(input_to_list),
                lambda s: all(i in ['fp32', 'bf16', 'fp16'] for i in s)),
            }
    }
})

mixed_precision_schema = Schema({

    Optional('precisions', default={'precisions': ['fp32']}): And(
        Or(str, list),
        Use(input_to_list),
        lambda s: all(i in [ 'fp32', 'bf16', 'fp16'] for i in s)),

    Optional('op_wise', default={'weight': {}, 'activation': {}}): {
        Optional('weight', default=None): {
            Optional('dtype', default=None): And(
                Or(str, list),
                Use(input_to_list),
                lambda s: all(i in ['fp32', 'bf16', 'fp16'] for i in s)),
        },
        Optional('activation', default=None): {
            Optional('dtype', default=None): And(
                Or(str, list),
                Use(input_to_list),
                lambda s: all(i in ['fp32', 'bf16', 'fp16'] for i in s)),
            }
    }
})

model_conversion_schema = Schema({
    'source': And(str, lambda s: s.lower() == 'qat'),
    'destination': And(str, lambda s: s.lower() == 'default')
})

filter_schema = Schema({
    Optional('LabelBalance'): {
        'size': And(int, lambda s: s > 0)
    },
})

transform_schema = Schema({
    Optional('ResizeWithRatio'):{
        Optional('min_dim'): int,
        Optional('max_dim'): int,
        Optional('padding'): bool,
        Optional('constant_value'): int
    },
    Optional('CropToBoundingBox'): {
        'offset_height': int,
        'offset_width': int,
        'target_height': int,
        'target_width': int
    },
    Optional('Cast'): {
        Optional('dtype'): str
    },
    Optional('RandomResizedCrop'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0)),
        Optional('scale'): And(list, lambda s: all(isinstance(i, float) for i in s)),
        Optional('ratio'): And(list, lambda s: all(isinstance(i, float) for i in s)),
        Optional('interpolation'): And(
            str,
            lambda s: s in ['nearest', 'bilinear', 'bicubic']),
    },
    Optional('AlignImageChannel'): {
        Optional('dim'): int
    },
    Optional('ToNDArray'): Or({}, None),
    Optional('CropResize'): {
        'x': int,
        'y': int,
        'width': int,
        'height': int,
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0)),
        Optional('interpolation'): And(
            str,
            lambda s: s in ['nearest', 'bilinear', 'bicubic']),
    },
    Optional('RandomHorizontalFlip'): Or({}, None),
    Optional('RandomVerticalFlip'): Or({}, None),
    Optional('ToTensor'): Or({}, None),
    Optional('ToPILImage'): Or({}, None),
    Optional('Normalize'): {
        Optional('mean'): And(list, lambda s: all(isinstance(i, float) for i in s)),
        Optional('std'): And(list, lambda s: all(isinstance(i, float) for i in s)),
        Optional('rescale'): list
    },
    Optional('Resize'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0)),
        Optional('interpolation'): And(
            str,
            lambda s: s in ['nearest', 'bilinear', 'bicubic']),
    },
    Optional('RandomCrop'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0))
    },
    Optional('Rescale'): Or({}, None),
    Optional('CenterCrop'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0))
    },
    Optional('PaddedCenterCrop'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0)),
        Optional('crop_padding'): And(int, lambda s: s > 0),
    },
    Optional('ToArray'): Or({}, None),
    Optional('QuantizedInput'): {
        Optional('dtype', default='int8'): And(str, lambda s: s in ['int8', 'uint8']),
        Optional('scale'): And(float, lambda s: s > 0),
    },
    Optional('Transpose'): {
        'perm': And(list, lambda s: all(isinstance(i, int) for i in s)),
    },
    # THIS API IS TO BE DEPRECATED!
    Optional('ParseDecodeImagenet'): Or({}, None),
    Optional('ParseDecodeCoco'): Or({}, None),
    Optional('ParseDecodeVoc'): Or({}, None),
    Optional('BilinearImagenet'): {
        'height': And(int, lambda s: s > 0),
        'width': And(int, lambda s: s > 0),
        Optional('central_fraction'): float,
        Optional('mean_value'): And(Or(str, list), Use(input_to_list_float)),
        Optional('scale'): float,
    },
    Optional('ResizeCropImagenet'): {
        'height': And(int, lambda s: s > 0),
        'width': And(int, lambda s: s > 0),
        Optional('random_crop'): bool,
        Optional('slice_crop'): bool,
        Optional('resize_side'): And(int, lambda s: s > 0),
        Optional('resize_method', default='bilinear'): \
            And(str, lambda s: s in ['bilinear', 'lanczos3', 'lanczos5',
                                     'bicubic', 'gaussian', 'nearest',
                                     'area', 'mitchellcubic']),
        Optional('random_flip_left_right'): bool,
        Optional('data_format', default='channels_last'): \
            And(str, lambda s: s in ['channels_first', 'channels_last']),
        Optional('subpixels', default='RGB'): \
            And(str, lambda s: s in ['BGR', 'RGB']),
        Optional('mean_value'): And(Or(str, list), Use(input_to_list_float)),
        Optional('scale'): float,
    },
    Optional('ResizeWithAspectRatio'):{
        'height': And(int, lambda s: s > 0),
        'width': And(int, lambda s: s > 0),
    },
    Optional('ParseDecodeImagenet'): Or({}, None),
    Optional('ToArray'): Or({}, None),
    Optional('QuantizedInput'): {
        Optional('dtype', default='int8'): And(str, lambda s: s in ['int8', 'uint8']),
        Optional('scale'): And(float, lambda s: s > 0),
    },
    Optional('Transpose'): {
        'perm': And(list, lambda s: all(isinstance(i, int) for i in s)),
    },
})

postprocess_schema = Schema({
    Optional('LabelShift'):  int,
    Optional('Collect'): {
        'length': int
    },
    Optional('SquadV1'): {
        'label_file': str,
        'vocab_file': str,
        Optional('do_lower_case', default='True'): bool,
        Optional('max_seq_length', default=384): int,
    },
    Optional('SquadV1ModelZoo'): {
        'label_file': str,
        'vocab_file': str,
        Optional('do_lower_case', default='True'): bool,
        Optional('max_seq_length', default=384): int,
    },
})

dataset_schema = Schema({
    Optional('CIFAR10'): {
        'root': str,
        Optional('train'): bool,
        Optional('download'): bool,
    },
    Optional('CIFAR100'): {
        'root': str,
        Optional('train'): bool,
        Optional('download'): bool,
    },
    Optional('MNIST'): {
        'root': str,
        Optional('train'): bool,
        Optional('download'): bool,
    },
    Optional('FashionMNIST'): {
        'root': str,
        Optional('train'): bool,
        Optional('download'): bool,
    },
    Optional('ImageFolder'): {
        'root': str,
    },
    Optional('TFRecordDataset'): {
        'root': str,
    },
    Optional('ImageRecord'): {
        'root': str,
    },
    Optional('dummy_v2'): {
        'input_shape': And(Or(str, list), Use(list_to_tuple)),
        Optional('label_shape', default=[1]): And(Or(str, list), Use(list_to_tuple)),
        Optional('low'): Or(
            float,
            And(int, Use(input_int_to_float)),
            And(list, Use(input_int_to_float)),
            And(str, Use(input_int_to_float))),
        Optional('high'): Or(
            float,
            And(int, Use(input_int_to_float)),
            And(list, Use(input_int_to_float)),
            And(str, Use(input_int_to_float))),
        Optional('dtype'): And(Or(str, list), Use(input_to_list)),
    },
    Optional('sparse_dummy_v2'): {
        'dense_shape': And(Or(str, list), Use(list_to_tuple)),
        Optional('label_shape', default=[1]): And(Or(str, list), Use(list_to_tuple)),
        Optional('sparse_ratio'): Or(
            float,
            And(list, Use(input_int_to_float)),
            And(int, Use(input_int_to_float))),
        Optional('low'): Or(
            float,
            And(int, Use(input_int_to_float)),
            And(list, Use(input_int_to_float)),
            And(str, Use(input_int_to_float))),
        Optional('high'): Or(
            float,
            And(int, Use(input_int_to_float)),
            And(list, Use(input_int_to_float)),
            And(str, Use(input_int_to_float))),
        Optional('dtype'): And(Or(str, list), Use(input_to_list)),
    },

    Optional('dummy'): {
        'shape': And(Or(str, list), Use(list_to_tuple)),
        Optional('low'): Or(
            float,
            And(int, Use(input_int_to_float)),
            And(list, Use(input_int_to_float)),
            And(str, Use(input_int_to_float))),
        Optional('high'): Or(
            float,
            And(int, Use(input_int_to_float)),
            And(list, Use(input_int_to_float)),
            And(str, Use(input_int_to_float))),
        Optional('dtype'): And(Or(str, list), Use(input_to_list)),
        Optional('label'): bool,
    },
    Optional('bert'): {
        'root': str,
        'label_file': str,
        Optional('task'): And(str, lambda s: s in ["classifier", "squad"]),
        Optional('model_type'): And(str, lambda s: s in ['bert', 'xlnet', 'xlm']),
    },
    Optional('mzbert'): {
        'root': str,
        'label_file': str,
        Optional('task'): And(str, lambda s: s in ["classifier", "squad"]),
        Optional('model_type'): And(str, lambda s: s in ['bert', 'xlnet', 'xlm']),
    },
    Optional('VOCRecord'): {
        'root': str,
    },
    Optional('COCORecord'): {
        'root': str,
        Optional('num_cores'): int,
    },
    Optional('COCORaw'): {
        'root': str,
        Optional('img_dir'): str,
        Optional('anno_dir'): str,
        Optional('num_cores'): int,
    },
    Optional('COCONpy'): {
        'root': str,
        'npy_dir': str,
        Optional('anno_dir'): str,
        Optional('num_cores'): int,
    },
    Optional('ImagenetRaw'): {
        'data_path': str,
        Optional('image_list'): str,
    },
    Optional('style_transfer'): {
        'content_folder': str,
        'style_folder': str,
        Optional('crop_ratio'): float,
        Optional('resize_shape'): And(Or(str, list), Use(input_to_list_int)),
        Optional('image_format'): str,
    },
    Optional('GLUE'): {
        'data_dir': str,
        'model_name_or_path': str,
        Optional('max_seq_length'): int,
        Optional('do_lower_case'): bool,
        Optional('task'): str,
        Optional('model_type'): str,
        Optional('dynamic_length'): bool,
        Optional('evaluate'): bool
    },
    # TO BE DEPRECATED!
    Optional('Imagenet'): {
        'root': str,
    },
})

dataloader_schema = Schema({
    Optional('last_batch', default='rollover'): And(str, lambda s: s in ['rollover', 'discard']),
    Optional('batch_size', default=None): And(int, lambda s: s > 0),
    'dataset': dataset_schema,
    Optional('filter'): filter_schema,
    Optional('transform'): transform_schema,
    Optional('shuffle', default = False): And(bool, lambda s: s in [True, False]),
    Optional('distributed', default = False): And(bool, lambda s: s in [True, False]),
})

configs_schema = Schema({
    Optional('cores_per_instance'): And(int, lambda s: s > 0),
    Optional('num_of_instance', default=1): And(int, lambda s: s > 0),
    Optional('inter_num_of_threads'): And(int, lambda s: s > 0),
    Optional('intra_num_of_threads'): And(int, lambda s: s > 0),
    Optional('kmp_blocktime'): And(int, lambda s: s >= 0),
    Optional('kmp_affinity', default='granularity=fine,verbose,compact,1,0'): str,
})

optimizer_schema = Schema({
    Optional('SGD'): {
        'learning_rate': Use(float),
        Optional('momentum'): Use(float),
        Optional('nesterov'): bool,
        Optional('weight_decay'): Use(float)
    },
    Optional('AdamW'): {
        'weight_decay': Use(float),
        Optional('learning_rate', default=0.001): Use(float),
        Optional('beta_1', default=0.9): Use(float),
        Optional('beta_2', default=0.999): Use(float),
        Optional('epsilon', default=1e-07): Use(float),
        Optional('amsgrad', default=False): bool
    },
    Optional('Adam'): {
        Optional('learning_rate', default=0.001): Use(float),
        Optional('beta_1', default=0.9): Use(float),
        Optional('beta_2', default=0.999): Use(float),
        Optional('epsilon', default=1e-07): Use(float),
        Optional('amsgrad', default=False): bool
    },
})

criterion_schema = Schema({
    Optional('CrossEntropyLoss'): {
        Optional('reduction', default='mean'): \
            And(str, lambda s: s in ['none', 'sum', 'mean', 'auto', 'sum_over_batch_size']),
        Optional('from_logits', default=False):
            And(bool, lambda s: s in [True, False]),
    },
    Optional('SparseCategoricalCrossentropy'): {
        Optional('reduction', default='mean'): \
            And(str, lambda s: s in ['none', 'sum', 'mean', 'auto', 'sum_over_batch_size']),
        Optional('from_logits', default=False):
            And(bool, lambda s: s in [True, False]),
    },
    Optional('KnowledgeDistillationLoss'): {
        Optional('temperature'): And(float, lambda s: s > 0),
        Optional('loss_types'): And(list, lambda s: all(i in ['CE', 'KL', 'MSE'] for i in s)),
        Optional('loss_weights'): And(list, lambda s: all(i >= 0 for i in s) and sum(s) == 1.0),
    },
    Optional('IntermediateLayersKnowledgeDistillationLoss'): {
        'layer_mappings':
            And(Or(tuple, list), lambda s: all(len(i) in [1, 2] for i in s)),
        Optional('loss_types'):
            And(Or(tuple, list), lambda s: all(i in ['MSE', 'KL', 'L1'] for i in s)),
        Optional('loss_weights'):
            And(Or(tuple, list), lambda s: all(i >= 0 for i in s)),
        Optional('add_origin_loss'): bool,
    },
    Optional('SelfKnowledgeDistillationLoss'): {
        'layer_mappings':
            And(Or(tuple, list), lambda s: all(len(i) >= 1 for i in s)),
        Optional('loss_types'):
            And(Or(tuple, list), lambda s: all(i in ['L2', 'CE', 'KL'] for i in s)),
        Optional('loss_weights'):
            And(Or(tuple, list), lambda s: all(i >= 0.0 and i < 1.0 for i in s)),
        Optional('add_origin_loss'): bool,
        Optional('temperature'): And(float, lambda s: s > 0),
    }
})

train_schema = Schema({
    'criterion': criterion_schema,
    Optional('optimizer', default={'SGD': {'learning_rate': 0.001}}): optimizer_schema,
    Optional('dataloader'): dataloader_schema,
    Optional('epoch', default=1): int,
    Optional('start_epoch', default=0): int,
    Optional('end_epoch'): int,
    Optional('iteration'): int,
    Optional('frequency'): int,
    Optional('execution_mode', default='eager'): And(str, lambda s: s in ['eager', 'graph']),
    Optional('postprocess'): {
        Optional('transform'): postprocess_schema
    },
    # TODO reserve for multinode training support
    Optional('hostfile'): str
})

weight_compression_schema = Schema({
    Optional('initial_sparsity', default=0): And(float, lambda s: s < 1.0 and s >= 0.0),
    Optional('target_sparsity', default=0.97): float,
    Optional('max_sparsity_ratio_per_layer', default=0.98):float,
    Optional('prune_type', default="basic_magnitude"): str,
    Optional('start_epoch', default=0): int,
    Optional('end_epoch', default=4): int,
    Optional('start_step', default=0): int,
    Optional('end_step', default=0): int,
    Optional('update_frequency', default=1.0): float,
    Optional('update_frequency_on_step', default=1):int,
    Optional('excluded_names', default=[]):list,
    Optional('prune_domain', default="global"): str,
    Optional('names', default=[]): list,
    Optional('extra_excluded_names', default=None): list,
    Optional('prune_layer_type', default=None): list,
    Optional('sparsity_decay_type', default="exp"): str,
    Optional('pattern', default="tile_pattern_1x1"): str,

    Optional('pruners'): And(list, \
                               lambda s: all(isinstance(i, Pruner) for i in s))
})

weight_compression_v2_schema = Schema({
    Optional('target_sparsity', default=0.9): float,
    Optional('pruning_type', default="snip_momentum"): str,
    Optional('pattern', default="4x1"): str,
    Optional('op_names', default=[]): list,
    Optional('excluded_op_names', default=[]): list,
    Optional('start_step', default=0): int,
    Optional('end_step', default=0): int,
    Optional('pruning_scope', default="global"): str,
    Optional('pruning_frequency', default=1): int,
    Optional('min_sparsity_ratio_per_op', default=0.0): float,
    Optional('max_sparsity_ratio_per_op', default=0.98): float,
    Optional('sparsity_decay_type', default="exp"): str,
    Optional('pruning_op_types', default=['Conv', 'Linear']): list,
    Optional('reg_type', default=None): str,
    Optional('criterion_reduce_type', default="mean"): str,
    Optional('parameters', default={"reg_coeff": 0.0}): dict,
    Optional('resume_from_pruned_checkpoint', default=False): str,
    Optional('pruners'): And(list, \
                             lambda s: all(isinstance(i, Pruner) for i in s))
})


# weight_compression_pytorch_schema = Schema({},ignore_extra_keys=True)

approach_schema = Schema({
    Hook('weight_compression', handler=_valid_prune_sparsity): object,
    Hook('weight_compression_pytorch', handler=_valid_prune_sparsity): object,
    Optional('weight_compression'): weight_compression_schema,
    Optional('weight_compression_v2'): weight_compression_v2_schema,
    Optional('weight_compression_pytorch'): weight_compression_schema,
})

default_workspace = './nc_workspace/{}/'.format(
                                           datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

COCOmAP_input_order_schema = Schema({
    Optional('num_detections'): int,
    'boxes': int,
    'scores': int,
    'classes': int
})

schema = Schema({
    'model': {
        'name': str,
        'framework': And(str, lambda s: s in list(FRAMEWORKS.keys()) + ['NA']),
        Optional('inputs', default=[]): And(Or(str, list), Use(input_to_list)),
        Optional('outputs', default=[]): And(Or(str, list), Use(input_to_list)),
    },
    Optional('version', default=float(__version__.split('.')[0])): Or(float,
                                                                      And(int, Use(input_int_to_float)),
                                                                      And(str, Use(input_int_to_float))),
    Optional('device', default='cpu'): And(str, lambda s: s in ['cpu', 'gpu']),
    Optional('quantization', default={'approach': 'post_training_static_quant', \
                                      'calibration': {'sampling_size': [100]}, \
                                      'recipes': {'scale_propagation_max_pooling': True,
                                                      'scale_propagation_concat': True,
                                                      'first_conv_or_matmul_quantization': True,
                                                      'last_conv_or_matmul_quantization': True,
                                                      'pre_post_process_quantization': True},
                                      'model_wise': {'weight': {'bit': [7.0]},
                                                     'activation': {}},
                                      'quant_level': "auto",
                                      }): {
        Optional('approach', default='post_training_static_quant'): And(
            str,
            # TODO check if framework support dynamic quantize
            # Now only onnruntime and pytorch support
            lambda s: s in ['post_training_static_quant',
                            'post_training_dynamic_quant',
                            'post_training_auto_quant',
                            'quant_aware_training']),
        Optional('train', default=None): train_schema,
        Optional('advance', default=None): {
            Optional('bias_correction'): And(str, lambda s: s in ['weight_empirical']),
        },
        Optional('calibration', default={'sampling_size': [100]}): {
            Optional('sampling_size', default=[100]): And(Or(str, int, list), Use(input_to_list)),
            Optional('dataloader', default=None): dataloader_schema
        },
        Optional('recipes', default={'scale_propagation_max_pooling': True,
                                         'scale_propagation_concat': True,
                                         'first_conv_or_matmul_quantization': True,
                                         'last_conv_or_matmul_quantization': True,
                                         'pre_post_process_quantization': True}): {
            Optional('scale_propagation_max_pooling', default=True):
                    And(bool, lambda s: s in [True, False]),
            Optional('scale_propagation_concat', default=True):
                    And(bool, lambda s: s in [True, False]),
            Optional('first_conv_or_matmul_quantization', default=True):
                    And(bool, lambda s: s in [True, False]),
            Optional('last_conv_or_matmul_quantization', default=True):
                    And(bool, lambda s: s in [True, False]),
            Optional('pre_post_process_quantization', default=True):
                    And(bool, lambda s: s in [True, False]),
            Optional('fast_bias_correction', default=False):
                    And(bool, lambda s: s in [True, False]),
            Optional('weight_correction', default=False):
                    And(bool, lambda s: s in [True, False]),
        },
        Optional('model_wise', default={'weight': {'bit': [7.0]}, 'activation': {}}): {
            Optional('weight', default= {'bit': [7.0]}): {
                Optional('granularity', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
                # asym_float are only for PyTorch framework
                Optional('scheme', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['asym', 'sym', 'asym_float'] for i in s)),
                Optional('dtype', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16', 'fp16'] for i in s)),
                Optional('algorithm', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['minmax'] for i in s)),
                Optional('bit', default=[7.0]):  And(
                    Or(float, list),
                    Use(input_to_list_float),
                    lambda s: all(0.0 < i <= 7.0 for i in s))

            },
            Optional('activation', default=None): {
                Optional('granularity', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
                Optional('scheme', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['asym', 'sym'] for i in s)),
                Optional('dtype', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16', 'fp16'] for i in s)),
                # compute_dtypeis only for PyTorch framework
                Optional('compute_dtype', default=['uint8']): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16', 'None'] for i in s)),
                # placeholder are only for PyTorch framework
                Optional('algorithm', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['minmax', 'kl', 'placeholder', 'percentile'] for i in s)),
            }
        },
        Optional('optype_wise', default=None): {
            str: ops_schema
        },
        Optional('op_wise', default=None): {
            str: ops_schema
        },
        Optional('quant_level', default="auto"): And(Or(str, int), lambda level: level in ["auto", 0, 1]),
    },
    Optional('use_bf16', default=True): bool,
    Optional('graph_optimization'): graph_optimization_schema,
    Optional('mixed_precision'): mixed_precision_schema,

    Optional('model_conversion'): model_conversion_schema,

    Optional('tuning', default={
        'strategy': {'name': 'basic'},
        'accuracy_criterion': {'relative': 0.01, 'higher_is_better': True},
        'objective': 'performance',
        'exit_policy': {'timeout': 0, 'max_trials': 100, 'performance_only': False},
        'random_seed': 1978, 'tensorboard': False,
        'workspace': {'path': default_workspace},
        'diagnosis': False,
        }): {
        Optional('strategy', default={'name': 'basic'}): {
            'name': And(str, lambda s: s in EXP_STRATEGIES),
            Optional('sigopt_api_token'): str,
            Optional('sigopt_project_id'): str,
            Optional('sigopt_experiment_name', default='nc-tune'): str,
            Optional('accuracy_weight', default=1.0): float,
            Optional('latency_weight', default=1.0): float,
            Optional('confidence_batches', default=2): int,
            Optional('hawq_v2_loss', default=None): object,
        } ,
        Hook('accuracy_criterion', handler=_valid_accuracy_field): object,
        Optional('accuracy_criterion', default={'relative': 0.01}): {
            Optional('relative'): And(Or(str, float), Use(percent_to_float)),
            Optional('absolute'): And(Or(str, int, float), Use(percent_to_float)),
            Optional('higher_is_better', default=True): bool,
        },
        Optional('objective', default='performance'): And(str, lambda s: s in OBJECTIVES),
        Hook('multi_objectives', handler=_valid_multi_objectives): object,
        Optional('multi_objectives'):{
            Optional('objective'): And(
                Or(str, list), Use(input_to_list), lambda s: all(i in OBJECTIVES for i in s)),
            Optional('weight'): And(Or(str, list), Use(input_to_list_float)),
            Optional('higher_is_better'): And(
                Or(str, bool, list), Use(input_to_list_bool)),
        },
        Optional('exit_policy', default={'timeout': 0,
                                         'max_trials': 100,
                                         'performance_only': False}): {
            Optional('timeout', default=0): int,
            Optional('max_trials', default=100): int,
            Optional('performance_only', default=False): bool,
        },
        Optional('random_seed', default=1978): int,
        Optional('tensorboard', default=False): And(bool, lambda s: s in [True, False]),
        Optional('workspace', default={'path': default_workspace}): {
            Optional('path', default=None): str,
            Optional('resume'): str
        },
        Optional('diagnosis', default=False): And(bool, lambda s: s in [True, False]),
    },
    Optional('evaluation'): {
        Hook('accuracy', handler=_valid_multi_metrics): object,
        Optional('accuracy'): {
            Hook('multi_metrics', handler=_valid_metric_length): object,
            Optional('multi_metrics', default=None): {
                Optional('weight'): And(Or(str, list), Use(input_to_list_float)),
                Optional('higher_is_better'): And(
                    Or(str, list), Use(input_to_list_bool)),
                Optional('topk'): And(int, lambda s: s in [1, 5]),
                Optional('mAP'): {
                    Optional('anno_path'): str,
                    Optional('iou_thrs', default=0.5):
                            Or(And(str, lambda s: s in ['0.5:0.05:0.95']),
                               And(float, lambda s: s <= 1.0 and s >= 0.0)),
                    Optional('map_points', default=0): And(int, lambda s: s in [0, 11, 101])
                },
                Optional('COCOmAP'): {
                    Optional('anno_path'): str,
                    Optional('map_key', default='DetectionBoxes_Precision/mAP'): str
                },
                Optional('COCOmAPv2'): {
                    Optional('anno_path'): str,
                    Optional('map_key', default='DetectionBoxes_Precision/mAP'): str,
                    Optional('output_index_mapping', default={'num_detections': -1,
                                                      'boxes': 0,
                                                      'scores': 1,
                                                      'classes': 2}): COCOmAP_input_order_schema
                },
                Optional('VOCmAP'): {
                    Optional('anno_path'): str
                },
                Optional('SquadF1'): Or({}, None),
                Optional('MSE'): {
                    Optional('compare_label'): bool
                },
                Optional('RMSE'): {
                    Optional('compare_label'): bool
                },
                Optional('MAE'): {
                    Optional('compare_label'): bool
                },
                Optional('Accuracy'): Or({}, None),
                Optional('Loss'): Or({}, None),
                Optional('BLEU'): Or({}, None),
                Optional('SquadF1'): Or({}, None),
                Optional('F1'): Or({}, None),
                Optional('mIOU'): {
                    Optional('num_classes'): int
                },
                Optional('GLUE'): {
                    Optional('task'): str
                },
                Optional('ROC'): {
                    Optional('task'): str
                },
            },
            Optional('metric', default=None): {
                Optional('topk'): And(int, lambda s: s in [1, 5]),
                Optional('mAP'): {
                    Optional('anno_path'): str,
                    Optional('iou_thrs', default=0.5):
                            Or(And(str, lambda s: s in ['0.5:0.05:0.95']),
                               And(float, lambda s: s <= 1.0 and s >= 0.0)),
                    Optional('map_points', default=0): And(int, lambda s: s in [0, 11, 101])
                },
                Optional('COCOmAP'): {
                    Optional('anno_path'): str,
                    Optional('map_key', default='DetectionBoxes_Precision/mAP'): str
                },
                Optional('COCOmAPv2'): {
                    Optional('anno_path'): str,
                    Optional('map_key', default='DetectionBoxes_Precision/mAP'): str,
                    Optional('output_index_mapping', default={'num_detections': -1,
                                                      'boxes': 0,
                                                      'scores': 1,
                                                      'classes': 2}): COCOmAP_input_order_schema
                },
                Optional('VOCmAP'): {
                    Optional('anno_path'): str
                },
                Optional('SquadF1'): Or({}, None),
                Optional('MSE'): {
                    Optional('compare_label'): bool
                },
                Optional('RMSE'): {
                    Optional('compare_label'): bool
                },
                Optional('MAE'): {
                    Optional('compare_label'): bool
                },
                Optional('Accuracy'): Or({}, None),
                Optional('Loss'): Or({}, None),
                Optional('BLEU'): Or({}, None),
                Optional('SquadF1'): Or({}, None),
                Optional('F1'): Or({}, None),
                Optional('mIOU'): {
                    Optional('num_classes'): int
                },
                Optional('GLUE'): {
                    Optional('task'): str
                },
                Optional('ROC'): {
                    Optional('task'): str
                },
            },
            Optional('configs'): configs_schema,
            Optional('iteration', default=-1): int,
            Optional('dataloader'): dataloader_schema,
            Optional('postprocess'): {
                Optional('transform'): postprocess_schema
            },
        },
        Optional('performance'): {
            Optional('warmup', default=5): int,
            Optional('iteration', default=-1): int,
            Optional('configs'): configs_schema,
            Optional('dataloader'): dataloader_schema,
            Optional('postprocess'): {
                Optional('transform'): postprocess_schema
            }
        },
        Optional('diagnosis', default=False): And(bool, lambda s: s in [True, False]),
    },
    Optional('pruning'): {
        Hook('train', handler=_valid_prune_epoch): object,
        Optional("train"): train_schema,
        Optional("approach"): approach_schema
    },

    Optional('distillation'): {
        Optional("train"): train_schema
    },

    Optional('auto_distillation'): {
        "search": {
            "search_space": dict,
            Optional("search_algorithm", default=""): str,
            Optional("metrics", default=[]): list,
            Optional("higher_is_better", default=[]): list,
            Optional("max_trials", default=1): int,
            Optional("seed", default=42): int,
            },
        Optional("flash_distillation"): {
            Optional("knowledge_transfer"): {
                Optional("block_names", default=[]): list,
                "layer_mappings_for_knowledge_transfer": list,
                Optional("loss_types", default=[]): list,
                Optional("loss_weights", default=[]): list,
                Optional("add_origin_loss", default=[]): list,
                Optional("train_steps", default=[]): list,
                },
            Optional("regular_distillation"): {
                Optional("block_names", default=[]): list,
                "layer_mappings_for_knowledge_transfer": list,
                Optional("loss_types", default=[]): list,
                Optional("loss_weights", default=[]): list,
                Optional("add_origin_loss", default=[]): list,
                Optional("train_steps", default=[]): list,
                },
            },
    },

    Optional('nas'): {
        Optional("approach", default=None): Or(str, None),
        Optional("search"): {
            Optional("search_space", default=None): Or(dict, None),
            Optional("search_algorithm", default=None): Or(str, None),
            Optional("metrics", default=None): list,
            Optional("higher_is_better", default=None): list,
            Optional("max_trials", default=None): int,
            Optional("seed", default=42): int,
            },
        Optional("dynas"): {
            Optional("supernet", default=None): str,
            Optional("metrics", default=None): list,
            Optional("population", default=50): int,
            Optional("num_evals", default=100000): int,
            Optional("results_csv_path", default=None): str,
            Optional("dataset_path", default=None): str,
            Optional("supernet_ckpt_path", default=None): str,
            Optional("batch_size", default=64): int,
            Optional("eval_batch_size", default=64): int,
            Optional("num_workers", default=20): int,
            Optional("distributed", default=False): bool,
            Optional("test_fraction", default=1.0): float,
            },
    },

    Optional("train"): train_schema
})

quantization_default_schema = Schema({
    Optional('model', default={'name': 'default_model_name', \
                               'framework': 'NA', \
                                'inputs': [], 'outputs': []}): dict,

    Optional('version', default=float(__version__.split('.')[0])): str,

    Optional('device', default='cpu'): str,

    Optional('quantization', default={'approach': 'post_training_static_quant', \
                                      'calibration': {'sampling_size': [100]},
                                      'recipes': {'scale_propagation_max_pooling': True,
                                                      'scale_propagation_concat': True,
                                                      'first_conv_or_matmul_quantization': True,
                                                      'last_conv_or_matmul_quantization': True,
                                                      'pre_post_process_quantization': True},
                                      'model_wise': {'weight': {'bit': [7.0]},
                                                     'activation': {}},
                                      'quant_level': "auto",
                                    }): dict,
    Optional('use_bf16', default=False): bool,
    Optional('tuning', default={
        'strategy': {'name': 'basic'},
        'accuracy_criterion': {'relative': 0.01, 'higher_is_better': True},
        'objective': 'performance',
        'exit_policy': {'timeout': 0, 'max_trials': 100, 'performance_only': False},
        'random_seed': 1978, 'tensorboard': False,
        'workspace': {'path': default_workspace}}): dict,

    Optional('evaluation', default={'accuracy': {'metric': {'topk': 1}}}): dict
})

pruning_default_schema = Schema({
    Optional('model', default={'name': 'default_model_name', \
                               'framework': 'NA', \
                                'inputs': [], 'outputs': []}): dict,

    Optional('version', default=float(__version__.split('.')[0])): str,

    Optional('device', default='cpu'): str,

    Optional('use_bf16', default=False): bool,

    Optional('tuning', default={
        'random_seed': 1978, 'tensorboard': False,
        'workspace': {'path': default_workspace}}): dict,

    Optional('pruning', default={'approach': {'weight_compression':{'initial_sparsity': 0.0, \
                                            'target_sparsity': 0.97, 'start_epoch': 0, \
                                            'end_epoch': 4}}}): dict,

    Optional('evaluation', default={'accuracy': {'metric': {'topk': 1}}}): dict
})

graph_optimization_default_schema = Schema({
    Optional('model', default={'name': 'resnet50', \
                               'framework': 'NA', \
                                'inputs': [], 'outputs': []}): dict,

    Optional('version', default=float(__version__.split('.')[0])): str,

    Optional('device', default='cpu'): str,

    Optional('quantization', default={'approach': 'post_training_static_quant',
                                    'calibration': {'sampling_size': [100]},
                                    'recipes': {'scale_propagation_max_pooling': True,
                                                    'scale_propagation_concat': True,
                                                    'first_conv_or_matmul_quantization': True,
                                                    'last_conv_or_matmul_quantization': True,
                                                    'pre_post_process_quantization': True},
                                    'model_wise': {'weight': {'bit': [7.0]},
                                                    'activation': {}}}): dict,

    Optional('use_bf16', default=False): bool,

    Optional('tuning', default={
        'strategy': {'name': 'basic'},
        'accuracy_criterion': {'relative': 0.01, 'higher_is_better': True},
        'objective': 'performance',
        'exit_policy': {'timeout': 0, 'max_trials': 100, 'performance_only': False},
        'random_seed': 1978, 'tensorboard': False,
        'workspace': {'path': default_workspace}}): dict,

    Optional('evaluation', default={'accuracy': {'metric': {'topk': 1}}}): dict,

    Optional('graph_optimization', default={'precisions': ['bf16, fp32']}): dict
})

mixed_precision_default_schema = Schema({
    Optional('model', default={'name': 'resnet50', \
                               'framework': 'NA', \
                                'inputs': [], 'outputs': []}): dict,

    Optional('version', default=float(__version__.split('.')[0])): str,

    Optional('device', default='cpu'): str,

    Optional('quantization', default={'approach': 'post_training_static_quant',
                                    'calibration': {'sampling_size': [100]},
                                    'recipes': {'scale_propagation_max_pooling': True,
                                                    'scale_propagation_concat': True,
                                                    'first_conv_or_matmul_quantization': True,
                                                    'last_conv_or_matmul_quantization': True,
                                                    'pre_post_process_quantization': True},
                                    'model_wise': {'weight': {'bit': [7.0]},
                                                    'activation': {}}}): dict,

    Optional('use_bf16', default=False): bool,

    Optional('tuning', default={
        'strategy': {'name': 'basic'},
        'accuracy_criterion': {'relative': 0.01, 'higher_is_better': True},
        'objective': 'performance',
        'exit_policy': {'timeout': 0, 'max_trials': 100, 'performance_only': False},
        'random_seed': 1978, 'tensorboard': False,
        'workspace': {'path': default_workspace}}): dict,

    Optional('evaluation', default={'accuracy': {'metric': {'topk': 1}}}): dict,

    Optional('mixed_precision', default={'precisions': ['bf16, fp32']}): dict
})

benchmark_default_schema = Schema({
    Optional('model', default={'name': 'resnet50', \
                               'framework': 'NA', \
                                'inputs': [], 'outputs': []}): dict,

    Optional('version', default=float(__version__.split('.')[0])): str,

    Optional('device', default='cpu'): str,

    Optional('use_bf16', default=False): bool,

    Optional('quantization', default={'approach': 'post_training_static_quant',
                                    'calibration': {'sampling_size': [100]},
                                    'recipes': {'scale_propagation_max_pooling': True,
                                                    'scale_propagation_concat': True,
                                                    'first_conv_or_matmul_quantization': True,
                                                    'last_conv_or_matmul_quantization': True,
                                                    'pre_post_process_quantization': True},
                                    'model_wise': {'weight': {'bit': [7.0]},
                                                    'activation': {}}}): dict,

    Optional('tuning', default={
        'strategy': {'name': 'basic'},
        'accuracy_criterion': {'relative': 0.01, 'higher_is_better': True},
        'objective': 'performance',
        'exit_policy': {'timeout': 0, 'max_trials': 100, 'performance_only': False},
        'random_seed': 1978, 'tensorboard': False,
        'workspace': {'path': default_workspace}}): dict,

    Optional('evaluation', default={'accuracy': {'metric': {'topk': 1}}}): dict
})

distillation_default_schema = Schema({
    Optional('model', default={'name': 'default_model_name', \
                               'framework': 'NA', \
                               'inputs': [], 'outputs': []}): dict,

    Optional('version', default=float(__version__.split('.')[0])): str,

    Optional('device', default='cpu'): str,

    Optional('use_bf16', default=False): bool,

    Optional('tuning', default={
        'random_seed': 1978, 'tensorboard': False,
        'workspace': {'path': default_workspace}}): dict,

    Optional('distillation', default={
        'train': {'start_epoch': 0, 'end_epoch': 10,
                  'iteration': 1000, 'frequency': 1,
                  'optimizer': {'SGD': {'learning_rate': 0.001}},
                  'criterion': {'KnowledgeDistillationLoss':
                                 {'temperature': 1.0,
                                  'loss_types': ['CE', 'KL'],
                                  'loss_weights': [0.5, 0.5]}}}}): dict,

    Optional('evaluation', default={'accuracy': {'metric': {'topk': 1}}}): dict

})

class Conf(object):
    """Config parser.

    Args:
        cfg_fname (string): The path to the configuration file.
    """
    def __init__(self, cfg_fname):
        assert cfg_fname is not None
        self.usr_cfg = DotDict(self._read_cfg(cfg_fname))

    def _read_cfg(self, cfg_fname):
        """Load a config file following yaml syntax.

        Args:
            cfg_fname(string): The name of configuration yaml file
        """
        try:
            with open(cfg_fname, 'r') as f:
                content = f.read()
                cfg = yaml.safe_load(content)
                validated_cfg = schema.validate(cfg)

            # if user yaml doesn't include version field, neural_compressor will write a supported version
            # into it.
            if 'version' not in cfg:
                leading_whitespace = re.search(r"[ \t]*model\s*:",
                                               content).group().split("model")[0]
                content = re.sub(r'model\s*:',
                                 'version: {}\n\n{}model:'.format(
                                                               float(__version__.split('.')[0]),
                                                               leading_whitespace
                                                           ),
                                 content)
                with open(cfg_fname, 'w') as f:
                    f.write(content)

            return validated_cfg
        except FileNotFoundError as f:
            logger.error("{}.".format(f))
            raise RuntimeError(
                "The yaml file is not exist. Please check the file name or path."
            )
        except Exception as e:
            logger.error("{}.".format(e))
            raise RuntimeError(
                "The yaml file format is not correct. Please refer to document."
            )

    def map_pyconfig_to_cfg(self, pythonic_config):
        mapping = {}
        if pythonic_config.quantization is not None:
            mapping.update({
                'device': pythonic_config.quantization.device,
                'model.inputs': pythonic_config.quantization.inputs,
                'model.outputs': pythonic_config.quantization.outputs,
                'model.backend': pythonic_config.quantization.backend,
                'model.quant_format': pythonic_config.quantization.quant_format,
                'model.domain': pythonic_config.quantization.domain,
                'quantization.recipes': pythonic_config.quantization.recipes,
                'quantization.approach': pythonic_config.quantization.approach,
                'quantization.example_inputs': pythonic_config.quantization.example_inputs,
                'quantization.calibration.sampling_size':
                    pythonic_config.quantization.calibration_sampling_size,
                'quantization.optype_wise': pythonic_config.quantization.op_type_dict,
                'quantization.op_wise': pythonic_config.quantization.op_name_dict,
                'tuning.strategy.name': pythonic_config.quantization.strategy,
                'tuning.accuracy_criterion.relative':
                    pythonic_config.quantization.accuracy_criterion.relative,
                'tuning.accuracy_criterion.absolute':
                    pythonic_config.quantization.accuracy_criterion.absolute,
                'tuning.accuracy_criterion.higher_is_better':
                    pythonic_config.quantization.accuracy_criterion.higher_is_better,
                'tuning.objective': pythonic_config.quantization.objective,
                'tuning.exit_policy.timeout': pythonic_config.quantization.timeout,
                'tuning.exit_policy.max_trials': pythonic_config.quantization.max_trials,
                'tuning.exit_policy.performance_only': pythonic_config.quantization.performance_only,
                'use_bf16': pythonic_config.quantization.use_bf16,
                'quantization.quant_level': pythonic_config.quantization.quant_level,
                'reduce_range': pythonic_config.quantization.reduce_range
            })

            if pythonic_config.quantization.diagnosis:
                mapping.update({
                    'tuning.diagnosis': True,
                    'tuning.exit_policy.max_trials': 1,
                })

            if pythonic_config.quantization.strategy_kwargs:
                st_kwargs = pythonic_config.quantization.strategy_kwargs
                for st_key in ['sigopt_api_token', 'sigopt_project_id', 'sigopt_experiment_name', \
                    'accuracy_weight', 'latency_weight', 'hawq_v2_loss', 'confidence_batches']:
                    if st_key in st_kwargs:
                        st_val =  st_kwargs[st_key]
                        mapping.update({'tuning.strategy.' + st_key: st_val})

        if pythonic_config.distillation is not None:
            mapping.update({
                'distillation.train.criterion': pythonic_config.distillation.criterion,
                'distillation.train.optimizer': pythonic_config.distillation.optimizer,
            })
        if pythonic_config.pruning is not None:
            mapping.update({
                'pruning.approach.weight_compression': pythonic_config.pruning.weight_compression,
            })
        if pythonic_config.nas is not None:
            mapping.update({
                'nas.approach': pythonic_config.nas.approach,
                'nas.search': pythonic_config.nas.search,
                'nas.dynas': pythonic_config.nas.dynas,
            })
        if pythonic_config.options is not None:
            mapping.update({
                'tuning.random_seed': pythonic_config.options.random_seed,
                'tuning.workspace.path': pythonic_config.options.workspace,
                'tuning.workspace.resume': pythonic_config.options.resume_from,
                'tuning.tensorboard': pythonic_config.options.tensorboard,
            })
        if pythonic_config.benchmark is not None:
            if pythonic_config.benchmark.inputs != []:
                mapping.update({'model.inputs': pythonic_config.benchmark.inputs})
            if pythonic_config.benchmark.outputs != []:
                mapping.update({'model.outputs': pythonic_config.benchmark.outputs})
            mapping.update({
                'evaluation.performance.warmup': pythonic_config.benchmark.warmup,
                'evaluation.performance.iteration': pythonic_config.benchmark.iteration,
                'evaluation.performance.configs.cores_per_instance':
                    pythonic_config.benchmark.cores_per_instance,
                'evaluation.performance.configs.num_of_instance':
                    pythonic_config.benchmark.num_of_instance,
                'evaluation.performance.configs.inter_num_of_threads':
                    pythonic_config.benchmark.inter_num_of_threads,
                'evaluation.performance.configs.intra_num_of_threads':
                    pythonic_config.benchmark.intra_num_of_threads,
                'evaluation.accuracy.configs.cores_per_instance':
                    pythonic_config.benchmark.cores_per_instance,
                'evaluation.accuracy.configs.num_of_instance':
                    pythonic_config.benchmark.num_of_instance,
                'evaluation.accuracy.configs.inter_num_of_threads':
                    pythonic_config.benchmark.inter_num_of_threads,
                'evaluation.accuracy.configs.intra_num_of_threads':
                    pythonic_config.benchmark.intra_num_of_threads,
            })
            if pythonic_config.benchmark.diagnosis:
                mapping.update({'evaluation.diagnosis': pythonic_config.benchmark.diagnosis})

            if "model.backend" not in mapping:
                mapping.update({
                    'model.backend': pythonic_config.benchmark.backend,
                })
            else:
                if mapping['model.backend'] == 'default' and \
                        pythonic_config.benchmark.backend != 'default':
                    mapping.update({
                        'model.backend': pythonic_config.benchmark.backend,
                    })

        if "model.backend" not in mapping:
            mapping.update({
                'model.backend': "default",
            })

        for k, v in mapping.items():
            if k in ['tuning.accuracy_criterion.relative', 'tuning.accuracy_criterion.absolute']:
                target_key = str(pythonic_config.quantization.accuracy_criterion)
                if target_key not in k and 'accuracy_criterion' in self.usr_cfg.tuning:
                    if target_key in self.usr_cfg.tuning.accuracy_criterion and \
                                    k.split('.')[-1] in self.usr_cfg.tuning.accuracy_criterion:
                        self.usr_cfg.tuning.accuracy_criterion.pop(k.split('.')[-1])
                    continue
            if v is not None:
                deep_set(self.usr_cfg, k, v)

    def _convert_cfg(self, src, dst):
        """Helper function to merge user defined dict into default dict.

           If the key in src doesn't exist in dst, then add this key and value
           pair to dst.
           If the key in src is in dst, then override the value in dst with the
           value in src.

        Args:
            src (dict): The source dict merged from
            dst (dict): The source dict merged to

        Returns:
            dict: The merged dict from src to dst
        """
        for key in src:
            if key in dst:
                if isinstance(dst[key], dict) and isinstance(src[key], dict):
                    if key in ['accuracy_criterion', 'metric', 'dataset',
                        'criterion', 'optimizer']:
                        # accuracy_criterion can only have one of absolute and relative
                        # others can only have one item
                        inter_key = src[key].keys() & dst[key].keys()-{'higher_is_better'}
                        if len(inter_key) == 0:
                            dst[key] = {}
                    if key == 'accuracy' and src[key].get('multi_metrics', None):
                        dst[key].pop('metric', None)
                    self._convert_cfg(src[key], dst[key])
                elif dst[key] == src[key]:
                    pass  # same leaf value
                else:
                    dst[key] = src[key]
            elif isinstance(src[key], dict):
                dst[key] = DotDict(self._convert_cfg(src[key], {}))
            else:
                dst[key] = src[key]
        return dst

class Quantization_Conf(Conf):
    """Config parser.

    Args:
        cfg: The path to the configuration file or DotDict object or None.
    """

    def __init__(self, cfg=None):
        if isinstance(cfg, str):
            self.usr_cfg = DotDict(self._read_cfg(cfg))
        elif isinstance(cfg, DotDict):
            self.usr_cfg = DotDict(schema.validate(self._convert_cfg(
                cfg, copy.deepcopy(quantization_default_schema.validate(dict())))))
        else:
            self.usr_cfg = DotDict(quantization_default_schema.validate(dict()))
        self._model_wise_tune_space = None
        self._opwise_tune_space = None

    def _merge_dicts(self, src, dst):
        """Helper function to merge src dict into dst dict.

           If the key in src doesn't exist in dst, then skip
           If the key in src is in dst and the value intersects with the one in
           dst, then override the value in dst with the intersect value.

        Args:
            src (dict): The source dict merged from
            dst (dict): The source dict merged to

        Returns:
            dict: The merged dict from src to dst
        """
        for key in src:
            if key in dst:
                if isinstance(dst[key], dict) and isinstance(src[key], dict):
                    self._merge_dicts(src[key], dst[key])
                elif dst[key] == src[key] or src[key] is None:
                    pass  # same leaf value
                else:
                    value = [value for value in src[key]
                             if value in dst[key] or isinstance(value, float)]
                    if value != []:
                        dst[key] = value

        return dst

    def modelwise_tune_space(self, model_wise_quant):
        cfg = self.usr_cfg

        self._model_wise_tune_space = OrderedDict()
        for optype in model_wise_quant.keys():
            if cfg.quantization.optype_wise and optype in cfg.quantization.optype_wise:
                self._model_wise_tune_space[optype] = self._merge_dicts(
                    cfg.quantization.optype_wise[optype],
                    model_wise_quant[optype])
            else:
                self._model_wise_tune_space[optype] = self._merge_dicts(
                    cfg.quantization.model_wise,
                    model_wise_quant[optype])

        return self._model_wise_tune_space

class Pruning_Conf(Conf):
    """Config parser.

    Args:
        cfg: The path to the configuration file or DotDict object or None.
    """

    def __init__(self, cfg=None):
        if isinstance(cfg, str):
            self._read_cfg(cfg)
            self.usr_cfg = DotDict(self._read_cfg(cfg))
        elif isinstance(cfg, DotDict):
            self.usr_cfg = DotDict(schema.validate(self._convert_cfg(
                cfg, copy.deepcopy(pruning_default_schema.validate(dict())))))
        else:
            self.usr_cfg = DotDict(pruning_default_schema.validate(dict()))

class Graph_Optimization_Conf(Quantization_Conf):
    """Config parser.

    Args:
        cfg: The path to the configuration file or DotDict object or None.
    """

    def __init__(self, cfg=None):
        if isinstance(cfg, str):
            self.usr_cfg = DotDict(self._read_cfg(cfg))
        elif isinstance(cfg, DotDict):
            self.usr_cfg = DotDict(schema.validate(self._convert_cfg(
                cfg, copy.deepcopy(graph_optimization_default_schema.validate(dict())))))
        else:
            self.usr_cfg = DotDict(graph_optimization_default_schema.validate(dict()))

class MixedPrecision_Conf(Quantization_Conf):
    """Config parser.

    Args:
        cfg: The path to the configuration file or DotDict object or None.
    """

    def __init__(self, cfg=None):
        if isinstance(cfg, str):
            self.usr_cfg = DotDict(self._read_cfg(cfg))
        elif isinstance(cfg, DotDict):
            self.usr_cfg = DotDict(self._convert_cfg(
                cfg, copy.deepcopy(mixed_precision_default_schema.validate(dict()))))
        else:
            self.usr_cfg = DotDict(mixed_precision_default_schema.validate(dict()))

class Benchmark_Conf(Conf):
    """Config parser.

    Args:
        cfg: The path to the configuration file or DotDict object or None.
    """

    def __init__(self, cfg=None):
        if isinstance(cfg, str):
            self.usr_cfg = DotDict(self._read_cfg(cfg))
        elif isinstance(cfg, DotDict):
            self.usr_cfg = DotDict(schema.validate(self._convert_cfg(
                cfg, copy.deepcopy(benchmark_default_schema.validate(dict())))))
        else:
            self.usr_cfg = DotDict(benchmark_default_schema.validate(dict()))

class Distillation_Conf(Conf):
    """Config parser.

    Args:
        cfg: The path to the configuration file or DotDict object or None.
    """

    def __init__(self, cfg=None):
        if isinstance(cfg, str):
            self.usr_cfg = DotDict(self._read_cfg(cfg))
        elif isinstance(cfg, DotDict):
            self.usr_cfg = DotDict(schema.validate(self._convert_cfg(
                cfg, copy.deepcopy(distillation_default_schema.validate(dict())))))
        else:
            self.usr_cfg = DotDict(distillation_default_schema.validate(dict()))

class NASConfig(Conf):
    """Config parser.

    Args:
        approach: The approach of the NAS.
        search_algorithm: The search algorithm for NAS procedure.
    """

    def __init__(self, approach=None, search_space=None, search_algorithm=None):
        assert approach is None or (isinstance(approach, str) and approach), \
            "Should set approach with a string."
        usr_cfg = {
            'model': {'name': 'nas', 'framework': 'NA'},
            'nas': {'approach': approach,
                    'search': {
                        'search_space': search_space,
                        'search_algorithm': search_algorithm}
                    },
        }
        self.usr_cfg = DotDict(usr_cfg)
        if approach and approach != 'basic':
            self.usr_cfg.nas[approach] = DotDict({})
            self.__setattr__(approach, self.usr_cfg.nas[approach])

    def validate(self):
        self.usr_cfg = schema.validate(self.usr_cfg)

    @property
    def nas(self):
        return self.usr_cfg.nas

class DefaultConf(DotDict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = DefaultConf({})
        value = self.get(key, None)
        return value

    __getattr__ = __getitem__

conf = DefaultConf({})
QuantConf = Quantization_Conf
PruningConf = Pruning_Conf
GraphOptConf = Graph_Optimization_Conf
BenchmarkConf = Benchmark_Conf
DistillationConf = Distillation_Conf
MixedPrecisionConf = MixedPrecision_Conf
