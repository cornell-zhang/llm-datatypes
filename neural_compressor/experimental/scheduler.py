"""Scheduler class."""

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

import os

from deprecated import deprecated

from ..conf.dotdict import DotDict, deep_set
from ..model import BaseModel
from ..model.model import get_model_fwk_name
from ..utils import logger
from .benchmark import Benchmark
from .common import Model as NCModel
from .component import Component
from .distillation import Distillation
from .graph_optimization import Graph_Optimization
from .model_conversion import ModelConversion
from .pruning import Pruning
from .quantization import Quantization

SUPPORTED_COMPONENTS = [Quantization, Pruning, Graph_Optimization, ModelConversion, Benchmark, Component]


@deprecated(version="2.0")
class Scheduler(object):
    """Scheduler for neural_compressor component pipeline execution.

    Neural Compressor supports several separate components: Quantization, Pruning, Benchmarking.
    This scheduler will sequentially execute specified components by the order of
    appending. This interface provides an unique entry to pipeline execute all supported
    components.

    There are two typical usages:
    1) if all information are set in user configuration yaml files by using neural_compressor built-in
       dataloaders/datasets/metrics, the code usage is like below:

       prune = Pruning('/path/to/pruning.yaml')
       quantizer = Quantization('/path/to/quantization.yaml')
       scheduler = Scheduler()
       scheduler.model('/path/to/model')
       scheduler.append(prune)
       scheduler.append(quantizer)
       opt_model = scheduler()
       opt_model.save()


    2) if neural_compressor built-in dataloaders/datasets/metrics could not fully meet user requirements,
       customized dataloaders/datasets/metrics are needed, the code usage is like below:

       prune = Pruning('/path/to/pruning.yaml')
       prune.train_func = ...                   # optional if it is configured in user yaml.
       prune.eval_dataloader = ...                # optional if it is configured in user yaml.
       prune.eval_func = ...                      # optional if it is configured in user yaml.

       quantizer = Quantization('/path/to/quantization.yaml')
       quantizer.metric = ...                     # optional if it is configured in user yaml.
       quantizer.calib_dataloader = ...           # optional if it is configured in user yaml.
       quantizer.eval_dataloader = ...            # optional if it is configured in user yaml.

       scheduler = Scheduler()
       scheduler.model('/path/to/model')
       scheduler.append(prune)
       scheduler.append(quantizer)
       opt_model = scheduler()
       opt_model.save()
    """

    def __init__(self):
        """Initialize the attributes."""
        self._model = None
        self._train_func = None
        self._eval_func = None
        self.components = []

    def append(self, *args):
        """Add neural_compressor component into pipeline for sequential execution.

        Args:
            conf_fname_or_obj (string or obj): The path to user configuration yaml file or
                                                conf class.
            calib_dataloader (generator): Optional. Data loader for calibration of Post-
                                            Training Static Quantization,
                                            or None for Post-Training Dynamic Quantization,
                                            or Data loader for training phase of Quantization-
                                            aware Training,
                                            or Data loader for training phase of Pruning,
                                            if its corresponding field is not configured in yaml.
            eval_dataloader (generator):  Optional. Data loader for evaluation phase of all
                                            neural_compressor components, if its corresponding field is not
                                            configured in yaml and eval_func is not specified.
            postprocess (Postprocess):    Optional. Object initialized from common.Postprocess.
            metric (Metric):              Optional. Object initialized from common.Metric.
            q_func (func):                Optional. Training function for Quantization-Aware
                                            Training and Pruning cases if user doesn't provide
                                            training info in user configuration yaml file.
            eval_func (func):             Optional. Evaluation function for Quantization-Aware
                                            Training and Pruning, being None for other cases.
            kwargs (named arguments):     Reserved for interface extension.
        """
        for item in args:
            assert any([isinstance(item, supported_component) for supported_component in SUPPORTED_COMPONENTS])
            self.components.append(item)

    def __call__(self):
        """Do sequential pipeline execution.

        NOTE: it's user responsibility to ensure the output of each component could be fed as
        the input of next component.

        Returns:
            optimized model: best optimized model generated, otherwise return None
        """
        assert self.model, "Scheduler class's model property should be set " "before invoking this __call__() function"
        model = self.model
        assert len(self.components) > 0
        logger.info("Start sequential pipeline execution.")
        for i, component in enumerate(self.components):
            # print appropriate ordinal number representation (1st, 2nd, 3rd) for each step
            ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4])
            logger.info("The {} step being executing is {}.".format(ordinal(i), repr(component).upper()))

            component.model = model
            if self._train_func is not None:
                component.train_func = self._train_func
            if self._eval_func is not None:
                component.eval_func = self._eval_func
            model = component()

        return model

    fit = __call__

    def combine(self, *args):
        """Combine neural_compressor components into a new component.

        Args:
            args (Component): Components to be combined together. Input Component should be
            supported in Neural Compressor and pass the sanity check during combine. The illegal combination
            (e.g. Components uses different frameworks) returns an error.

        Returns:
            Combined Component: The return component is created as base Component class.
            It syncs input components configuration and creates dataloaders/functions
            accordingly.
        """
        assert len(args) > 1, "Combine requires at least 2 components. Please check your inputs."
        # check if input components are good for combine
        self._combination_sanity_check(*args)
        # create component for the combination
        combination = []
        for arg in args:
            combination += [arg.__class__.__name__] if arg.combination is None else arg.combination
        new_component = Component(combination=combination)
        self._combine_components(*args, dist_component=new_component)

        return new_component

    def _combination_sanity_check(self, *args):
        """Check sanity of the combination."""
        TEMP_SUPPORTED_COMPONENTS = ["Quantization", "Pruning", "Distillation"]
        checked_components = []
        for component in args:
            component_class = component.__class__.__name__
            if component_class in TEMP_SUPPORTED_COMPONENTS and component_class not in checked_components:
                checked_components.append(component_class)
            else:
                logger.error("The combination of {} is not supported.".format(checked_components + [component_class]))

    def _combine_components(self, *args, dist_component=None):
        """Actual implementation of combine().

        It loops input component and sync-up status to distance component.

        Args:
            args(tuple): input components.
            dist_component(Component): the distance component for the combination

        Returns: None
        """
        assert len(args) > 0, "Empty input detected in combine."
        framework = args[0].framework
        device = args[0].cfg.device
        train_cfg = DotDict()
        eval_cfg = DotDict()
        tuning_cfg = DotDict()
        model_cfg = DotDict()
        quantization_cfg = DotDict()
        has_distillation = False
        for combine_component in args:
            if isinstance(combine_component, Distillation):
                has_distillation = True

        for combine_component in args:
            # check if config is valid
            assert combine_component.framework == framework, (
                "Combined components should have "
                "same framework. Detect different frameworks: {} and {} are used.".format(
                    framework, combine_component.framework
                )
            )
            assert (
                combine_component.cfg.device == device
            ), "Combined components should have " "same device. Detect different device: {} and {} are used.".format(
                device, combine_component.cfg.device
            )

            # sync configs
            component_name = combine_component.__class__.__name__.lower()
            component_cfg = getattr(combine_component.cfg, component_name, None)
            assert combine_component is not None, "Please ensure field {} is configured " "in input yaml".format(
                component_name
            )

            # in case of key train/evaluation not exist, return an empty DotDict
            component_train_cfg = component_cfg.get("train", DotDict())

            # TODO: Assumption here: train phase is defined inside component yaml field.
            # But eval is defined at root yaml field.
            component_eval_cfg = combine_component.cfg.get("evaluation", DotDict())
            component_tuning_cfg = combine_component.cfg.get("tuning", DotDict())
            component_model_cfg = combine_component.cfg.get("model", DotDict())
            component_quantization_cfg = (
                combine_component.cfg.get("quantization", DotDict()) if component_name == "quantization" else DotDict()
            )

            combine_component._model = self._model
            if component_eval_cfg and component_train_cfg:
                # create attributes if eval and train are defined in component config
                combine_component.pre_process()
            elif isinstance(combine_component, Pruning):
                # if no eval/train configuration. Make minimal initialization
                combine_component.generate_hooks()
                combine_component.generate_pruners()
            elif isinstance(combine_component, Distillation):
                # if no eval/train configuration. Make minimal initialization
                combine_component.create_criterion()
                combine_component.create_optimizer()
                combine_component.generate_hooks()

            # If distillation is combined, set training configs
            if has_distillation:
                if isinstance(combine_component, Distillation):
                    component_train_cfg.criterion = combine_component.train_cfg.criterion
                    component_train_cfg.optimizer = combine_component.train_cfg.optimizer
                    dist_component.criterion = combine_component.train_cfg.criterion
                    dist_component.on_after_compute_loss = combine_component.on_after_compute_loss
                    dist_component.teacher_model = combine_component.teacher_model
                    self._sync_config(train_cfg, component_train_cfg)
                # Only sync train_cfg of distillation because criterion should include only one element
                else:
                    component_train_cfg = DotDict()

            self._sync_config(train_cfg, component_train_cfg)
            self._sync_config(eval_cfg, component_eval_cfg)
            self._sync_config(tuning_cfg, component_tuning_cfg)
            self._sync_config(model_cfg, component_model_cfg)
            self._sync_config(quantization_cfg, component_quantization_cfg)

            # sync hooks
            if combine_component.hooks is None:
                continue
            for k, v in combine_component.hooks.items():
                dist_component.register_hook(k, v)

        # sync to dist component
        dist_component_cfg = DotDict()
        if dist_component is not None:
            deep_set(dist_component_cfg, "train", train_cfg)
            deep_set(dist_component_cfg, "evaluation", eval_cfg)
            deep_set(dist_component_cfg, "tuning", tuning_cfg)
            deep_set(dist_component_cfg, "device", device)
            deep_set(dist_component_cfg, "model", model_cfg)
            deep_set(dist_component_cfg, "quantization", quantization_cfg)
            dist_component._model = self._model
            dist_component.framework = framework
            dist_component.cfg = dist_component_cfg

    def _sync_config(self, dist_config, src_config):
        """Sync the configuration between src and dist.

        It updates the missing keys in dist config from src config.
        And check the value for same keys between src and dist.

        Args:
            dist_config(DotDict): dist configuration to sync
            src_config(DotDict): src configuration to sync

        Returns: None
        """
        # sync the config if dist and src configs are not empty
        if dist_config and src_config:
            # update missing keys in dist_config
            for key in src_config.keys() - dist_config.keys():
                dist_config[key] = src_config[key]
            # check the same keys in dist and src whether is valid
            for key in src_config.keys() & dist_config.keys():
                if isinstance(dist_config[key], dict) and isinstance(src_config[key], dict):
                    self._sync_config(dist_config[key], src_config[key])
                elif dist_config[key] != src_config[key]:
                    logger.warning(
                        "Find different value {} and {} on key {}.".format(dist_config[key], src_config[key], key)
                        + " Use first key-value ({}: {}) pair as default".format(key, dist_config[key])
                    )
        # update src config to dist if dist is empty.
        elif not dist_config and src_config:
            dist_config.update(src_config)

    @property
    def model(self):
        """Getter of model.

        Returns:
            The model used in the Scheduler process.
        """
        return self._model

    @model.setter
    def model(self, user_model):
        """Set the user model and dispatch to framework specific internal model object.

        Args:
           user_model: user are supported to set model from original framework model format
                       (eg, tensorflow frozen_pb or path to a saved model),
                       but not recommended. Best practice is to set from a initialized
                       neural_compressor.experimental.common.Model.
                       If tensorflow model is used, model's inputs/outputs will be
                       auto inferenced, but sometimes auto inferenced
                       inputs/outputs will not meet your requests,
                       set them manually in config yaml file.
                       Another corner case is slim model of tensorflow,
                       be careful of the name of model configured in yaml file,
                       make sure the name is in supported slim model list.
        """
        if not isinstance(user_model, BaseModel):
            logger.warning("Force convert framework model to neural_compressor model.")
            self._model = NCModel(user_model)
        else:
            self._model = user_model

    @property
    def train_func(self):
        """Do not support get train_func."""
        assert False, "Should not try to get the value of `train_func` attribute."
        return None

    @train_func.setter
    def train_func(self, user_train_func):
        """Training function.

        Args:
            user_train_func: This function takes "model" as input parameter
                         and executes entire training process with self
                         contained training hyper-parameters. If training_func set,
                         an evaluation process must be triggered and user should
                         set eval_dataloader with metric configured or directly eval_func
                         to make evaluation of the model executed. training_func will return
                         a trained model.
        """
        self._train_func = user_train_func

    @property
    def eval_func(self):
        """Do not support get eval_func."""
        assert False, "Should not try to get the value of `eval_func` attribute."
        return None

    @eval_func.setter
    def eval_func(self, user_eval_func):
        """Evaluate function.

        Args:
            user_eval_func: This function takes "model" as input parameter
                         and executes entire evaluation process with self
                         contained metrics. If eval_func set,
                         an evaluation process must be triggered
                         to make evaluation of the model executed.
        """
        self._eval_func = user_eval_func
