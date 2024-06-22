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
# ==============================================================================
"""Neural Compressor Quantization API."""

import os
import pickle
import random

import numpy as np
from deprecated import deprecated

from ..conf.config import QuantConf
from ..conf.dotdict import DotDict, deep_get, deep_set
from ..conf.pythonic_config import Config
from ..model import BaseModel
from ..model.model import get_model_fwk_name
from ..model.tensorflow_model import TensorflowQATModel
from ..utils import logger
from ..utils.create_obj_from_config import create_dataloader
from ..utils.utility import time_limit
from .component import Component
from .strategy import EXP_STRATEGIES


@deprecated(version="2.0")
class Quantization(Component):
    """This class provides easy use API for quantization.

       It automatically searches for optimal quantization recipes for low precision model inference,
       achieving best tuning objectives like inference performance within accuracy loss constraints.
       Tuner abstracts out the differences of quantization APIs across various DL frameworks
       and brings a unified API for automatic quantization that works on frameworks including
       tensorflow, pytorch and mxnet.
       Since DL use cases vary in the accuracy metrics (Top-1, MAP, ROC etc.), loss criteria
       (<1% or <0.1% etc.) and tuning objectives (performance, memory footprint etc.).
       Tuner class provides a flexible configuration interface via YAML for users to specify
       these parameters.

    Args:
        conf_fname_or_obj (string or obj): The path to the YAML configuration file or
            QuantConf class containing accuracy goal, tuning objective and preferred
            calibration & quantization tuning space etc.
    """

    def __init__(self, conf_fname_or_obj=None):
        """Quantization constructor."""
        super(Quantization, self).__init__()
        if isinstance(conf_fname_or_obj, QuantConf):
            self.conf = conf_fname_or_obj
        elif isinstance(conf_fname_or_obj, Config):
            self.conf = QuantConf()
            self.conf.map_pyconfig_to_cfg(conf_fname_or_obj)
        else:
            self.conf = QuantConf(conf_fname_or_obj)
        self._init_with_conf()

        seed = self.cfg.tuning.random_seed
        random.seed(seed)
        np.random.seed(seed)
        self._calib_dataloader = None
        self._calib_func = None

    def _create_eval_dataloader(self, cfg):
        """Create default evaluation dataloader if eval_func is not set."""
        # when eval_func is set, will be directly used and eval_dataloader can be None
        if self._eval_func is None:
            if self._eval_dataloader is None:
                eval_dataloader_cfg = deep_get(cfg, "evaluation.accuracy.dataloader")
                if eval_dataloader_cfg is None:
                    logger.info(
                        "Because both eval_dataloader_cfg and user-defined eval_func are None,"
                        " automatically setting 'tuning.exit_policy.performance_only = True'."
                    )
                    deep_set(cfg, "tuning.exit_policy.performance_only", True)
                    logger.info(
                        "The cfg.tuning.exit_policy.performance_only is: {}".format(
                            cfg.tuning.exit_policy.performance_only
                        )
                    )
                else:
                    if deep_get(cfg, "evaluation.accuracy.iteration") == -1 and "dummy_v2" in deep_get(
                        cfg, "evaluation.accuracy.dataloader.dataset", {}
                    ):
                        deep_set(cfg, "evaluation.accuracy.iteration", 10)

                    self._eval_dataloader = create_dataloader(self.framework, eval_dataloader_cfg)
        if os.environ.get("PERFORMANCE_ONLY") in ["0", "1"]:
            performance_only = bool(int(os.environ.get("PERFORMANCE_ONLY")))
            deep_set(cfg, "tuning.exit_policy.performance_only", performance_only)
            logger.info(
                "Get environ 'PERFORMANCE_ONLY={}',"
                " force setting 'tuning.exit_policy.performance_only = True'.".format(performance_only)
            )

    def _create_calib_dataloader(self, cfg):
        """Create default calibration dataloader if train_func is not set."""
        approach_cfg = deep_get(cfg, "quantization.approach")

        if self._calib_dataloader is None and self._calib_func is None:
            if approach_cfg in ["post_training_static_quant", "post_training_auto_quant"]:
                calib_dataloader_cfg = deep_get(cfg, "quantization.calibration.dataloader")

                if approach_cfg == "post_training_auto_quant" and calib_dataloader_cfg is None:
                    logger.error(
                        "dataloader is required for 'post_training_auto_quant'. "
                        "use 'post_training_dynamic_quant' instead if no dataloader provided."
                    )
                assert calib_dataloader_cfg is not None, (
                    "dataloader field of calibration field of quantization section "
                    "in yaml file should be configured as calib_dataloader property is NOT set!"
                )

                if deep_get(calib_dataloader_cfg, "shuffle"):
                    logger.warning("Reset `shuffle` field to False when post_training_static_quant" " is selected.")
                    deep_set(calib_dataloader_cfg, "shuffle", False)
            elif approach_cfg == "quant_aware_training":
                calib_dataloader_cfg = deep_get(cfg, "quantization.train.dataloader")
                assert calib_dataloader_cfg is not None, (
                    "dataloader field of train field of quantization section "
                    "in yaml file should be configured as calib_dataloader property is NOT set!"
                )
            else:
                calib_dataloader_cfg = None

            if calib_dataloader_cfg:
                self._calib_dataloader = create_dataloader(self.framework, calib_dataloader_cfg)

    def pre_process(self):
        """Prepare dataloaders, qfuncs for Component."""
        cfg = self.conf.usr_cfg
        assert isinstance(self._model, BaseModel), "need set your Model for quantization...."

        self._create_eval_dataloader(cfg)
        self._create_calib_dataloader(cfg)
        strategy = cfg.tuning.strategy.name.lower()
        if cfg.quantization.quant_level == 0:
            strategy = "conservative"
            logger.info("On the premise that the accuracy meets the conditions, improve the performance.")

        if strategy == "mse_v2":
            if not (self.framework.startswith("tensorflow") or self.framework == "pytorch_fx"):
                strategy = "basic"
                logger.warning(f"MSE_v2 does not support {self.framework} now, use basic instead.")
                logger.warning("Only tensorflow, pytorch_fx is supported by MSE_v2 currently.")
        assert strategy in EXP_STRATEGIES, "Tuning strategy {} is NOT supported".format(strategy)

        _resume = None
        # check if interrupted tuning procedure exists. if yes, it will resume the
        # whole auto tune process.
        self.resume_file = (
            os.path.abspath(os.path.expanduser(cfg.tuning.workspace.resume))
            if cfg.tuning.workspace and cfg.tuning.workspace.resume
            else None
        )
        if self.resume_file:
            assert os.path.exists(self.resume_file), "The specified resume file {} doesn't exist!".format(
                self.resume_file
            )
            with open(self.resume_file, "rb") as f:
                _resume = pickle.load(f).__dict__

        self.strategy = EXP_STRATEGIES[strategy](
            self._model,
            self.conf,
            self._calib_dataloader,
            self._calib_func,
            self._eval_dataloader,
            self._eval_func,
            _resume,
            self.hooks,
        )

        if getattr(self._calib_dataloader, "distributed", False):
            self.register_hook("on_train_begin", self.strategy.adaptor._pre_hook_for_hvd)

    def execute(self):
        """Quantization execute routine based on strategy design."""
        try:
            with time_limit(self.conf.usr_cfg.tuning.exit_policy.timeout):
                logger.debug("Dump user yaml configuration:")
                logger.debug(self.conf.usr_cfg)
                self.strategy.traverse()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error("Unexpected exception {} happened during tuning.".format(repr(e)))
            import traceback

            traceback.print_exc()
        finally:
            if self.strategy.best_qmodel:
                logger.info(
                    "Specified timeout or max trials is reached! "
                    "Found a quantized model which meet accuracy goal. Exit."
                )
                self.strategy.deploy_config()
            else:
                logger.error(
                    "Specified timeout or max trials is reached! "
                    "Not found any quantized model which meet accuracy goal. Exit."
                )

            return self.strategy.best_qmodel

    def __call__(self):
        """Automatic quantization tuning main entry point.

           This interface works on all the DL frameworks that neural_compressor supports
           and provides three usages:
           a) Fully yaml configuration: User specifies all the info through yaml,
              including dataloaders used in calibration and evaluation phases
              and quantization tuning settings.

              For this usage, only model parameter is mandatory.

           b) Partial yaml configuration: User specifies dataloaders used in calibration
              and evaluation phase by code.
              The tool provides built-in dataloaders and evaluators, user just need provide
              a dataset implemented __iter__ or __getitem__ methods and invoke dataloader()
              with dataset as input parameter to create neural_compressor dataloader before calling this
              function.

              After that, User specifies fp32 "model", calibration dataset "calib_dataloader"
              and evaluation dataset "eval_dataloader".
              The calibrated and quantized model is evaluated with "eval_dataloader"
              with evaluation metrics specified in the configuration file. The evaluation tells
              the tuner whether the quantized model meets the accuracy criteria. If not,
              the tuner starts a new calibration and tuning flow.

              For this usage, model, calib_dataloader and eval_dataloader parameters are mandatory.

           c) Partial yaml configuration: User specifies dataloaders used in calibration phase
              by code.
              This usage is quite similar with b), just user specifies a custom "eval_func"
              which encapsulates the evaluation dataset by itself.
              The calibrated and quantized model is evaluated with "eval_func".
              The "eval_func" tells the tuner whether the quantized model meets
              the accuracy criteria. If not, the Tuner starts a new calibration and tuning flow.

              For this usage, model, calib_dataloader and eval_func parameters are mandatory.

        Returns:
            quantized model: best qanitized model found, otherwise return None
        """
        return super(Quantization, self).__call__()

    fit = __call__

    def dataset(self, dataset_type, *args, **kwargs):
        """Get dataset according to dataset_type."""
        from ..data import Datasets

        return Datasets(self.framework)[dataset_type](*args, **kwargs)

    @property
    def calib_dataloader(self):
        """Get `calib_dataloader` attribute."""
        return self._calib_dataloader

    @calib_dataloader.setter
    def calib_dataloader(self, dataloader):
        """Set Data loader for calibration, mandatory for post-training quantization.

        It is iterable and the batched data should consists of a tuple like
        (input, label) if the calibration dataset containing label, or yield (input, _)
        for label-free calibration dataset, the input in the batched data will be used for
        model inference, so it should satisfy the input format of specific model.
        In calibration process, label of data loader will not be used and
        neither the postprocess and metric. User only need to set
        calib_dataloader when calib_dataloader can not be configured from yaml file.

        Args:
            dataloader(generator): user are supported to set a user defined dataloader
                                    which meet the requirements that can yield tuple of
                                    (input, label)/(input, _) batched data. Another good
                                    practice is to use neural_compressor.experimental.common.DataLoader
                                    to initialize a neural_compressor dataloader object. Notice
                                    neural_compressor.experimental.common.DataLoader is just a wrapper of the
                                    information needed to build a dataloader, it can't yield
                                    batched data and only in this setter method
                                    a 'real' calib_dataloader will be created,
                                    the reason is we have to know the framework info
                                    and only after the Quantization object created then
                                    framework information can be known.
                                    Future we will support creating iterable dataloader
                                    from neural_compressor.experimental.common.DataLoader
        """
        from .common import _generate_common_dataloader

        self._calib_dataloader = _generate_common_dataloader(dataloader, self.framework)

    @property
    def metric(self):
        """Get `metric` attribute."""
        assert False, "Should not try to get the value of `metric` attribute."
        return None

    @metric.setter
    def metric(self, user_metric):
        """Set metric class and neural_compressor will initialize this class when evaluation.

        neural_compressor have many built-in metrics, but user can set specific metric through
        this api. The metric class should take the outputs of the model or
        postprocess(if have) as inputs, neural_compressor built-in metric always take
        (predictions, labels) as inputs for update,
        and user_metric.metric_cls should be sub_class of neural_compressor.metric.BaseMetric
        or user defined metric object
        Args:
            user_metric(neural_compressor.experimental.common.Metric):
                user_metric should be object initialized from
                neural_compressor.experimental.common.Metric, in this method the
                user_metric.metric_cls will be registered to
                specific frameworks and initialized.
        """
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.metric"):
            logger.warning(
                "Override the value of `metric` field defined in yaml file"
                " as user defines the value of `metric` attribute by code."
            )

        from ..metric import METRICS
        from .common import Metric as NCMetric

        if isinstance(user_metric, NCMetric):
            name = user_metric.name
            metric_cls = user_metric.metric_cls
            metric_cfg = {name: {**user_metric.kwargs}}
        else:
            for i in ["reset", "update", "result"]:
                assert hasattr(user_metric, i), "Please realise {} function" "in user defined metric".format(i)
            metric_cls = type(user_metric).__name__
            name = "user_" + metric_cls
            metric_cfg = {name: id(user_metric)}
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.metric", metric_cfg)
        self.conf.usr_cfg = DotDict(self.conf.usr_cfg)
        metrics = METRICS(self.framework)
        metrics.register(name, metric_cls)
        self._metric = user_metric

    @property
    def objective(self):
        """Get `objective` attribute."""
        assert False, "Should not try to get the value of `objective` attribute."
        return None

    @objective.setter
    def objective(self, user_objective):
        """Set objective, neural_compressor supports built-in objectives and user defined objective.

        The built-in objectives include Accuracy, Performance, Footprint and ModelSize.
        """
        if deep_get(self.conf.usr_cfg, "tuning.multi_objectives.objective") or deep_get(
            self.conf.usr_cfg, "tuning.objective"
        ):
            logger.warning(
                "Override the value of `objective` field defined in yaml file"
                " as user defines the value of `objective` attribute by code."
            )

        user_obj_cfg = (
            "tuning.objective"
            if deep_get(self.conf.usr_cfg, "tuning.objective")
            else "tuning.multi_objectives.objective"
        )
        from ..objective import objective_custom_registry

        objective_cls = type(user_objective)
        name = user_objective.__class__.__name__
        objective_cfg = name if deep_get(self.conf.usr_cfg, "tuning.objective") else [name]
        deep_set(self.conf.usr_cfg, user_obj_cfg, objective_cfg)
        self.conf.usr_cfg = DotDict(self.conf.usr_cfg)
        objective_custom_registry(name, objective_cls)

    @property
    def postprocess(self, user_postprocess):
        """Get `postprocess` attribute."""
        assert False, "Should not try to get the value of `postprocess` attribute."
        return None

    @postprocess.setter
    def postprocess(self, user_postprocess):
        """Set postprocess class and neural_compressor will initialize this class when evaluation.

        The postprocess class should take the outputs of the model as inputs, and
        output (predictions, labels) as inputs for metric update.
        user_postprocess.postprocess_cls should be sub_class of neural_compressor.data.BaseTransform.

        Args:
            user_postprocess: neural_compressor.experimental.common.Postprocess
                user_postprocess should be object initialized from
                neural_compressor.experimental.common.Postprocess,
                in this method the user_postprocess.postprocess_cls will be
                registered to specific frameworks and initialized.
        """
        from .common import Postprocess as NCPostprocess

        assert isinstance(
            user_postprocess, NCPostprocess
        ), "please initialize a neural_compressor.experimental.common.Postprocess and set...."
        postprocess_cfg = {user_postprocess.name: {**user_postprocess.kwargs}}
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.postprocess"):
            logger.warning(
                "Override the value of `postprocess` field defined in yaml file"
                " as user defines the value of `postprocess` attribute by code."
            )
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.postprocess.transform", postprocess_cfg)
        from neural_compressor.data import TRANSFORMS

        postprocesses = TRANSFORMS(self.framework, "postprocess")
        postprocesses.register(user_postprocess.name, user_postprocess.postprocess_cls)

    # BELOW API TO BE DEPRECATED!
    @property
    def q_func(self):
        """Get `q_func` attribute."""
        assert False, "Should not try to get the value of `q_func` attribute."
        return None

    @q_func.setter
    def q_func(self, user_q_func):
        """Calibrate quantization parameters for Post-training static quantization.

           It is optional and only takes effect when user choose
           "post_training_static_quant" approach in yaml.

        Args:
            user_q_func: This function takes "model" as input parameter
                         and executes entire inference process or training process with self
                         contained training hyper-parameters..
        """
        self._calib_func = user_q_func

    calib_func = q_func

    @property
    def model(self):
        """Override model getter method to handle quantization aware training case."""
        return self._model

    @model.setter
    def model(self, user_model):
        """Override model setter method to handle quantization aware training case.

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
        approach_cfg = deep_get(self.cfg, "quantization.approach")
        if not self.framework:
            self.framework = get_model_fwk_name(user_model)
        if self.framework == "tensorflow" and approach_cfg == "quant_aware_training":
            if type(user_model) == str:
                self._model = TensorflowQATModel(user_model)
            else:
                self._model = TensorflowQATModel(user_model._model)
        else:
            Component.model.__set__(self, user_model)

    def __repr__(self):
        """Return the class string."""
        return "Quantization"
