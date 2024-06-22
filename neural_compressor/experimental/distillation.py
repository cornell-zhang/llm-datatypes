"""Distillation class."""

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

from deprecated import deprecated

from neural_compressor.compression.distillation.criterions import Criterions
from neural_compressor.experimental.common import Optimizers

from ..adaptor import FRAMEWORKS
from ..conf.config import DistillationConf
from ..conf.pythonic_config import Config, DotDict
from ..model import BaseModel
from ..utils import logger
from ..utils.create_obj_from_config import create_dataloader, create_eval_func, create_train_func
from .common import Model
from .component import Component


@deprecated(version="2.0")
class Distillation(Component):
    """Distillation class derived from Component class.

    Distillation class abstracted the pipeline of knowledge distillation,
    transfer the knowledge of the teacher model to the student model.

    Args:
        conf_fname_or_obj (string or obj): The path to the YAML configuration file or
            Distillation_Conf containing accuracy goal, distillation objective and related
            dataloaders etc.

    Attributes:
        _epoch_ran: A integer indicating how much epochs ran.
        eval_frequency: The frequency for doing evaluation of the student model
            in terms of epoch.
        best_score: The best metric of the student model in the training.
        best_model: The best student model found in the training.
    """

    def __init__(self, conf_fname_or_obj=None):
        """Initialize the attributes."""
        super(Distillation, self).__init__()
        if isinstance(conf_fname_or_obj, DistillationConf):
            self.conf = conf_fname_or_obj
        elif isinstance(conf_fname_or_obj, Config):
            self.conf = DistillationConf()
            self.conf.map_pyconfig_to_cfg(conf_fname_or_obj)
        else:
            self.conf = DistillationConf(conf_fname_or_obj)
        self._init_with_conf()

        self._teacher_model = None
        self._criterion = None
        self._optimizer = None
        self._epoch_ran = 0
        self.eval_frequency = 1
        self.best_score = 0
        self.best_model = None
        self._train_cfg = None

    def _on_train_begin(self, dataloader=None):
        """Operations called on the beginning of the training.

        Called before training, evaluate the teacher model and the student model.
        """
        assert self._model, "student_model must be set."
        if self._eval_func is not None:
            if self.teacher_model:
                score = self._eval_func(
                    self.teacher_model if getattr(self._eval_func, "builtin", None) else self.teacher_model.model
                )
                logger.info("teacher model score is {}.".format(str(score)))

            score = self._eval_func(self._model if getattr(self._eval_func, "builtin", None) else self._model.model)
            logger.info("initial model score is {}.".format(str(score)))
            if self.eval_frequency > 0:
                self.best_score = score
                if self.framework == "pytorch":
                    self.best_model = copy.deepcopy(self._model)
                else:
                    self.best_model = self._model

    def _on_step_begin(self, batch_id):
        """Operations called on the beginning of batches."""
        if self.criterion is not None and hasattr(self.criterion, "clear_features"):
            self.criterion.clear_features()

    def _on_after_compute_loss(self, input, student_output, student_loss, teacher_output=None):
        """Set or compute output of teacher model.

        Called after student model forward, calculate the output of the teacher model
        with the same input of the student model.

        Args:
            input (tensor or list or dict): The input of the student model.
            student_output (tensor): The output logits of the student model.
            student_loss (tensor or float): The original loss of the student model.
            teacher_output (tensor, optional): The output logits of the teacher model.
        """
        if self.criterion is None:
            self.create_criterion()
        assert self.criterion, "criterion must be set in yaml config file."
        if teacher_output is None:
            assert self.teacher_model, "teacher_model must be set."
            teacher_output = self.criterion.teacher_model_forward(input, teacher_model=self.teacher_model._model)
        return self.criterion.loss_cal_sloss(student_output, teacher_output, student_loss)

    def on_post_forward(self, input, teacher_output=None):  # pragma: no cover
        """Set or compute output of teacher model.

        Deprecated.
        """
        assert False, (
            "This method is deprecated. please use `on_after_compute_loss` instead."
            "on_after_compute_loss(input, student_output, student_loss, teacher_output=None)"
        )

    def _on_epoch_end(self):
        """Operations called on the end of every epochs.

        Called on the end of every epochs, evaluate the student model
        and record the best one regularly.
        """
        self._epoch_ran += 1
        if self._eval_func is not None and self.eval_frequency > 0 and self._epoch_ran % self.eval_frequency == 0:
            score = self._eval_func(self._model if getattr(self._eval_func, "builtin", None) else self._model.model)
            logger.info("model score of epoch {} is {}.".format(self._epoch_ran, str(score)))
            if (
                isinstance(score, list) and all([s > b_s for s, b_s in zip(score, self.best_score)])
            ) or score > self.best_score:
                self.best_score = score
                if self.framework == "pytorch":
                    self.best_model = copy.deepcopy(self._model)
                else:
                    self.best_model = self._model

    def init_train_cfg(self):
        """Initialize the training configuration."""
        if self._train_cfg is None:
            # train section of distillation section in yaml file should be configured.
            self._train_cfg = self.cfg.distillation.train
        assert self._train_cfg, (
            "train field of distillation section in yaml file must "
            "be configured for distillation if train_func is NOT set."
        )

    def create_criterion(self):
        """Create the criterion for training."""
        self.init_train_cfg()
        if self.criterion is None:
            assert "criterion" in self._train_cfg.keys(), (
                "criterion part in train field of distillation section in yaml file "
                "must be configured for distillation if criterion is NOT set."
            )

            if isinstance(self._train_cfg.criterion, DotDict):
                criterion_cfg = self._train_cfg.criterion
            else:
                criterion_cfg = self._train_cfg.criterion.config

            assert (
                len(criterion_cfg) == 1
            ), "There must be exactly one loss in " "criterion part, instead got {} loss.".format(len(criterion_cfg))
            loss = [i for i in criterion_cfg.keys()][0]
            loss_cfg = criterion_cfg[loss]
            criterion_builder = Criterions(self.framework)[loss](loss_cfg)
            criterion_tuple = criterion_builder()
            if self.teacher_model and self.student_model:
                if self.framework == "tensorflow":  # new, for tf
                    teacher_model = self.teacher_model._model
                    student_model = self.student_model._model
                else:  # for pytorch and other frameworks
                    teacher_model = self.teacher_model.model
                    student_model = self.student_model.model
                criterion_tuple[1]["student_model"] = student_model
                criterion_tuple[1]["teacher_model"] = teacher_model
            self.criterion = criterion_tuple[0](**criterion_tuple[1])
        else:
            logger.warning("Use user defined criterion, " "ignoring the criterion setting in yaml file.")

        self._train_cfg.criterion = self.criterion

    def create_optimizer(self):
        """Create the optimizer for training."""
        self.init_train_cfg()
        if self.optimizer is None:
            assert "optimizer" in self._train_cfg.keys(), (
                "optimizer part in train field of distillation section in yaml file "
                "must be configured for distillation if optimizer is NOT set."
            )
            optimizer_cfg = self._train_cfg.optimizer
            assert (
                len(optimizer_cfg) == 1
            ), "There must be exactly one optimizer in " "optimizer part, instead got {} optimizer.".format(
                len(optimizer_cfg)
            )
            optimizer_name = list(optimizer_cfg.keys())[0]
            optimizer_cfg_ = optimizer_cfg[optimizer_name]
            optimizer_builder = Optimizers(self.framework)[optimizer_name](optimizer_cfg_)
            optimizer_tuple = optimizer_builder()
            if self.framework == "tensorflow":
                self.optimizer = optimizer_tuple[0](**optimizer_tuple[1])
            elif self.framework == "pytorch":
                # pylint: disable=no-member
                self.optimizer = optimizer_tuple[0](self.model.model.parameters(), **optimizer_tuple[1])
        else:
            logger.warning("Use user defined optimizer, " "ignoring the optimizer setting in yaml file.")

        self._train_cfg.optimizer = self.optimizer

    def prepare(self):
        """Prepare hooks."""
        self.generate_hooks()
        self.create_criterion()

    def pre_process(self):
        """Preprocessing before the disillation pipeline.

        Initialize necessary parts for distillation pipeline.
        """
        framework_specific_info = {
            "device": self.cfg.device,
            "random_seed": self.cfg.tuning.random_seed,
            "workspace_path": self.cfg.tuning.workspace.path,
            "q_dataloader": None,
            "format": "default",
            "backend": "default",
        }

        if self.framework == "tensorflow":
            framework_specific_info.update({"inputs": self.cfg.model.inputs, "outputs": self.cfg.model.outputs})

        self.adaptor = FRAMEWORKS[self.framework](framework_specific_info)

        self.generate_hooks()
        assert isinstance(self._model, BaseModel), "need set neural_compressor Model for distillation...."

        if (
            self._train_dataloader is None
            and self._train_func is None
            and self.cfg.distillation.train.dataloader is not None
        ):
            train_dataloader_cfg = self.cfg.distillation.train.dataloader

            self._train_dataloader = create_dataloader(self.framework, train_dataloader_cfg)

        if (
            self.cfg.evaluation
            and self.cfg.evaluation.accuracy
            and self.cfg.evaluation.accuracy.dataloader
            and self._eval_dataloader is None
            and self._eval_func is None
        ):
            eval_dataloader_cfg = self.cfg.evaluation.accuracy.dataloader
            assert eval_dataloader_cfg is not None, (
                "dataloader field of evaluation "
                "in yaml file should be configured as eval_dataloader property is NOT set!"
            )

            self._eval_dataloader = create_dataloader(self.framework, eval_dataloader_cfg)

        if self._train_func is None:
            if self.criterion is None:
                self.create_criterion()
            self.create_optimizer()
            if self._train_dataloader is not None:
                self._train_func = create_train_func(
                    self.framework, self.train_dataloader, self.adaptor, self._train_cfg, hooks=self.hooks
                )
        if self.cfg.evaluation and self.eval_dataloader and self._eval_func is None:
            # eval section in yaml file should be configured.
            eval_cfg = self.cfg.evaluation
            assert eval_cfg, (
                "eval field of distillation section in yaml file must "
                "be configured for distillation if eval_func is NOT set."
            )
            self._eval_func = create_eval_func(
                self.framework,
                self.eval_dataloader,
                self.adaptor,
                eval_cfg.accuracy.metric,
                eval_cfg.accuracy.postprocess,
                fp32_baseline=False,
            )

    def execute(self):
        """Do distillation pipeline.

        First train the student model with the teacher model, after training,
        evaluating the best student model if any.

        Returns:
            Best distilled model found.
        """
        self._train_func(self._model if getattr(self._train_func, "builtin", None) else self._model.model)
        if self.criterion is not None and hasattr(self.criterion, "remove_all_hooks"):
            self.criterion.remove_all_hooks()
        logger.info("Model distillation is done.")
        if self._eval_func is not None:
            logger.info("Start to evaluate the distilled model.")
            self._model = self.best_model if self.best_model else self._model
            score = self._eval_func(self._model if getattr(self._eval_func, "builtin", None) else self._model.model)

            logger.info("distilled model score is {}.".format(str(score)))
        return self._model

    def generate_hooks(self):
        """Register hooks for distillation.

        Register necessary hooks for distillation pipeline.
        """
        self.register_hook("on_train_begin", self._on_train_begin)
        self.register_hook("on_step_begin", self._on_step_begin)
        self.register_hook("on_after_compute_loss", self._on_after_compute_loss)
        self.register_hook("on_epoch_end", self._on_epoch_end)

    def __call__(self):
        """Do distillation workflow.

           This interface currently only works on pytorch
           and provides three usages:
           a) Fully yaml configuration: User specifies all the info through yaml,
              including dataloaders used in training and evaluation phases
              and distillation settings.

              For this usage, only student_model and teacher_model parameter is mandatory.

           b) Partial yaml configuration: User specifies dataloaders used in training
              and evaluation phase by code.
              The tool provides built-in dataloaders and evaluators, user just need provide
              a dataset implemented __iter__ or __getitem__ methods and invoke dataloader()
              with dataset as input parameter to create neural_compressor dataloader before calling this
              function.

              After that, User specifies fp32 "model", training dataset "train_dataloader"
              and evaluation dataset "eval_dataloader".

              For this usage, student_model, teacher_model, train_dataloader and eval_dataloader
              parameters are mandatory.

           c) Partial yaml configuration: User specifies dataloaders used in training phase
              by code.
              This usage is quite similar with b), just user specifies a custom "eval_func"
              which encapsulates the evaluation dataset by itself.
              The trained and distilled model is evaluated with "eval_func".
              The "eval_func" tells the tuner whether the distilled model meets
              the accuracy criteria. If not, the Tuner starts a new training and tuning flow.

              For this usage, student_model, teacher_model, train_dataloader and eval_func
              parameters are mandatory.

        Returns:
            distilled model: best distilled model found, otherwise return None
        """
        return super(Distillation, self).__call__()

    fit = __call__

    @property
    def criterion(self):
        """Getter of criterion.

        Returns:
            The criterion used in the distillation process.
        """
        return self._criterion

    @criterion.setter
    def criterion(self, user_criterion):
        """Setter of criterion used in the distillation process.

        Set the user defined criterion. When using built-in train_func, user can
        specify the customized criterion through this setter.

        Args:
            user_criterion (criterion object): User defined criterion.
        """
        self._criterion = user_criterion

    @property
    def optimizer(self):
        """Getter of optimizer.

        Returns:
            The optimizer used in the distillation process.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, user_optimizer):
        """Setter of optimizer used in the distillation process.

        Set the user defined optimizer. When using built-in train_func, user can
        specify the customized optimizer through this setter.

        Args:
            user_optimizer (criterion object): User defined optimizer.
        """
        self._optimizer = user_optimizer

    @property
    def teacher_model(self):
        """Getter of the teacher model.

        Returns:
            The teacher model used in the distillation process.
        """
        return self._teacher_model

    @teacher_model.setter
    def teacher_model(self, user_model):
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
            self._teacher_model = Model(user_model)
        else:
            self._teacher_model = user_model

    @property
    def student_model(self):
        """Getter of the student model.

        Returns:
            The student model used in the distillation process.
        """
        return self._model

    @student_model.setter
    def student_model(self, user_model):
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
            self._model = Model(user_model)
        else:
            self._model = user_model

    @property
    def train_cfg(self):
        """Getter of the train configuration.

        Returns:
            The train configuration used in the distillation process.
        """
        return self._train_cfg

    @property
    def evaluation_distributed(self):
        """Getter to know whether need distributed evaluation dataloader."""
        return self._evaluation_distributed

    @evaluation_distributed.setter
    def evaluation_distributed(self, distributed):
        """Setter to know whether need distributed evaluation dataloader."""
        self._evaluation_distributed = distributed

    @property
    def train_distributed(self):
        """Getter to know whether need distributed training dataloader."""
        return self._train_distributed

    @train_distributed.setter
    def train_distributed(self, distributed):
        """Setter to know whether need distributed training dataloader."""
        self._train_distributed = distributed

    def __repr__(self):
        """Class representation."""
        return "Distillation"
