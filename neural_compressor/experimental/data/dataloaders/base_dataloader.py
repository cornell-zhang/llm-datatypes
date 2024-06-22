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
"""BaseDataloder of all dataloaders."""

from abc import abstractmethod

from deprecated import deprecated


@deprecated(version="2.0")
class BaseDataLoader:
    """Base class for all DataLoaders.

    _generate_dataloader is needed to create a dataloader object
    from the general params like batch_size and sampler. The dynamic batching is just to
    generate a new dataloader by setting batch_size and last_batch.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        last_batch="rollover",
        collate_fn=None,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        distributed=False,
    ):
        """Initialize BaseDataLoader.

        Args:
            dataset (object): dataset from which to load the data
            batch_size (int, optional): number of samples per batch. Defaults to 1.
            last_batch (str, optional): whether to drop the last batch if it is incomplete.
                                        Support ['rollover', 'discard'], rollover means False, discard means True.
                                        Defaults to 'rollover'.
            collate_fn (callable, optional): merge data with outer dimension batch size. Defaults to None.
            sampler (Sampler, optional): Sampler object to sample data. Defaults to None.
            batch_sampler (BatchSampler, optional): BatchSampler object to generate batch of indices. Defaults to None.
            num_workers (int, optional): number of subprocesses to use for data loading. Defaults to 0.
            pin_memory (bool, optional): whether to copy data into pinned memory before returning. Defaults to False.
            shuffle (bool, optional): whether to shuffle data. Defaults to False.
            distributed (bool, optional): whether the dataloader is distributed. Defaults to False.
        """
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        self.last_batch = last_batch
        self.drop_last = False if last_batch == "rollover" else True

        self.dataloader = self._generate_dataloader(
            self.dataset,
            batch_size=batch_size,
            last_batch=last_batch,
            collate_fn=collate_fn,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            distributed=distributed,
        )

    def batch(self, batch_size, last_batch=None):
        """Set batch size for dataloader.

        Args:
            batch_size (int): number of samples per batch.
            last_batch (str, optional): whether to drop the last batch if it is incomplete.
                                        Support ['rollover', 'discard'], rollover means False, discard means True.
                                        Defaults to None.
        """
        self._batch_size = batch_size
        if last_batch is not None:
            self.last_batch = last_batch
        self.dataloader = self._generate_dataloader(
            self.dataset,
            batch_size,
            self.last_batch,
            self.collate_fn,
            self.sampler,
            self.batch_sampler,
            self.num_workers,
            self.pin_memory,
            self.shuffle,
            self.distributed,
        )

    @property
    def batch_size(self):
        """Get dataloader's batch_size.

        Returns:
            int: batch_size
        """
        return self._batch_size

    def __iter__(self):
        """Yield data in iterative order.

        Returns:
            iterator: iterator for dataloder
        """
        return iter(self.dataloader)

    @abstractmethod
    def _generate_dataloader(
        self,
        dataset,
        batch_size,
        last_batch,
        collate_fn,
        sampler,
        batch_sampler,
        num_workers,
        pin_memory,
        shuffle,
        distributed,
    ):
        raise NotImplementedError
