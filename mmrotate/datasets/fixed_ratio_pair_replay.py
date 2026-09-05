"""Deterministic pair-wise replay for source-only temporal training."""

import copy

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS

from crane_project.utils.fixed_ratio_replay_schedule import (
    replay_schedule_contract, route_replay_batch)


@DATASETS.register_module()
class FixedRatioPairReplayDataset:
    """Replay original and auxiliary source pairs at a fixed step ratio.

    A batch is always drawn wholly from one child dataset. This preserves
    adjacency checks in the causal trainer and prevents a short auxiliary
    video from dominating merely because it was appended to the source set.
    The wrapper length is fixed in optimizer-step units, so adding source
    frames does not silently change the training budget.
    """

    def __init__(self, original_dataset, auxiliary_dataset,
                 samples_per_batch=2, original_batches_per_auxiliary_batch=14,
                 optimizer_steps_per_epoch=1391):
        # Import lazily to avoid a builder import cycle at module load time.
        from .builder import build_dataset

        self.original_dataset = build_dataset(original_dataset)
        self.auxiliary_dataset = build_dataset(auxiliary_dataset)
        self.samples_per_batch = int(samples_per_batch)
        self.original_batches_per_auxiliary_batch = int(
            original_batches_per_auxiliary_batch)
        self.optimizer_steps_per_epoch = int(optimizer_steps_per_epoch)
        if self.samples_per_batch != 2:
            raise ValueError(
                'FixedRatioPairReplayDataset requires pair batches of two')
        if self.original_batches_per_auxiliary_batch <= 0:
            raise ValueError('Replay ratio must contain original batches')
        if self.optimizer_steps_per_epoch <= 0:
            raise ValueError('optimizer_steps_per_epoch must be positive')
        if len(self.original_dataset) < 2 or len(self.auxiliary_dataset) < 2:
            raise ValueError('Both replay datasets require at least two rows')
        self._replay_schedule_contract = replay_schedule_contract(
            self.optimizer_steps_per_epoch,
            self.original_batches_per_auxiliary_batch)
        self.CLASSES = getattr(
            self.original_dataset, 'CLASSES',
            getattr(self.auxiliary_dataset, 'CLASSES', None))
        self.PALETTE = getattr(self.original_dataset, 'PALETTE', None)
        self.epoch = 0
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        return self.optimizer_steps_per_epoch * self.samples_per_batch

    def _route(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        batch = index // self.samples_per_batch
        within_batch = index % self.samples_per_batch
        auxiliary, pair_number = route_replay_batch(
            batch, self.original_batches_per_auxiliary_batch)
        if auxiliary:
            dataset = self.auxiliary_dataset
        else:
            dataset = self.original_dataset
        contract = self.replay_contract()
        samples_per_epoch = self.samples_per_batch * (
            contract['scheduled_auxiliary_steps'] if auxiliary else
            contract['scheduled_original_steps'])
        epoch_offset = self.epoch * samples_per_epoch
        child_index = (
            epoch_offset + pair_number * self.samples_per_batch
            + within_batch) % len(dataset)
        return dataset, child_index, auxiliary

    def set_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError('Replay epoch cannot be negative')
        self.epoch = epoch

    def __getitem__(self, index):
        dataset, child_index, auxiliary = self._route(index)
        sample = copy.copy(dataset[child_index])
        # Added after the child pipeline so it cannot become a model feature.
        # It is supervision provenance used only to mask teacher retention.
        sample['source_replay_is_auxiliary'] = DC(
            torch.tensor(int(auxiliary), dtype=torch.uint8),
            stack=True, pad_dims=None)
        return sample

    def get_cat_ids(self, index):
        dataset, child_index, _ = self._route(index)
        if not hasattr(dataset, 'get_cat_ids'):
            return []
        return dataset.get_cat_ids(child_index)

    def evaluate(self, *args, **kwargs):
        raise RuntimeError(
            'FixedRatioPairReplayDataset is training-only and not evaluable')

    def replay_contract(self):
        contract = dict(self._replay_schedule_contract)
        contract.update(dict(
            samples_per_batch=self.samples_per_batch,
            actual_route_enumerated=True))
        return contract

    def coverage_contract(self, training_epochs):
        saved_epoch = self.epoch
        original_seen = set()
        auxiliary_seen = set()
        try:
            for epoch in range(int(training_epochs)):
                self.set_epoch(epoch)
                for index in range(len(self)):
                    dataset, child_index, auxiliary = self._route(index)
                    del dataset
                    (auxiliary_seen if auxiliary else original_seen).add(
                        child_index)
        finally:
            self.set_epoch(saved_epoch)
        schedule = self.replay_contract()
        return dict(
            replay_schedule_protocol=schedule['protocol'],
            schedule_sha256=schedule['schedule_sha256'],
            scheduled_original_steps_per_epoch=(
                schedule['scheduled_original_steps']),
            scheduled_auxiliary_steps_per_epoch=(
                schedule['scheduled_auxiliary_steps']),
            training_epochs=int(training_epochs),
            original_unique_covered=len(original_seen),
            original_unique_total=len(self.original_dataset),
            auxiliary_unique_covered=len(auxiliary_seen),
            auxiliary_unique_total=len(self.auxiliary_dataset),
            full_original_coverage=(
                len(original_seen) == len(self.original_dataset)),
            full_auxiliary_coverage=(
                len(auxiliary_seen) == len(self.auxiliary_dataset)))
