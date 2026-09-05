"""Pure scheduling helpers for deterministic fixed-ratio source replay."""

import hashlib
import math


REPLAY_SCHEDULE_PROTOCOL = 'fixed_ratio_pair_replay_schedule_v2'
LEGACY_REPLAY_SCHEDULE_PROTOCOL = (
    'fixed_ratio_pair_replay_schedule_legacy_ceil_contract_v1')


def route_replay_batch(batch_index, original_batches_per_auxiliary_batch):
    """Return ``(is_auxiliary, lane_batch_index)`` for one optimizer step."""
    batch_index = int(batch_index)
    original_batches = int(original_batches_per_auxiliary_batch)
    if batch_index < 0:
        raise ValueError('Replay batch index cannot be negative')
    if original_batches <= 0:
        raise ValueError('Replay ratio must contain original batches')
    cycle = original_batches + 1
    cycle_number, cycle_index = divmod(batch_index, cycle)
    if cycle_index == original_batches:
        return True, cycle_number
    return False, cycle_number * original_batches + cycle_index


def enumerate_replay_schedule(optimizer_steps_per_epoch,
                              original_batches_per_auxiliary_batch):
    """Enumerate the exact batch route used by one training epoch."""
    steps = int(optimizer_steps_per_epoch)
    if steps <= 0:
        raise ValueError('optimizer_steps_per_epoch must be positive')
    return [
        route_replay_batch(batch, original_batches_per_auxiliary_batch)
        for batch in range(steps)]


def replay_schedule_contract(optimizer_steps_per_epoch,
                             original_batches_per_auxiliary_batch):
    """Describe the exact enumerated route without rounding partial cycles."""
    routes = enumerate_replay_schedule(
        optimizer_steps_per_epoch, original_batches_per_auxiliary_batch)
    original = sum(not auxiliary for auxiliary, _ in routes)
    auxiliary = sum(auxiliary for auxiliary, _ in routes)
    serialized = '\n'.join(
        '{}:{}'.format('auxiliary' if auxiliary else 'original', lane_batch)
        for auxiliary, lane_batch in routes).encode('ascii')
    return dict(
        protocol=REPLAY_SCHEDULE_PROTOCOL,
        route_semantics='original_batches_then_one_auxiliary',
        optimizer_steps_per_epoch=len(routes),
        original_batches_per_auxiliary_batch=int(
            original_batches_per_auxiliary_batch),
        auxiliary_batches_per_cycle=1,
        scheduled_original_steps=int(original),
        scheduled_auxiliary_steps=int(auxiliary),
        enumerated_total_steps=int(original + auxiliary),
        schedule_sha256=hashlib.sha256(serialized).hexdigest())


def legacy_replay_schedule_contract(optimizer_steps_per_epoch,
                                    original_batches_per_auxiliary_batch):
    """Describe the historical ceil-count contract used by V4 checkpoints.

    The route itself was always enumerated by :func:`route_replay_batch`.
    Historical code used a ceil-based auxiliary count only for reporting and
    the next epoch's child offset.  Keeping that behavior behind an explicit
    protocol preserves old checkpoint reproducibility without carrying it
    into V2 CV runs.
    """
    steps = int(optimizer_steps_per_epoch)
    original_batches = int(original_batches_per_auxiliary_batch)
    if steps <= 0 or original_batches <= 0:
        raise ValueError('legacy replay parameters must be positive')
    cycle = original_batches + 1
    auxiliary = int(math.ceil(float(steps) / float(cycle)))
    original = steps - auxiliary
    actual = replay_schedule_contract(steps, original_batches)
    return dict(
        protocol=LEGACY_REPLAY_SCHEDULE_PROTOCOL,
        route_semantics='historical_route_with_ceil_offset_contract',
        optimizer_steps_per_epoch=steps,
        original_batches_per_auxiliary_batch=original_batches,
        auxiliary_batches_per_cycle=1,
        scheduled_original_steps=original,
        scheduled_auxiliary_steps=auxiliary,
        enumerated_original_steps=actual['scheduled_original_steps'],
        enumerated_auxiliary_steps=actual['scheduled_auxiliary_steps'],
        enumerated_total_steps=steps,
        schedule_sha256=actual['schedule_sha256'])
