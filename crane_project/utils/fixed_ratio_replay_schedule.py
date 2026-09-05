"""Pure scheduling helpers for deterministic fixed-ratio source replay."""

import hashlib


REPLAY_SCHEDULE_PROTOCOL = 'fixed_ratio_pair_replay_schedule_v2'


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
