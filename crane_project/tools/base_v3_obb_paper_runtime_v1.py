#!/usr/bin/env python3
"""Unified online OBB observation runtime for the focused paper.

The runtime exposes the raw Base V3 observation, a frozen anchor-score
rejection baseline, and the frozen V5.1 component policy for every frame.
Only detector outputs and causal history are consumed. Ground truth is never
accepted by the online adapter and is attached by evaluation code afterwards.
"""

import copy

from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS, _measurement_value)
from crane_project.tools.base_v3_obb_hybrid_runtime_v51 import (
    HybridObservationManagerV51, validate_runtime_calibration)


PROTOCOL = 'base_v3_obb_paper_runtime_v1'
METHODS = ('raw', 'score_rejection_only', 'v51_hybrid')
ONLINE_FIELDS = (
    'frame_key', 'domain', 'sequence', 'frame', 'timestamp_seconds',
    'image_size', 'anchor_source', 'anchor_score', 'base_v3_box', 'k1_box',
    'dino_box')


def _online_row(row):
    """Copy only fields authorized for online inference."""
    required = ('frame_key', 'domain', 'sequence', 'frame', 'anchor_source',
                'anchor_score', 'base_v3_box', 'k1_box', 'dino_box')
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError('Online row missing fields: {}'.format(missing))
    return {name: copy.deepcopy(row.get(name)) for name in ONLINE_FIELDS}


def _simple_component(row, component, accepted, gate):
    available = row.get('base_v3_box') is not None
    return dict(
        value=(_measurement_value(row, component) if accepted else None),
        valid=bool(accepted),
        state='measurement' if accepted else 'unavailable',
        source='base_v3_measurement' if accepted else None,
        age_since_measurement_frames=0 if accepted else None,
        measurement_available=available,
        measurement_accepted=bool(accepted), risk=None, gate=gate,
        reason=('accepted_measurement' if accepted else
                'detector_missing' if not available else
                'rejected_measurement'))


class PaperObservationManagerV1:
    """Emit the three frozen paper comparison methods frame by frame."""

    def __init__(self, runtime_calibration):
        validate_runtime_calibration(runtime_calibration)
        self.calibration = copy.deepcopy(runtime_calibration)
        self._v51 = HybridObservationManagerV51(self.calibration)

    def reset(self):
        self._v51.reset()

    def update(self, detector_row):
        row = _online_row(detector_row)
        available = row.get('base_v3_box') is not None
        score = row.get('anchor_score')
        score_accept = bool(
            available and score is not None and
            float(score) >= float(
                self.calibration['score_gate']['score_threshold']))
        raw = {component: _simple_component(
            row, component, available, 'detector_availability')
            for component in COMPONENTS}
        score_only = {component: _simple_component(
            row, component, score_accept, 'anchor_score')
            for component in COMPONENTS}
        v51 = self._v51.update(row)
        return dict(
            protocol=PROTOCOL,
            frame_key=row['frame_key'], domain=row['domain'],
            sequence=row['sequence'], frame=int(row['frame']),
            timestamp_seconds=row.get('timestamp_seconds'),
            image_size=row.get('image_size'),
            detector=dict(
                raw_obb=copy.deepcopy(row.get('base_v3_box')),
                measurement_available=available,
                anchor_source=row.get('anchor_source'),
                anchor_score=row.get('anchor_score')),
            observations=dict(
                raw=raw, score_rejection_only=score_only,
                v51_hybrid=v51['components']),
            v51_online_features=v51['online_features'],
            online_gt_fields_consumed=[])
