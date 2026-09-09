#!/usr/bin/env python3
"""Online-only runtime adapter for the frozen V5.1 hybrid OBB policy."""

import copy

import numpy as np

from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS, _frame_features, _measurement_value)
from crane_project.tools.base_v3_obb_hybrid_policy_freeze_v51 import (
    RUNTIME_PROTOCOL)


def validate_runtime_calibration(calibration):
    if calibration.get('protocol') != RUNTIME_PROTOCOL:
        raise ValueError('Unexpected V5.1 runtime calibration protocol')
    online = calibration.get('online_contract', {})
    if online.get('gt_fields_consumed') != []:
        raise ValueError('V5.1 runtime must not consume ground truth')
    if online.get('future_frames_used') is not False:
        raise ValueError('V5.1 runtime must remain causal')
    if online.get('domain_identity_used_in_decisions') is not False:
        raise ValueError('V5.1 runtime must not route decisions by domain')
    policy = calibration.get('component_policy', {})
    required = {
        'center': ('anchor_score', 'unavailable', 0),
        'scale': ('v5_supervised_scale_error_risk', 'unavailable', 0),
        'angle': ('anchor_score', 'bounded_last_measurement_hold', 1)}
    for component, expected in required.items():
        item = policy.get(component, {})
        actual = (item.get('gate'), item.get('rejected_or_missing'),
                  int(item.get('max_prediction_frames', -1)))
        if actual != expected:
            raise ValueError('Unexpected V5.1 runtime {} policy'.format(
                component))
    ranker = calibration.get('scale_error_ranker', {})
    if ranker.get('online_gt_fields_consumed') != []:
        raise ValueError('Scale ranker must not consume online GT')
    names = ranker.get('feature_names', [])
    expanded = ranker.get('expanded_feature_names', [])
    if len(expanded) != 2*len(names):
        raise ValueError('Scale ranker feature schema is inconsistent')
    if len(ranker.get('weights', [])) != len(expanded)+1:
        raise ValueError('Scale ranker weight dimension is inconsistent')


def scale_error_risk(features, calibration):
    ranker = calibration['scale_error_ranker']
    names = ranker['feature_names']
    raw = np.asarray([
        np.nan if features.get(name) is None else float(features[name])
        for name in names], dtype=np.float64)
    missing = np.isnan(raw).astype(np.float64)
    preprocessing = ranker['preprocessing']
    medians = np.asarray(preprocessing['medians'], dtype=np.float64)
    raw = np.where(np.isnan(raw), medians, raw)
    combined = np.concatenate([raw, missing])
    means = np.asarray(preprocessing['means'], dtype=np.float64)
    scales = np.asarray(preprocessing['scales'], dtype=np.float64)
    vector = np.concatenate([[1.0], (combined-means)/scales])
    weights = np.asarray(ranker['weights'], dtype=np.float64)
    logit = float(np.clip(vector.dot(weights), -30.0, 30.0))
    return float(1.0/(1.0+np.exp(-logit)))


class HybridObservationManagerV51:
    """Convert one causal detector row into component observation states."""

    def __init__(self, calibration):
        validate_runtime_calibration(calibration)
        self.calibration = copy.deepcopy(calibration)
        self.reset()

    def reset(self):
        self._raw_history = []
        self._last_angle_measurement = None
        self._last_identity = None
        self._last_frame = None

    def _boundary(self, row):
        identity = (row['domain'], row['sequence'])
        return (identity != self._last_identity or self._last_frame is None or
                int(row['frame']) != self._last_frame+1)

    def update(self, row):
        """Emit states without reading GT, future frames, or domain routing."""
        if self._boundary(row):
            self.reset()
        identity = (row['domain'], row['sequence'])
        available = row.get('base_v3_box') is not None
        if available:
            features = _frame_features(row, self._raw_history)
            score = row.get('anchor_score')
            score_accept = (score is not None and float(score) >= float(
                self.calibration['score_gate']['score_threshold']))
            risk = scale_error_risk(features, self.calibration)
            scale_accept = risk <= float(
                self.calibration['scale_error_ranker']['risk_threshold'])
        else:
            features, score_accept, scale_accept, risk = None, False, False, None
        decisions = dict(
            center=bool(score_accept), scale=bool(scale_accept),
            angle=bool(score_accept))
        observations = {}
        for component in COMPONENTS:
            accepted = decisions[component]
            if accepted:
                value = _measurement_value(row, component)
                state, source, age = 'measurement', 'base_v3_measurement', 0
                if component == 'angle':
                    self._last_angle_measurement = dict(
                        frame=int(row['frame']), value=copy.deepcopy(value))
            else:
                state, source, value = 'unavailable', None, None
                age = (None if component != 'angle' or
                       self._last_angle_measurement is None else
                       int(row['frame'])-
                       self._last_angle_measurement['frame'])
                if (component == 'angle' and
                        self._last_angle_measurement is not None and age == 1):
                    state = 'prediction'
                    source = 'one_frame_angle_hold'
                    value = copy.deepcopy(
                        self._last_angle_measurement['value'])
            observations[component] = dict(
                value=value, valid=state != 'unavailable', state=state,
                source=source, age_since_measurement_frames=age,
                measurement_available=available,
                measurement_accepted=accepted,
                risk=risk if component == 'scale' else None,
                gate=('v5_supervised_scale_error_risk'
                      if component == 'scale' else 'anchor_score'),
                reason=(('accepted_measurement' if accepted else
                         'detector_missing' if not available else
                         'rejected_measurement')))
        if available:
            self._raw_history.append(dict(
                frame=int(row['frame']),
                base_v3_box=copy.deepcopy(row['base_v3_box'])))
            self._raw_history = self._raw_history[-2:]
        else:
            self._raw_history = []
        self._last_identity = identity
        self._last_frame = int(row['frame'])
        return dict(
            frame_key=row['frame_key'], domain=row['domain'],
            sequence=row['sequence'], frame=int(row['frame']),
            online_features=features, components=observations)
