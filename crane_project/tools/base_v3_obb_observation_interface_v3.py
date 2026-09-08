#!/usr/bin/env python3
"""Build and replay the Base V3 component observation interface.

The online interface consumes predictions only and emits an explicit state for
center, scale and angle. Ground truth is attached only by the offline replay
code after the observation has been emitted.
"""

import argparse
import bisect
import copy
import json
from pathlib import Path

import numpy as np

from crane_project.tools.base_v3_obb_component_reliability_ablation_v2 import (
    evaluate, synthetic_dropout_keys)
from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS, _error_from_value, _frame_features, _measurement_value,
    _raw_component_error, assign_split, build_feature_rows)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    _identity, _write_exact)
from crane_project.tools.base_v3_reliability_failure_audit import validate


PROTOCOL = 'base_v3_obb_observation_interface_v3'
CALIBRATION_PROTOCOL = 'base_v3_obb_observation_calibration_v3'
CONTRACT_PROTOCOL = 'base_v3_obb_observation_interface_contract_v3'


def _cdf(value, sorted_values):
    if value is None or not sorted_values:
        return None
    return bisect.bisect_right(sorted_values, float(value))/len(sorted_values)


def _quantile(values, q):
    if not values:
        raise ValueError('Cannot calibrate an empty distribution')
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _component_risk(features, component, calibration):
    names = calibration['component_features'][component]
    ranks = [_cdf(features.get(name),
                  calibration['empirical_cdf_values'][name])
             for name in names]
    available = [rank for rank in ranks if rank is not None]
    return float(np.mean(available)) if available else 1.0


def build_calibration(rows, contract):
    """Fit empirical ranks and thresholds from calibration-prefix rows only."""
    calibration_rows = [row for row in rows
                        if row['split'] == 'calibration']
    selected_groups = contract['selected_feature_groups']
    component_features = {
        component: list(contract['feature_groups'][component][group])
        for component, group in selected_groups.items()}
    feature_names = sorted({name for names in component_features.values()
                            for name in names})
    distributions = {
        name: sorted(float(row['online_features'][name])
                     for row in calibration_rows
                     if row['online_features'].get(name) is not None)
        for name in feature_names}
    provisional = dict(
        component_features=component_features,
        empirical_cdf_values=distributions)
    risks = {component: [
        _component_risk(row['online_features'], component, provisional)
        for row in calibration_rows] for component in COMPONENTS}
    target = float(contract['target_measurement_coverage'])
    return dict(
        protocol=CALIBRATION_PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        target_measurement_coverage=target,
        selected_feature_groups=selected_groups,
        component_features=component_features,
        empirical_cdf_values=distributions,
        component_risk_thresholds={
            component: _quantile(risks[component], target)
            for component in COMPONENTS},
        score_threshold=_quantile(
            [float(row['anchor_score']) for row in calibration_rows],
            1.0-target),
        component_output_policy=copy.deepcopy(
            contract['component_output_policy']),
        online_feature_contract=dict(
            gt_fields_consumed=[], domain_identity_used_in_decision=False,
            future_frames_used=False),
        claim_limit=contract['claim_limit'])


class ComponentObservationManager:
    """Causal stateful adapter from detector rows to component observations."""

    def __init__(self, calibration, gate_mode='component'):
        if calibration.get('protocol') != CALIBRATION_PROTOCOL:
            raise ValueError('Unexpected observation calibration protocol')
        if gate_mode not in ('raw', 'score', 'component'):
            raise ValueError('gate_mode must be raw, score, or component')
        self.calibration = calibration
        self.gate_mode = gate_mode
        self.reset()

    def reset(self):
        self._raw_history = []
        self._last_measurement = {component: None for component in COMPONENTS}
        self._last_identity = None
        self._last_frame = None

    def _sequence_boundary(self, row):
        identity = (row['domain'], row['sequence'])
        return (identity != self._last_identity or self._last_frame is None
                or int(row['frame']) != self._last_frame+1)

    def _accepted(self, row, risks):
        if row.get('base_v3_box') is None:
            return {component: False for component in COMPONENTS}
        if self.gate_mode == 'raw':
            return {component: True for component in COMPONENTS}
        if self.gate_mode == 'score':
            score = row.get('anchor_score')
            value = (score is not None and
                     float(score) >= self.calibration['score_threshold'])
            return {component: value for component in COMPONENTS}
        return {
            component: risks[component] <= self.calibration[
                'component_risk_thresholds'][component]
            for component in COMPONENTS}

    def update(self, row):
        """Emit online-only values and states for one chronological frame."""
        if self._sequence_boundary(row):
            self.reset()
        identity = (row['domain'], row['sequence'])
        measurement_available = row.get('base_v3_box') is not None
        if measurement_available:
            features = _frame_features(row, self._raw_history)
            risks = {component: _component_risk(
                features, component, self.calibration)
                for component in COMPONENTS}
        else:
            features = None
            risks = {component: None for component in COMPONENTS}
        accepted = self._accepted(row, risks)
        observations = {}
        for component in COMPONENTS:
            if accepted[component]:
                value = _measurement_value(row, component)
                self._last_measurement[component] = dict(
                    frame=int(row['frame']), value=value)
                state = 'measurement'
                source = 'base_v3_measurement'
                age = 0
                reason = 'accepted_{}_gate'.format(self.gate_mode)
            else:
                previous = self._last_measurement[component]
                age = (None if previous is None else
                       int(row['frame'])-previous['frame'])
                policy = self.calibration['component_output_policy'][component]
                may_hold = (policy['rejected_or_missing'] ==
                            'bounded_last_measurement_hold')
                if (may_hold and previous is not None and age > 0 and
                        age <= int(policy['max_hold_frames'])):
                    state = 'prediction'
                    value = copy.deepcopy(previous['value'])
                    source = 'bounded_last_measurement_hold'
                else:
                    state = 'unavailable'
                    value = None
                    source = None
                reason = ('detector_missing' if not measurement_available
                          else 'rejected_{}_gate'.format(self.gate_mode))
            observations[component] = dict(
                value=value,
                valid=state != 'unavailable',
                state=state,
                source=source,
                age_since_measurement_frames=age,
                measurement_available=measurement_available,
                measurement_accepted=accepted[component],
                risk=risks[component],
                gate_threshold=(None if self.gate_mode == 'raw' else
                    (self.calibration['score_threshold']
                     if self.gate_mode == 'score' else
                     self.calibration['component_risk_thresholds'][component])),
                reason=reason)
        if measurement_available:
            self._raw_history.append({key: copy.deepcopy(row[key]) for key in (
                'frame', 'base_v3_box')})
            self._raw_history = self._raw_history[-2:]
        else:
            # Never extrapolate a causal residual across an actual detector miss.
            self._raw_history = []
        self._last_identity = identity
        self._last_frame = int(row['frame'])
        return dict(
            frame_key=row['frame_key'], domain=row['domain'],
            sequence=row['sequence'], frame=int(row['frame']),
            gate_mode=self.gate_mode, online_features=features,
            components=observations)


ONLINE_INPUT_KEYS = (
    'frame_key', 'domain', 'sequence', 'frame', 'anchor_source',
    'anchor_score', 'base_v3_box', 'k1_box', 'dino_box')


def _online_row(row):
    return {key: copy.deepcopy(row.get(key)) for key in ONLINE_INPUT_KEYS}


def _replay(rows, calibration, gate_mode, forced_missing=None):
    manager = ComponentObservationManager(calibration, gate_mode)
    forced_missing = set(forced_missing or ())
    outputs = {}
    for row in sorted(rows, key=lambda item: (
            item['domain'], item['sequence'], item['frame'], item['frame_key'])):
        online = _online_row(row)
        if row['frame_key'] in forced_missing:
            online['base_v3_box'] = None
        outputs[row['frame_key']] = manager.update(online)
    return outputs


def _evaluation_views(rows, outputs):
    states = {}
    accepted = {}
    for row in rows:
        result = outputs[row['frame_key']]
        states[row['frame_key']] = {}
        accepted[row['frame_key']] = {}
        for component in COMPONENTS:
            observation = result['components'][component]
            error = (None if observation['value'] is None else
                     _error_from_value(observation['value'], row, component))
            states[row['frame_key']][component] = dict(
                state=observation['state'], value=observation['value'],
                error=error)
            accepted[row['frame_key']][component] = bool(
                observation['measurement_accepted'])
    return states, accepted


def _thresholds(contract):
    return dict(
        center=contract['evaluation']['center_error_threshold_px'],
        scale=contract['evaluation']['scale_relative_error_threshold'],
        angle=contract['evaluation']['angle_error_threshold_deg'])


def _report(rows, outputs, contract):
    states, accepted = _evaluation_views(rows, outputs)
    return evaluate(rows, accepted, states, _thresholds(contract),
                    contract['evaluation']['tail_quantiles'])


def _direct_risk_curves(rows, calibration, contract):
    """Compare component risk with the untransformed raw anchor score."""
    result = {}
    error_thresholds = _thresholds(contract)
    for component in COMPONENTS:
        result[component] = {}
        orderings = {
            'raw_anchor_score': sorted(
                rows, key=lambda row: (-float(row['anchor_score']),
                                       row['frame_key'])),
            'component_risk': sorted(rows, key=lambda row: (
                _component_risk(row['online_features'], component,
                                calibration), row['frame_key']))}
        for method, ordered in orderings.items():
            curve = []
            for coverage in contract['risk_coverage_targets']:
                retained = ordered[:max(1, round(len(ordered)*coverage))]
                errors = [_raw_component_error(row, component)
                          for row in retained]
                curve.append(dict(
                    target_coverage=float(coverage),
                    retained_count=len(retained),
                    mean_error=float(np.mean(errors)),
                    failure_rate=sum(error > error_thresholds[component]
                                     for error in errors)/len(errors),
                    retained_frame_keys=[row['frame_key'] for row in retained]))
            result[component][method] = curve
    return result


def run(baseline, baseline_contract, contract):
    validate(baseline, baseline_contract)
    rows = build_feature_rows(baseline['records'])
    split_counts = assign_split(
        rows, float(contract['split']['calibration_prefix_fraction']),
        int(contract['split']['minimum_calibration_frames_per_segment']))
    calibration = build_calibration(rows, contract)
    evaluation_rows = [row for row in rows if row['split'] == 'evaluation']
    outputs_by_method = {
        gate: _replay(evaluation_rows, calibration, gate)
        for gate in ('raw', 'score', 'component')}
    method_reports = {}
    for gate, outputs in outputs_by_method.items():
        method_reports[gate] = {}
        for group, selected in [('all', evaluation_rows)] + [
                (domain, [row for row in evaluation_rows
                          if row['domain'] == domain])
                for domain in ('real', 'sim')]:
            method_reports[gate][group] = _report(
                selected, outputs, contract)

    dropout_contract = contract['synthetic_missing_stress']
    dropout_keys, dropout_runs = synthetic_dropout_keys(
        evaluation_rows, dropout_contract)
    dropout_outputs = (_replay(evaluation_rows, calibration, 'raw', dropout_keys)
                       if dropout_contract['enabled'] else {})
    dropout_report = (_report(evaluation_rows, dropout_outputs, contract)
                      if dropout_outputs else {})

    output_rows = []
    for row in rows:
        item = {key: row[key] for key in (
            'frame_key', 'domain', 'sequence', 'frame', 'split',
            'anchor_source', 'anchor_score')}
        item['online_features'] = row['online_features']
        item['offline_errors'] = row['errors']
        if row['split'] == 'evaluation':
            item['observations'] = {
                gate: outputs[row['frame_key']]['components']
                for gate, outputs in outputs_by_method.items()}
            item['synthetic_missing'] = row['frame_key'] in dropout_keys
            if dropout_outputs:
                item['synthetic_missing_observation'] = dropout_outputs[
                    row['frame_key']]['components']
        output_rows.append(item)

    curves = {group: _direct_risk_curves(selected, calibration, contract)
              for group, selected in [('all', evaluation_rows)] + [
                  (domain, [row for row in evaluation_rows
                            if row['domain'] == domain])
                  for domain in ('real', 'sim')]}
    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        fixed_test_read=False, target_data_read=False,
        claim_status='DEVELOPMENT_REPLAY_INTERFACE_ONLY',
        selection_status=(
            'FEATURE_GROUPS_SELECTED_AFTER_V2_EXPLORATORY_EVALUATION'),
        leakage_controls=dict(
            gt_used_in_online_interface=False,
            gt_used_in_calibration=False,
            gt_attached_after_online_output=True,
            domain_identity_used_in_decisions=False,
            future_frames_used=False,
            state_reset_on_sequence_or_frame_gap=True,
            synthetic_missing_independent_of_gt=True),
        split_counts=split_counts,
        calibration=calibration,
        online_interface_schema=dict(
            component_fields=['value', 'valid', 'state', 'source',
                              'age_since_measurement_frames',
                              'measurement_available', 'measurement_accepted',
                              'risk', 'gate_threshold', 'reason'],
            allowed_states=['measurement', 'prediction', 'unavailable'],
            meaning_of_continuous=(
                'Every frame has an explicit state; unavailable has no value.')),
        method_reports=method_reports,
        corrected_matched_coverage=dict(
            score_ordering='descending raw anchor_score; no empirical-CDF ties',
            curves=curves),
        synthetic_missing_stress=dict(
            frame_count=len(dropout_keys), runs=dropout_runs,
            report=dropout_report,
            interpretation=(
                'Artificial detector-missing stress only; not real miss evidence.')),
        records=output_rows,
        claim_limit=contract['claim_limit'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--baseline-contract', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--runtime-calibration-out', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text())
    baseline_contract = json.loads(Path(args.baseline_contract).read_text())
    contract = json.loads(Path(args.contract).read_text())
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected V3 observation-interface contract')
    payload = run(baseline, baseline_contract, contract)
    calibration = copy.deepcopy(payload['calibration'])
    calibration['inputs'] = dict(
        baseline=_identity(args.baseline),
        baseline_contract=_identity(args.baseline_contract),
        contract=_identity(args.contract))
    calibration_output = _write_exact(
        args.runtime_calibration_out, calibration)
    payload['inputs'] = calibration['inputs']
    payload['runtime_calibration'] = calibration_output
    output = _write_exact(args.out_json, payload)
    print(json.dumps(dict(
        output=output,
        runtime_calibration=calibration_output,
        split_counts=payload['split_counts'],
        synthetic_missing_frames=payload['synthetic_missing_stress'][
            'frame_count'],
        claim_status=payload['claim_status'],
        leakage_controls=payload['leakage_controls']), indent=2))


if __name__ == '__main__':
    main()
