import math

import numpy as np
import pytest

from crane_project.utils.depth_interface_geometry_gate import (
    depth_interface_geometry_gate, geometry_metrics)


def _metadata(count=100):
    records = []
    for index in range(count):
        records.append(dict(
            domain='sim' if index % 2 else 'real',
            gt_box=np.asarray([
                100.0 + index, 80.0, 40.0, 20.0, 0.1],
                dtype=np.float64)))
    return records


def _boxes(metadata, long_scale=1.0, short_scale=1.0,
           center_dx=0.0, angle_delta=0.0):
    return [np.asarray([
        item['gt_box'][0] + center_dx,
        item['gt_box'][1],
        item['gt_box'][2] * long_scale,
        item['gt_box'][3] * short_scale,
        item['gt_box'][4] + angle_delta], dtype=np.float64)
            for item in metadata]


def test_geometry_metrics_are_periodic_and_edge_order_invariant():
    metadata = _metadata(4)
    predictions = []
    for item in metadata:
        gt = item['gt_box']
        predictions.append(np.asarray([
            gt[0], gt[1], gt[3], gt[2], gt[4] - math.pi / 2.0]))
    metrics = geometry_metrics(metadata, predictions)
    assert metrics['all']['center_error_px']['mean_abs'] == 0.0
    assert metrics['all']['angle_error_deg']['mean_abs'] == pytest.approx(0.0)
    assert metrics['all']['q_log_aspect_residual']['mean_abs'] == 0.0


def test_source_relative_depth_interface_gate_accepts_preserved_geometry():
    metadata = _metadata()
    reference = _boxes(metadata, long_scale=0.98, short_scale=0.98)
    candidate = _boxes(
        metadata, long_scale=0.985, short_scale=0.98,
        center_dx=0.2, angle_delta=math.radians(0.2))
    report = depth_interface_geometry_gate(
        metadata, candidate, reference)
    assert report['passed'] is True
    assert all(report['checks'].values())
    assert report['gated_scopes'] == ['all', 'sim']


def test_source_relative_depth_interface_gate_rejects_aspect_extrapolation():
    metadata = _metadata()
    reference = _boxes(metadata, long_scale=0.98, short_scale=0.98)
    candidate = _boxes(metadata, long_scale=0.80, short_scale=1.20)
    report = depth_interface_geometry_gate(
        metadata, candidate, reference)
    assert report['passed'] is False
    assert report['checks']['all_q_mean'] is False
    assert report['checks']['sim_q_envelope_exceedance'] is False
    assert report['candidate']['sim']['q_log_aspect_residual'][
        'mean_abs'] > 0.3


def test_gate_reports_real_geometry_but_does_not_turn_it_into_depth_truth():
    metadata = _metadata()
    reference = _boxes(metadata)
    candidate = _boxes(metadata)
    report = depth_interface_geometry_gate(
        metadata, candidate, reference)
    assert 'real' in report['candidate']
    assert report['real_scope_role'].startswith('reported_only')
    assert not any(key.startswith('real_') for key in report['checks'])


def test_gate_fails_cleanly_when_candidate_has_no_valid_geometry():
    metadata = _metadata()
    reference = _boxes(metadata)
    report = depth_interface_geometry_gate(
        metadata, [None] * len(metadata), reference)
    assert report['passed'] is False
    assert report['candidate']['sim']['valid_count'] == 0
    assert report['checks']['sim_q_mean'] is False
