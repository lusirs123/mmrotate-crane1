import json

import pytest

from crane_project.tools.symeood_dino_k1_retentive_source_select import (
    SELECTION_PROTOCOL, SOURCE_GATE_PROTOCOL, select)


def _gate(tmp_path, epoch, passed=True, riou=0.8, mcml=2, dfr=1.0):
    path = tmp_path / ('gate_epoch_{:02d}.json'.format(epoch))
    checkpoint = str(tmp_path / 'epoch_{}.pth'.format(epoch))
    results = str(tmp_path / 'source_val_epoch_{}.pkl'.format(epoch))
    payload = dict(
        protocol=SOURCE_GATE_PROTOCOL,
        evidence_boundary='source_val_only',
        target_data_read=False,
        fixed_test_read=False,
        passed=passed,
        eligible_for_checkpoint_promotion=passed,
        eligible_for_fixed_test=False,
        eligible_for_unknown_sequence_claim=False,
        decision=(
            'ALLOW_K1_RETENTIVE_CAUSAL_PHASE_CHECKPOINT_PROMOTION'
            if passed else 'STOP_CAUSAL_HISTORY_SOURCE_GATE_FAILED'),
        input=dict(
            candidate_checkpoint=checkpoint,
            candidate_checkpoint_sha256='c{:02d}'.format(epoch),
            candidate_results=results,
            candidate_results_sha256='r{:02d}'.format(epoch)),
        depth_interface_geometry_gate=dict(passed=passed),
        candidate_metrics={
            'real/MCML_max(frames)': mcml,
            'sim/MCML_max(frames)': mcml,
            'real/mean_RIoU': riou,
            'sim/mean_RIoU': riou,
            'real/DFR(%/frame)': dfr,
            'sim/DFR(%/frame)': dfr})
    path.write_text(json.dumps(payload))
    return str(path)


def test_selector_uses_only_passing_source_gates_and_locked_ranking(tmp_path):
    paths = []
    for epoch in range(1, 11):
        paths.append(_gate(
            tmp_path, epoch, passed=epoch != 3,
            riou=0.90 if epoch == 7 else 0.80,
            mcml=1 if epoch in (3, 7) else 2))
    report = select(paths)
    assert report['protocol'] == SELECTION_PROTOCOL
    assert report['selected']['epoch'] == 7
    assert 3 not in report['passing_epochs']
    assert report['eligible_for_fixed_test'] is False
    assert report['eligible_for_unknown_sequence_claim'] is False


def test_selector_stops_when_no_epoch_passes(tmp_path):
    paths = [_gate(tmp_path, epoch, passed=False)
             for epoch in range(1, 11)]
    with pytest.raises(RuntimeError, match='No V3 epoch passed'):
        select(paths)
