import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from crane_project.tools import base_v3_obb_focused_paper_result_archive_v31 as archive


def fixture_inputs(tmp_path):
    point = dict(target_coverage=1.0,
                 anchor_score=dict(retained_count=1, mean_error=0.1, failure_rate=0.),
                 learned_scale_risk=dict(retained_count=1, mean_error=0.1, failure_rate=0.),
                 learned_minus_score=dict(mean_error=0., failure_rate=0.))
    angle = {g: dict(mean_error_delta_vs_raw=0.2, prediction_cases=[])
             for g in ('all', 'real', 'sim')}
    historical = dict(protocol='historical', fixed_test_previously_exposed=True,
                      parameter_tuning_authorized=False, input=dict(sha256='f'*64),
                      scale_matched_coverage={'all': dict(points=[point])},
                      angle_hold_pair_audit=angle)
    online = dict(protocol='online', fixed_test_previously_exposed=True,
                  parameter_tuning_authorized=False, pipeline=['test_fixture'],
                  dataset_summary={}, evaluation_thresholds={}, paper_tables={},
                  front_end_summary={}, inputs={}, runtime_measurement={},
                  supporting_diagnostics=dict(angle_hold_pair_audit=copy.deepcopy(angle)))
    online['supporting_diagnostics']['angle_hold_pair_audit']['all']['mean_error_delta_vs_raw'] = 0.3
    contract = dict(protocol=archive.CONTRACT_PROTOCOL, frozen_eval_sha256='f'*64)
    for name, payload in [('online', online), ('historical_diagnostic', historical)]:
        p = tmp_path / (name + '.json')
        p.write_text(json.dumps(payload))
        contract[name] = dict(protocol=payload['protocol'],
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (tmp_path / 'contract.json').write_text(json.dumps(contract))
    return historical, online, contract


def test_archive_cli_is_reproducible_and_preserves_sources(tmp_path):
    historical, online, contract = fixture_inputs(tmp_path)
    cmd = [sys.executable, '-m', archive.__name__, '--online',
           str(tmp_path / 'online.json'), '--historical-diagnostic',
           str(tmp_path / 'historical_diagnostic.json'), '--contract',
           str(tmp_path / 'contract.json'), '--out-dir', str(tmp_path / 'out')]
    for _ in range(2):
        subprocess.run(cmd, check=True, capture_output=True,
                       cwd=Path(__file__).parents[1])
    result = json.loads((tmp_path / 'out/fixed_test_focused_paper_result_archive_v31_r1.json').read_text())
    assert result['source_differences']['all']['online_mean_error_delta_vs_raw'] == 0.3
    assert result['source_differences']['all']['frozen_mean_error_delta_vs_raw'] == 0.2
    assert result['overall_winner_claimed'] is False
    assert json.loads((tmp_path / 'historical_diagnostic.json').read_text()) == historical
    assert (tmp_path / 'out/fixed_test_focused_paper_result_archive_v31_r1.md').exists()
    (tmp_path / 'online.json').write_text('{}')
    failed = subprocess.run(cmd, capture_output=True)
    assert failed.returncode != 0
    assert b'SHA256 mismatch' in failed.stderr


def test_revision_preserves_metrics_and_rejects_inconsistent_delta(tmp_path):
    history, _, _ = fixture_inputs(tmp_path)
    before = copy.deepcopy(history)
    revised = archive.revision_from_history(history, {'sha256': 'a'*64})
    assert history == before
    assert revised['angle_hold_pair_audit'] == before['angle_hold_pair_audit']
    point = revised['scale_matched_coverage']['all']['points'][0]
    assert point['learned_minus_score']['mean_error'] == 0.
    assert point['learned_minus_score']['tie_mean_and_failure'] is True
    history['scale_matched_coverage']['all']['points'][0]['learned_minus_score']['mean_error'] = 1.
    with pytest.raises(ValueError, match='Inconsistent'):
        archive.revision_from_history(history, {})
