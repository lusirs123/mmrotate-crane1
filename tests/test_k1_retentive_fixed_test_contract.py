from pathlib import Path

from crane_project.tools.symeood_dino_k1_retentive_promote import (
    LOCKED_CHECKPOINT_SHA256, LOCKED_EPOCH, LOCKED_GATE_SHA256,
    LOCKED_RESULTS_SHA256, PROMOTION_PROTOCOL)


ROOT = Path(__file__).resolve().parents[1]


def test_v3_promotion_locks_complete_source_run_and_epoch9():
    assert LOCKED_EPOCH == 9
    assert len(LOCKED_CHECKPOINT_SHA256) == 64
    assert len(LOCKED_RESULTS_SHA256) == 64
    assert sorted(LOCKED_GATE_SHA256) == list(range(1, 11))
    assert all(len(value) == 64 for value in LOCKED_GATE_SHA256.values())
    assert PROMOTION_PROTOCOL == (
        'source_gated_k1_retentive_causal_phase_refiner_v3')
    text = (ROOT / ('crane_project/tools/'
                    'symeood_dino_k1_retentive_promote.py')).read_text()
    assert 'selection = select(source_gates)' in text
    assert "passing_epochs') != [LOCKED_EPOCH]" in text
    assert 'fixed_benchmark_test=True' in text
    assert 'test_used_for_model_selection=False' in text
    assert 'parameter_update_after_test=False' in text


def test_v3_fixed_test_uses_promoted_epoch9_and_cached_inputs():
    text = (ROOT / ('crane_project/configs/'
                    'crane_symeood_dino_k1_retentive_causal_phase_'
                    'fixed_test.py')).read_text()
    assert 'selected_source_epoch=9' in text
    assert "ann_file='test/annfiles/'" in text
    assert "img_prefix='test/images/'" in text
    assert "type='LoadDinoProposalFromAudit'" in text
    assert "type='LoadCausalHistoryFromAudit'" in text
    assert 'expected_frame_count=992' in text
    assert "expected_split='test'" in text
    assert 'evaluation_only=True' in text
    assert 'retention_loss_weight=0.0' in text
    assert 'temporal_size_loss_weight=0.0' in text
    assert 'domain_routing=False' in text
    assert 'sequence_frame_routing=False' in text
    assert 'temporal_state=False' in text
    assert 'test_used_for_model_selection=False' in text
    assert 'parameter_update_after_test=False' in text
