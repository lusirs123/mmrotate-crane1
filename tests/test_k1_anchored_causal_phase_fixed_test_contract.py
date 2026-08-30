from pathlib import Path

from crane_project.tools.symeood_dino_k1_anchored_causal_phase_promote import (
    LOCKED_CHECKPOINT_SHA256, LOCKED_EPOCH, LOCKED_GATE_SHA256,
    LOCKED_RESULTS_SHA256,
    SELECTION_POLICY, _rank)


ROOT = Path(__file__).resolve().parents[1]


def test_source_selection_policy_prefers_epoch10_balanced_candidate():
    def row(epoch, real_mcml, sim_mcml, real_riou, sim_riou, dfr):
        return dict(epoch=epoch, metrics={
            'real/MCML_max(frames)': real_mcml,
            'sim/MCML_max(frames)': sim_mcml,
            'real/mean_RIoU': real_riou,
            'sim/mean_RIoU': sim_riou,
            'real/DFR(%/frame)': dfr,
            'sim/DFR(%/frame)': dfr})
    rows = [
        row(6, 0, 1, 0.76, 0.81, 4.4),
        row(10, 0, 0, 0.75, 0.80, 4.5),
        row(9, 1, 0, 0.78, 0.83, 4.2)]
    assert min(rows, key=_rank)['epoch'] == 10
    assert LOCKED_EPOCH == 10
    assert len(LOCKED_CHECKPOINT_SHA256) == 64
    assert len(LOCKED_RESULTS_SHA256) == 64
    assert sorted(LOCKED_GATE_SHA256) == list(range(1, 11))
    assert all(len(value) == 64 for value in LOCKED_GATE_SHA256.values())
    assert SELECTION_POLICY == 'min_worst_mcml_then_max_combined_riou_v1'


def test_promotion_recomputes_all_ten_gates_and_locks_artifact_hashes():
    path = ROOT / ('crane_project/tools/'
                   'symeood_dino_k1_anchored_causal_phase_promote.py')
    text = path.read_text()
    assert 'len(source_gates) != 10' in text
    assert 'list(range(1, 11))' in text
    assert 'selected_source_epoch=LOCKED_EPOCH' in text
    assert 'one_fixed_test_only=True' in text
    assert 'fixed_test_consumed=False' in text
    assert "eligible_for_one_fixed_test=True" in text
    assert "target_data_read=False, fixed_test_read=False" in text


def test_fixed_test_config_uses_cached_current_and_causal_history_only():
    path = ROOT / ('crane_project/configs/'
                   'crane_symeood_dino_k1_anchored_causal_phase_fixed_test.py')
    text = path.read_text()
    assert "type='LoadDinoProposalFromAudit'" in text
    assert "type='LoadCausalHistoryFromAudit'" in text
    assert "type='PrepareCausalHistoryInputs'" in text
    assert "type='FormatCausalHistoryInputs'" in text
    assert 'expected_frame_count=992' in text
    assert "expected_split='test'" in text
    assert "ann_file='test/annfiles/'" in text
    assert 'evaluation_only=True' in text
    assert 'selected_source_epoch=10' in text
    assert 'domain_routing=False' in text
    assert 'sequence_frame_routing=False' in text
    assert 'temporal_state=False' in text
    assert 'load_from = None' in text
    assert 'resume_from = None' in text


def test_fixed_test_audit_cannot_reselect_or_tune_from_test():
    path = ROOT / ('crane_project/tools/'
                   'symeood_dino_k1_anchored_causal_phase_fixed_test_audit.py')
    text = path.read_text()
    assert 'EXPECTED_FRAME_COUNT' in text
    assert "real_mcml_max_le_6" in text
    assert "sim_mcml_max_le_5" in text
    assert 'parameter_update_after_test=False' in text
    assert 'epoch_reselection_after_test=False' in text
    assert 'eligible_for_parameter_tuning_from_this_report=False' in text
    assert 'eligible_for_epoch_reselection_from_this_report=False' in text
    assert 'used_for_primary_decision=False' in text
