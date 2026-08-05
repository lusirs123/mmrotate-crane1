import copy

from crane_project.tools import (
    dino_teacher_s7_temporal_student_fixed_target_audit as audit)


def _summary(frame_count, top1_hits, mcml, recall):
    return dict(
        frame_count=frame_count, top1_hits=top1_hits, top1_mcml=mcml,
        recall_at_100=recall, top1_dfr_fraction_per_frame=0.08,
        top1_aci=0.83)


def _stage3_result():
    full = _summary(738, 691, 3, 738)
    small = _summary(350, 312, 3, 350)
    row = dict(
        epoch=1, selected_as_best=True, checkpoint_saved=True,
        source_val=full, source_small_val=small,
        source_selection_gate_passed=True, source_retention_passed=True,
        source_exact_retention=dict(
            baseline_correct_count=677, retained_correct_count=677,
            lost_correct_count=0),
        source_selection_gate=dict(
            passed=True,
            checks=dict(source_dfr_nonregression=True,
                        source_aci_nonregression=True)),
        s7_source_gate=dict(passed=True))
    return dict(
        protocol_version=22,
        source_selected_checkpoint='/tmp/stage3.pth',
        source=dict(
            best_epoch=1, history=[row],
            initial_teacher_reproduction_gate=dict(passed=True)),
        protocol=dict(
            source_candidate_student_training=True,
            s7_temporal_association=dict(target_read=False),
            s7_temporal_student=dict(source_only=True, target_read=False)),
        isolation=dict(
            train_components='s7_temporal_student',
            dino_parameters_unchanged=True,
            frozen_head_parameters_unchanged=True,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_labels_used_for_evaluation_only=False),
        target_dev=None)


def test_stage3_source_gate_accepts_positive_student_result():
    result = audit.strict_stage3_source_gate(_stage3_result())
    assert result['passed'] is True


def test_stage3_source_gate_rejects_phase2_association_result():
    result = _stage3_result()
    result['isolation']['train_components'] = 's7_temporal_association'
    gate = audit.strict_stage3_source_gate(result)
    assert gate['passed'] is False
    assert gate['checks']['student_training_protocol'] is False


def test_stage3_checkpoint_gate_requires_student_mode_and_reproduction():
    payload = dict(
        best_epoch=1, epoch=1, s7_inference_enabled=True,
        best_source_val_summary=_summary(738, 691, 3, 738),
        best_source_small_val_summary=_summary(350, 312, 3, 350),
        source_selection_gate_passed=True,
        source_exact_retention=dict(
            baseline_correct_count=677, retained_correct_count=677,
            lost_correct_count=0),
        s7_architecture=dict(
            temporal_association=True, temporal_quality_head=True,
            temporal_student=True, temporal_min_confirmations=1),
        training_protocol=dict(
            train_components='s7_temporal_student',
            s7_temporal_association=dict(
                relative_quality=True, candidate_quality_head=True),
            s7_temporal_student=dict(
                base_epoch=4, source_only=True, target_read=False)),
        s7_student_teacher_reproduction=dict(passed=True))
    gate = audit.candidate_checkpoint_gate(payload, dict(best_epoch=1))
    assert gate['passed'] is True

    broken = copy.deepcopy(payload)
    broken['training_protocol']['train_components'] = (
        's7_temporal_association')
    assert audit.candidate_checkpoint_gate(
        broken, dict(best_epoch=1))['passed'] is False
