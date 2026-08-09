from argparse import Namespace
from copy import deepcopy

from crane_project.tools import (
    dino_teacher_s7_highres_fixed_target_audit as audit)


def _source_margin_result():
    gate_checks = dict(
        source_dfr_nonregression=True,
        source_aci_nonregression=True)
    selected = dict(
        promotion_margin=0.225,
        full_summary=dict(frame_count=738, top1_hits=688, top1_mcml=3),
        small_summary=dict(frame_count=350, top1_hits=311, top1_mcml=3),
        source_exact_retention=dict(
            baseline_correct_count=677, retained_correct_count=677,
            lost_correct_count=0, gained_correct_count=11),
        source_gate=dict(passed=True, checks=gate_checks),
        gate_passed=True, epoch3_reference_reproduced=None)
    return dict(
        protocol_version=24,
        decision=(
            'SOURCE_ONLY_HIGHRES_MARGIN_AUDIT_GATE_PASSED_TARGET_NOT_READ'),
        protocol=dict(
            fixed_margins=[0.2, 0.225, 0.25], shared_model_forward=True,
            parameter_update=False, source_only=True, target_read=False),
        isolation=dict(
            dino_parameters_unchanged=True,
            detector_parameters_unchanged=True,
            read_only_evaluation=True,
            parameter_updates_performed=False,
            trainable_parameter_count=0,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_labels_used_for_evaluation_only=False),
        source_highres_margin_audit=dict(
            checkpoint='/tmp/labeller_epoch_03_source_only.pth',
            checkpoint_epoch=3, margins=[0.2, 0.225, 0.25],
            shared_model_forward=True, shared_model_forward_count=738,
            margin_decision_count=2214, formal_gate_passed=True,
            selected_margin=0.225, target_read=False,
            results=[
                dict(promotion_margin=0.2), selected,
                dict(promotion_margin=0.25,
                     epoch3_reference_reproduced=True)]),
        source_selected_checkpoint=(
            '/tmp/labeller_epoch_03_source_only.pth'),
        selected_promotion_margin=0.225,
        target_dev=None)


def _candidate_payload():
    return dict(
        source_only=True, frozen_dinov2=True, epoch=3,
        s7_inference_enabled=True,
        s7_architecture=dict(
            enabled=True, protected_merge=True,
            highres_roi_ranker=True, highres_channels=32,
            highres_hidden=32, highres_max_candidates=32,
            highres_score_weight=1.0, highres_promotion_margin=0.25),
        training_protocol=dict(
            train_components='s7_highres_roi_ranker',
            s7_highres_roi_ranker=dict(
                frozen_detector=True, source_only=True, target_read=False,
                inference_slice_routing=False,
                sequence_identity_feature=False,
                additional_dino_forward=False,
                dense_feature_history=False,
                highres_channels=32, hidden=32, max_candidates=32,
                score_weight=1.0, promotion_margin=0.25)))


def test_strict_source_margin_gate_accepts_only_protocol24_selection():
    gate = audit.strict_source_margin_gate(_source_margin_result())
    assert gate['passed'] is True
    assert gate['runtime_margin'] == 0.225
    assert gate['checkpoint_architecture_margin'] == 0.25


def test_strict_source_margin_gate_rejects_target_read_or_changed_margin():
    result = deepcopy(_source_margin_result())
    result['target_dev'] = {'read': True}
    result['selected_promotion_margin'] = 0.2
    gate = audit.strict_source_margin_gate(result)
    assert gate['passed'] is False
    assert gate['checks']['source_only_target_unread'] is False
    assert gate['checks']['selected_margin'] is False


def test_locked_model_separates_checkpoint_and_runtime_margins():
    args = Namespace()
    audit.configure_locked_model(args)
    assert args.train_components == 's7_highres_roi_ranker'
    assert args.s7_highres_promotion_margin == 0.25
    assert args.runtime_highres_promotion_margin == 0.225
    assert args.s7_highres_max_candidates == 32
    assert args.s7_temporal_association is False
    assert args.s7_selective_promotion is False


def test_candidate_checkpoint_gate_does_not_require_rejected_best_epoch():
    source_gate = audit.strict_source_margin_gate(_source_margin_result())
    payload = _candidate_payload()
    payload['best_epoch'] = 0
    gate = audit.candidate_checkpoint_gate(payload, source_gate)
    assert gate['passed'] is True
    assert gate['checks']['checkpoint_epoch'] is True


def test_candidate_checkpoint_gate_rejects_target_routing():
    source_gate = audit.strict_source_margin_gate(_source_margin_result())
    payload = _candidate_payload()
    payload['training_protocol']['s7_highres_roi_ranker'][
        'inference_slice_routing'] = True
    gate = audit.candidate_checkpoint_gate(payload, source_gate)
    assert gate['passed'] is False
    assert gate['checks']['highres_training_protocol'] is False


def test_target_gates_lock_all_three_slices_and_small_coverage():
    assert audit.TARGET_GATES['seq02_far']['min_candidate_top1'] == 39
    assert audit.TARGET_GATES['seq02_dark']['min_candidate_top1'] == 29
    small = audit.TARGET_GATES['seq03_small']
    assert small['min_candidate_top1'] == 51
    assert small['min_candidate_recall_at_100'] == 64
    assert small['max_candidate_mcml'] == 6


def test_native_lane_reproduction_checks_top1_geometry():
    baseline = [dict(
        split='test', seq='real_seq03', frame=129,
        detections=[[1.0, 2.0, 3.0, 4.0, 0.1, 0.9]])]
    same = deepcopy(baseline)
    reproduced = audit.native_lane_reproduction(baseline, same)
    assert reproduced['exact_native_top1'] is True
    changed = deepcopy(same)
    changed[0]['detections'][0][0] = 2.0
    rejected = audit.native_lane_reproduction(baseline, changed)
    assert rejected['exact_native_top1'] is False

