"""Evaluation-only V3 epoch-9 component attribution on source-val.

Use ``--cfg-options model.geometry_refiner.inference_component_mode=<mode>``
with one of: full, current_only, center_only, k1_identity.  All modes load the
same immutable promoted checkpoint and share one decode implementation.
"""

import json
import os


_base_ = [
    './crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v3.py']

promotion_report_path = (
    'work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_'
    'source_v3_seed3407/epoch9_source_promotion.json')
with open(promotion_report_path, 'r', encoding='utf-8') as _handle:
    promotion_report = json.load(_handle)
del _handle
if promotion_report.get('decision') != (
        'ALLOW_K1_RETENTIVE_CAUSAL_PHASE_FIXED_BENCHMARK_TEST'):
    raise RuntimeError('V3 promotion report is invalid')
if promotion_report.get('eligible_for_fixed_benchmark_test') is not True:
    raise RuntimeError('V3 epoch 9 was not source promoted')
if (promotion_report.get('target_data_read') is not False
        or promotion_report.get('fixed_test_read') is not False):
    raise RuntimeError('V3 promotion provenance includes target data')
selection = dict(promotion_report.get('selection') or {})
selected = dict(selection.get('selected') or {})
if selected.get('epoch') != 9:
    raise RuntimeError('Component audit requires source-selected epoch 9')

promoted = dict(promotion_report.get('output') or {})
promoted_checkpoint = os.fspath(promoted['checkpoint'])
promoted_checkpoint_sha256 = str(promoted['checkpoint_sha256'])

geometry_refiner = dict(
    zero_init_output=False,
    inference_component_mode='full',
    center_loss_weight=0.0,
    size_loss_weight=0.0,
    angle_loss_weight=0.0,
    decoded_geometry_loss_weight=0.0,
    temporal_size_loss_weight=0.0,
    retention_loss_weight=0.0)

model = dict(
    geometry_refiner=geometry_refiner,
    geometry_refiner_checkpoint=promoted_checkpoint,
    geometry_refiner_checkpoint_sha256=promoted_checkpoint_sha256,
    geometry_refiner_checkpoint_contract=dict(
        _delete_=True,
        protocol='source_gated_k1_retentive_causal_phase_refiner_v3',
        source_gate_passed=True,
        selected_source_epoch=9,
        selection_policy=(
            'passing_only_min_mcml_max_riou_min_dfr_earliest_v1'),
        fixed_benchmark_test=True,
        test_used_for_model_selection=False,
        parameter_update_after_test=False,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False),
    evaluation_only=True)

evaluation = dict(
    _delete_=True, metric='mAP', thresh_sim=10.0, thresh_real=25.0,
    weight_sim=0.7, weight_real=0.3, paper_temporal=True,
    temporal_center_thresh_px=15.0, temporal_ekf_window=10,
    temporal_mcml_limit=5, temporal_iou_thresh=0.5)
load_from = None
resume_from = None
work_dir = (
    'work_dirs/crane_symeood_dino_k1_retentive_component_source_val')
