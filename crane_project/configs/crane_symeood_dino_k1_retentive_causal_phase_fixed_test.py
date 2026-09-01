"""Fixed benchmark TEST for source-selected K1-retentive V3 epoch 9.

The same 992-frame TEST is used for comparable model reporting.  Source-val
alone selected the immutable checkpoint; TEST cannot select another epoch or
change model parameters.  Cached DINO/history records are reused without a
DINO forward or sequence/frame routing.
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
    raise RuntimeError('V3 promotion report does not authorize fixed TEST')
if promotion_report.get('eligible_for_fixed_benchmark_test') is not True:
    raise RuntimeError('V3 promotion is not fixed-benchmark eligible')
if (promotion_report.get('target_data_read') is not False
        or promotion_report.get('fixed_test_read') is not False):
    raise RuntimeError('V3 promotion provenance already includes TEST data')
selection = dict(promotion_report.get('selection') or {})
selected = dict(selection.get('selected') or {})
if (selected.get('epoch') != 9
        or selection.get('selection_policy') !=
        'passing_only_min_mcml_max_riou_min_dfr_earliest_v1'):
    raise RuntimeError('V3 promotion did not use locked source selection')

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
        promotion_before_fixed_test=True,
        fixed_benchmark_test=True,
        test_used_for_model_selection=False,
        parameter_update_after_test=False,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False),
    evaluation_only=True)

fixed_test_audit = (
    'work_dirs/crane_symeood_dino_conservative_takeover_v2/'
    'full_test_metric_v2/conservative_takeover_fusion_audit.json')
normalization = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375], to_rgb=True)
fixed_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDinoProposalFromAudit', audit_json=fixed_test_audit,
         expected_frame_count=992, expected_split='test'),
    dict(type='LoadCausalHistoryFromAudit', audit_json=fixed_test_audit,
         history_horizon=4, expected_frame_count=992,
         expected_split='test'),
    dict(type='MultiScaleFlipAug', img_scale=(1024, 1024), flip=False,
         transforms=[
             dict(type='RResize'),
             dict(type='Normalize', **normalization),
             dict(type='Pad', size=(1024, 1024),
                  pad_val=dict(img=(114.0, 114.0, 114.0))),
             dict(type='PrepareCausalHistoryInputs', **normalization),
             dict(type='DefaultFormatBundle'),
             dict(type='FormatDinoProposal'),
             dict(type='FormatCausalHistoryInputs'),
             dict(type='Collect', keys=[
                 'img', 'dino_proposals', 'causal_history_images',
                 'causal_history_proposals', 'causal_history_valid_mask',
                 'causal_history_ages'])])]

fixed_test_dataset = dict(
    type='CraneDataset', data_root='crane_project/data/crane_grab/',
    ann_file='test/annfiles/', img_prefix='test/images/',
    pipeline=fixed_test_pipeline, version='le90')
data = dict(
    _delete_=True,
    val=fixed_test_dataset,
    test=fixed_test_dataset,
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False))
evaluation = dict(
    _delete_=True, metric='mAP', thresh_sim=10.0, thresh_real=25.0,
    weight_sim=0.7, weight_real=0.3, paper_temporal=True,
    temporal_center_thresh_px=15.0, temporal_ekf_window=10,
    temporal_mcml_limit=5, temporal_iou_thresh=0.5)
load_from = None
resume_from = None
work_dir = (
    'work_dirs/crane_symeood_dino_k1_retentive_causal_phase_fixed_test')
