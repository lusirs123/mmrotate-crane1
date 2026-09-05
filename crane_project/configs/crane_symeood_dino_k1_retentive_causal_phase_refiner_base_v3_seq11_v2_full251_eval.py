"""Source-only Base-V3 epoch9 inference on all 251 seq11-v2 frames.

This is a frozen baseline diagnostic for the preregistered three-way
comparison.  It does not train, select an epoch, read target data, or read the
fixed TEST split.
"""

_base_ = [
    './crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v3.py']

data_root = 'crane_project/data/crane_grab/'
source_split = 'extra_source_real_seq11_pilot_k1p9_v2'
all_lane_audit = (
    'work_dirs/crane_symeood_dino_source_inventory_v2/'
    'real_seq11_k1p9_v2/all_lane_collect/'
    'source_inventory_all_lane_audit.json')
runtime_input_files = dict(dino_all_lane_audit=all_lane_audit)

expected_checkpoint = (
    'work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_'
    'source_v3_seed3407/k1_retentive_v3_epoch9_promoted.pth')
expected_checkpoint_protocol = (
    'source_gated_k1_retentive_causal_phase_refiner_v3')
expected_checkpoint_source_train_frames = 2781
expected_checkpoint_target_data_read = False
expected_checkpoint_fixed_test_read = False

normalization = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375], to_rgb=True)

full251_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDinoProposalFromAudit', audit_json=all_lane_audit,
         expected_frame_count=251, expected_split=source_split),
    dict(type='LoadCausalHistoryFromAudit', audit_json=all_lane_audit,
         history_horizon=4, expected_frame_count=251,
         expected_split=source_split),
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

full251_dataset = dict(
    type='CraneDataset', data_root=data_root,
    ann_file=source_split + '/annfiles/',
    img_prefix=source_split + '/images/',
    pipeline=full251_pipeline, version='le90')

data = dict(
    _delete_=True,
    val=full251_dataset,
    test=full251_dataset,
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

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
    geometry_refiner_checkpoint=None,
    geometry_refiner_checkpoint_sha256=None,
    geometry_refiner_checkpoint_contract=None,
    evaluation_only=True)

evaluation = dict(
    _delete_=True, metric='mAP', paper_temporal=False,
    thresh_sim=10.0, thresh_real=25.0)

source_only_result_contract = dict(
    protocol='base_v3_epoch9_seq11_v2_full251_inference_v1',
    expected_frame_count=251,
    checkpoint_role='base_v3_promoted_epoch9',
    source_training_frame_count=2781,
    source_split=source_split,
    prediction_coordinate_system='original_image_pixels',
    obb_convention='le90',
    annotation_k0=1.9,
    target_geometry='top_beam_only',
    optimizer_steps=0,
    target_data_read=False,
    fixed_test_read=False,
    eligible_for_epoch_selection=False,
    eligible_for_checkpoint_promotion=False)

load_from = None
resume_from = None
work_dir = (
    'work_dirs/crane_symeood_dino_source_inventory_v2/'
    'real_seq11_k1p9_v2/base_v3_epoch9_full251')
