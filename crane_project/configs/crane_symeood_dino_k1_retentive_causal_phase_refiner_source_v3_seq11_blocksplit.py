"""E1 V2: V3 plus leakage-safe real_seq11 auxiliary training frames.

The 59-frame pilot is split before training.  Forty-eight frames are used for
auxiliary source supervision and the complete final temporal block (including
the adjacent uniform sample at frame 6710) is held out as an 11-frame
same-video mechanism validation set.  Fixed TEST is absent.
"""

_base_ = [
    './crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v3.py']

history_horizon = 4
source_train_audit = (
    'work_dirs/crane_symeood_dino_distill_support_v1/source_collect/'
    'source_train_all_lane_audit.json')
aux_train_split = 'extra_source_real_seq11_pilot_k1p9_train_v1'
aux_val_split = 'extra_source_real_seq11_pilot_k1p9_val_v1'
aux_train_audit = (
    'work_dirs/crane_symeood_dino_source_inventory_v1/'
    'real_seq11_pilot_k1p9/blocksplit_v1/train_all_lane_audit.json')
aux_val_audit = (
    'work_dirs/crane_symeood_dino_source_inventory_v1/'
    'real_seq11_pilot_k1p9/blocksplit_v1/val_all_lane_audit.json')
aux_split_manifest = (
    'crane_project/data_contracts/'
    'real_seq11_pilot_k1p9_blocksplit_v1.json')
aux_split_manifest_sha256 = (
    '2f827e0b23b41a93394e063178caa0fc23f51a104b934f4f48835b1fe728e99a')
normalization = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)


def _train_pipeline(audit_json, expected_count, expected_split):
    return [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='LoadDinoProposalFromAudit', audit_json=audit_json,
            expected_frame_count=expected_count,
            expected_split=expected_split),
        dict(
            type='LoadCausalHistoryFromAudit', audit_json=audit_json,
            history_horizon=history_horizon,
            expected_frame_count=expected_count,
            expected_split=expected_split),
        dict(type='RResize', img_scale=(1024, 1024)),
        dict(type='SetNoFlipMetadata'),
        dict(
            type='RandomBrightnessContrast',
            brightness_range=(0.4, 1.0), contrast_range=(1.0, 1.0),
            noise_std_range=(0, 0), prob=0.5),
        dict(type='Normalize', **normalization),
        dict(type='Pad', size=(1024, 1024),
             pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(type='PrepareCausalHistoryInputs', **normalization),
        dict(
            type='CausalHistoryProposalAugment',
            current_probability=0.5, history_probability=0.35,
            history_dropout_probability=0.25, center_fraction=0.20,
            log_size=0.30, angle_deg=12.0),
        dict(type='DefaultFormatBundle'),
        dict(type='FormatDinoProposal'),
        dict(type='FormatCausalHistoryInputs'),
        dict(type='Collect', keys=[
            'img', 'gt_bboxes', 'gt_labels', 'dino_proposals',
            'causal_history_images', 'causal_history_proposals',
            'causal_history_valid_mask', 'causal_history_ages'])]


main_train_pipeline = _train_pipeline(
    source_train_audit, 2781, 'source-train')
aux_train_pipeline = _train_pipeline(
    aux_train_audit, 48, aux_train_split)

data = dict(
    train=[
        dict(
            type='CraneDataset', data_root='crane_project/data/crane_grab/',
            ann_file='train/annfiles/', img_prefix='train/images/',
            pipeline=main_train_pipeline, version='le90'),
        dict(
            type='CraneDataset', data_root='crane_project/data/crane_grab/',
            ann_file='train_sim/annfiles/', img_prefix='train/images/',
            pipeline=main_train_pipeline, version='le90'),
        dict(
            type='CraneDataset', data_root='crane_project/data/crane_grab/',
            ann_file=aux_train_split + '/annfiles/',
            img_prefix=aux_train_split + '/images/',
            pipeline=aux_train_pipeline, version='le90')],
    train_dataloader=dict(
        samples_per_gpu=2, workers_per_gpu=2, shuffle=False))

evidence_contract = dict(
    source_train_frames=2829,
    original_source_train_frames=2781,
    auxiliary_source_frames=59,
    auxiliary_source_train_frames=48,
    auxiliary_source_val_frames=11,
    auxiliary_source_sequence='real_seq11',
    auxiliary_split_manifest=aux_split_manifest,
    auxiliary_split_manifest_sha256=aux_split_manifest_sha256,
    auxiliary_train_val_overlap=0,
    auxiliary_validation_temporal_metrics=False,
    auxiliary_source_independent_sequence_claim=False,
    auxiliary_source_router_claim=False,
    auxiliary_source_sparse_history=True,
    appledouble_sidecars_are_samples=False)

model = dict(
    evidence_contract=evidence_contract,
    geometry_refiner_checkpoint=None,
    geometry_refiner_checkpoint_sha256=None,
    geometry_refiner_checkpoint_contract=None,
    evaluation_only=False)

checkpoint_config = dict(
    interval=1, max_keep_ckpts=10,
    meta=dict(
        geometry_refiner_checkpoint_contract=dict(
            protocol='source_only_k1_retentive_v3_seq11_blocksplit_e1_v2',
            architecture='k1_retentive_causal_phase_refiner_v3',
            frozen_baseline_variant='symeood_k1_epoch24',
            frozen_baseline_config='crane_project/configs/crane_symeood_k1.py',
            frozen_baseline_checkpoint='work_dirs/crane_symeood_k1/epoch_24.pth',
            source_train_frames=2829, original_source_train_frames=2781,
            auxiliary_source_frames=59,
            auxiliary_source_train_frames=48,
            auxiliary_source_val_frames=11,
            auxiliary_source_sequence='real_seq11',
            auxiliary_split_manifest=aux_split_manifest,
            auxiliary_split_manifest_sha256=aux_split_manifest_sha256,
            auxiliary_train_val_overlap=0,
            auxiliary_validation_temporal_metrics=False,
            source_val_frames=738,
            target_data_read=False, fixed_test_read=False,
            source_gate_passed=False,
            detector_forward_during_training=True,
            dino_detector_forward_during_training=False,
            frozen_symeood_feature_forward=True,
            frozen_symeood_detection_head_forward=True,
            frozen_symeood_detection_from_shared_features=True,
            cached_dino_proposals_only=True,
            domain_routing=False, sequence_frame_routing=False,
            temporal_state=False, causal_history_input=True,
            history_horizon=4, history_identity_model_input=False,
            current_k1_geometry_anchor=True,
            native_dino_anchor_fallback=True,
            native_dino_current_conditioning=True,
            same_forward_all_domains=True,
            bounded_current_residual=True,
            bounded_history_residual=True,
            rejectable_history_gate=True,
            exact_current_only_when_no_history=True,
            continuous_double_angle_phase=True,
            zero_phase_is_exact_identity=True,
            source_only_proposal_corruption=True,
            fixed_target_parameter_selection=False,
            representation='six_delta_xywh_sin2a_cos2a_residual',
            continuous_k1_retention=True, retention_loss_weight=0.25,
            source_adjacent_pair_supervision=True,
            adjacent_pair_identity_model_input=False,
            temporal_size_error_consistency=True,
            temporal_size_loss_weight=0.20,
            inference_sequence_input=False,
            single_gpu_adjacent_pair_training=True,
            train_samples_per_gpu=2, train_shuffle=False,
            auxiliary_source_independent_sequence_claim=False,
            auxiliary_source_router_claim=False,
            auxiliary_source_sparse_history=True,
            appledouble_sidecars_are_samples=False,
            angle_range='le90', edge_swap=True, proj_xy=True,
            refine_center=True, refine_size=True, refine_angle=True)))

seed = 3407
gpu_ids = [0]
load_from = None
resume_from = None
work_dir = (
    'work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_'
    'source_v3_seq11_blocksplit_e1_v2_seed3407')

