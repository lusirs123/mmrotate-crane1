"""Source-only V4: Base-V3 retention plus controlled seq11-v2 replay.

The student starts from the immutable source-promoted V3 epoch 9.  A frozen
copy of that refiner is the output teacher on the original 2781 source frames.
The 203 seq11 auxiliary-train frames enter through deterministic pair batches;
the 48-frame same-video auxiliary validation view and fixed TEST are excluded.
"""

import hashlib
import json
import os


_base_ = [
    './crane_symeood_dino_k1_retentive_causal_phase_refiner_source_v3.py']


def _read_json_contract(path, decision):
    if not os.path.isfile(path):
        raise RuntimeError('Required source audit is missing: ' + path)
    with open(path, 'rb') as handle:
        raw = handle.read()
    payload = json.loads(raw.decode('utf-8'))
    if payload.get('decision') != decision:
        raise RuntimeError(
            'Required source audit did not pass: {} ({!r})'.format(
                path, payload.get('decision')))
    return payload, hashlib.sha256(raw).hexdigest()


history_horizon = 4
data_root = 'crane_project/data/crane_grab/'
original_train_audit = (
    'work_dirs/crane_symeood_dino_distill_support_v1/source_collect/'
    'source_train_all_lane_audit.json')
official_source_val_audit = (
    'work_dirs/crane_symeood_dino_conservative_takeover_v2/'
    'source_calibration_collect/source_val_fusion_source_audit.json')
aux_root = (
    'work_dirs/crane_symeood_dino_source_inventory_v2/'
    'real_seq11_k1p9_v2')
all_lane_audit = aux_root + '/all_lane_collect/source_inventory_all_lane_audit.json'
blocksplit_root = aux_root + '/blocksplit_v2'
aux_train_audit = blocksplit_root + '/train_all_lane_audit.json'
aux_val_audit = blocksplit_root + '/aux_val_all_lane_audit.json'
split_report_path = blocksplit_root + '/audited_split_materialization.json'
full_contract_path = aux_root + '/full_source_contract.json'
train_contract_path = blocksplit_root + '/aux_train_contract.json'
val_contract_path = blocksplit_root + '/aux_val_contract.json'
split_manifest = (
    data_root + 'extra_source_real_seq11_pilot_k1p9_v2/split_manifest.json')


def _safe_data_root_child(value, role):
    """Normalize one materialized split to a safe data-root child name.

    The splitter accepts a leaf name, a repository-relative path, or an
    absolute path.  Its report intentionally preserves exactly what the user
    passed, so requiring a particular textual prefix rejects otherwise valid
    materializations.  Training only needs the canonical child name, while
    the audit hashes and frame-count contracts below establish its identity.
    """
    raw = os.fspath(value).strip()
    if not raw:
        raise RuntimeError('seq11-v2 {} split name is empty'.format(role))
    root = os.path.abspath(os.path.normpath(data_root))
    if os.path.isabs(raw):
        candidate = os.path.abspath(os.path.normpath(raw))
        try:
            inside_root = os.path.commonpath([root, candidate]) == root
        except ValueError:
            inside_root = False
        if not inside_root:
            raise RuntimeError(
                'seq11-v2 {} split is outside data root: {!r}'.format(
                    role, raw))
        relative = os.path.relpath(candidate, root)
    else:
        normalized = os.path.normpath(raw)
        normalized_root = os.path.normpath(data_root)
        if normalized.startswith(normalized_root + os.sep):
            relative = os.path.relpath(normalized, normalized_root)
        else:
            relative = normalized
    parts = [part for part in relative.split(os.sep) if part]
    if (len(parts) != 1 or parts[0] in {'.', '..'}
            or parts[0].startswith('._')):
        raise RuntimeError(
            'seq11-v2 {} split must be one direct data-root child: {!r}'
            .format(role, raw))
    child = parts[0]
    if child in {'train', 'train_sim', 'val', 'test'}:
        raise RuntimeError(
            'seq11-v2 {} split may not reuse a canonical dataset split: {!r}'
            .format(role, child))
    return child


split_report, split_report_sha256 = _read_json_contract(
    split_report_path, 'ALLOW_SEQ11_BLOCKSPLIT_SOURCE_TRAINING')
aux_source_split = _safe_data_root_child(
    split_report.get('source_split', ''), 'source')
aux_train_split = _safe_data_root_child(
    split_report.get('train_split', ''), 'train')
aux_val_split = _safe_data_root_child(
    split_report.get('val_split', ''), 'validation')
if (aux_source_split != 'extra_source_real_seq11_pilot_k1p9_v2'
        or len({aux_source_split, aux_train_split, aux_val_split}) != 3):
    raise RuntimeError(
        'seq11-v2 source/train/validation split identities are unsafe: '
        'source={!r}, train={!r}, validation={!r}'.format(
            aux_source_split, aux_train_split, aux_val_split))
full_contract, full_contract_sha256 = _read_json_contract(
    full_contract_path, 'ALLOW_AUXILIARY_SOURCE_TRAINING_INPUT')
train_contract, train_contract_sha256 = _read_json_contract(
    train_contract_path, 'ALLOW_AUXILIARY_SOURCE_TRAINING_INPUT')
val_contract, val_contract_sha256 = _read_json_contract(
    val_contract_path, 'ALLOW_AUXILIARY_SOURCE_TRAINING_INPUT')
if (split_report.get('all_frame_count') != 251
        or split_report.get('aux_train_frame_count') != 203
        or split_report.get('aux_val_frame_count') != 48
        or split_report.get('train_val_overlap_count') != 0):
    raise RuntimeError('seq11-v2 audited split counts are invalid')
if (full_contract.get('visible_image_count') != 251
        or train_contract.get('visible_image_count') != 203
        or val_contract.get('visible_image_count') != 48):
    raise RuntimeError('seq11-v2 auxiliary contract counts are invalid')
if (split_report.get('filtered_audits_written') is not True
        or split_report.get(
            'eligible_for_auxiliary_blocksplit_training') is not True
        or full_contract.get('audit_sha256') !=
        split_report.get('input_audit_sha256')
        or train_contract.get('audit_sha256') !=
        split_report.get('train_audit_sha256')
        or val_contract.get('audit_sha256') !=
        split_report.get('val_audit_sha256')):
    raise RuntimeError('seq11-v2 audit provenance chain is inconsistent')
if (_safe_data_root_child(
        train_contract.get('source_split', ''), 'train contract') !=
        aux_train_split
        or _safe_data_root_child(
            val_contract.get('source_split', ''), 'validation contract') !=
        aux_val_split):
    raise RuntimeError('seq11-v2 auxiliary contract split names are invalid')

promotion_report_path = (
    'work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_'
    'source_v3_seed3407/epoch9_source_promotion.json')
promotion_report, promotion_report_sha256 = _read_json_contract(
    promotion_report_path,
    'ALLOW_K1_RETENTIVE_CAUSAL_PHASE_FIXED_BENCHMARK_TEST')
if (promotion_report.get('target_data_read') is not False
        or promotion_report.get('fixed_test_read') is not False):
    raise RuntimeError('Base-V3 teacher promotion contains target evidence')
base_v3_output = dict(promotion_report.get('output') or {})
base_v3_checkpoint = os.fspath(base_v3_output['checkpoint'])
base_v3_checkpoint_sha256 = str(base_v3_output['checkpoint_sha256'])

normalization = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375], to_rgb=True)


def _train_pipeline(audit_json, expected_count, expected_split):
    return [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='LoadDinoProposalFromAudit', audit_json=audit_json,
             expected_frame_count=expected_count,
             expected_split=expected_split),
        dict(type='LoadCausalHistoryFromAudit', audit_json=audit_json,
             history_horizon=history_horizon,
             expected_frame_count=expected_count,
             expected_split=expected_split),
        dict(type='RResize', img_scale=(1024, 1024)),
        dict(type='SetNoFlipMetadata'),
        dict(type='RandomBrightnessContrast',
             brightness_range=(0.4, 1.0), contrast_range=(1.0, 1.0),
             noise_std_range=(0, 0), prob=0.5),
        dict(type='Normalize', **normalization),
        dict(type='Pad', size=(1024, 1024),
             pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(type='PrepareCausalHistoryInputs', **normalization),
        dict(type='CausalHistoryProposalAugment',
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


original_pipeline = _train_pipeline(
    original_train_audit, 2781, 'source-train')
aux_pipeline = _train_pipeline(aux_train_audit, 203, aux_train_split)

source_val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDinoProposalFromAudit',
         audit_json=official_source_val_audit,
         expected_frame_count=738, expected_split='val'),
    dict(type='LoadCausalHistoryFromAudit',
         audit_json=official_source_val_audit,
         history_horizon=history_horizon,
         expected_frame_count=738, expected_split='val'),
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
source_val_dataset = dict(
    type='CraneDataset', data_root=data_root,
    ann_file='val/annfiles/', img_prefix='val/images/',
    pipeline=source_val_pipeline, version='le90')

geometry_refiner = dict(
    zero_init_output=False,
    inference_component_mode='full')

evidence_contract = dict(
    source_train_frames=2984,
    original_source_train_frames=2781,
    auxiliary_source_all_frames=251,
    auxiliary_source_train_frames=203,
    auxiliary_source_val_frames=48,
    auxiliary_source_sequence='real_seq11',
    auxiliary_train_val_overlap=0,
    auxiliary_split_manifest_sha256=split_report['split_manifest_sha256'],
    auxiliary_annotation_k0=1.9,
    auxiliary_target_geometry='top_beam_only',
    auxiliary_source_independent_sequence_claim=False,
    auxiliary_source_router_claim=False,
    auxiliary_validation_role='same_video_mechanism_only',
    appledouble_sidecars_are_samples=False,
    target_data_read=False,
    fixed_test_read=False,
    base_v3_teacher_retention=True,
    original_source_replay=True,
    fixed_auxiliary_sampling_ratio=True,
    fixed_optimizer_steps=True,
    auxiliary_adjacent_pair_supervision=True)

model = dict(
    geometry_refiner=geometry_refiner,
    evidence_contract=evidence_contract,
    geometry_refiner_checkpoint=base_v3_checkpoint,
    geometry_refiner_checkpoint_sha256=base_v3_checkpoint_sha256,
    geometry_refiner_checkpoint_contract=dict(
        _delete_=True,
        protocol='source_gated_k1_retentive_causal_phase_refiner_v3',
        architecture='k1_retentive_causal_phase_refiner_v3',
        source_train_frames=2781,
        source_val_frames=738,
        target_data_read=False,
        fixed_test_read=False,
        source_gate_passed=True,
        selected_source_epoch=9,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False),
    base_teacher_retention_loss_weight=0.25,
    evaluation_only=False)

optimizer_steps_per_epoch = 1391
training_epochs = 10
total_optimizer_steps = optimizer_steps_per_epoch * training_epochs
original_batches_per_auxiliary_batch = 14
data = dict(
    _delete_=True,
    train=dict(
        type='FixedRatioPairReplayDataset',
        original_dataset=[
            dict(type='CraneDataset', data_root=data_root,
                 ann_file='train/annfiles/', img_prefix='train/images/',
                 pipeline=original_pipeline, version='le90'),
            dict(type='CraneDataset', data_root=data_root,
                 ann_file='train_sim/annfiles/', img_prefix='train/images/',
                 pipeline=original_pipeline, version='le90')],
        auxiliary_dataset=dict(
            type='CraneDataset', data_root=data_root,
            ann_file=aux_train_split + '/annfiles/',
            img_prefix=aux_train_split + '/images/',
            pipeline=aux_pipeline, version='le90'),
        samples_per_batch=2,
        original_batches_per_auxiliary_batch=(
            original_batches_per_auxiliary_batch),
        optimizer_steps_per_epoch=optimizer_steps_per_epoch),
    val=source_val_dataset,
    test=source_val_dataset,
    train_dataloader=dict(
        samples_per_gpu=2, workers_per_gpu=2, shuffle=False),
    val_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

optimizer = dict(
    _delete_=True, type='AdamW',
    constructor='GeometryRefinerOptimizerConstructor',
    lr=5e-5, weight_decay=1e-4)
lr_config = dict(
    policy='step',
    warmup='linear', warmup_iters=200, warmup_ratio=0.1,
    step=[6, 9])
runner = dict(_delete_=True, type='EpochBasedRunner',
              max_epochs=training_epochs)
evaluation = dict(interval=1, by_epoch=True)

checkpoint_config = dict(
    interval=1, by_epoch=True, max_keep_ckpts=10,
    meta=dict(
        geometry_refiner_checkpoint_contract=dict(
            protocol='source_only_k1_retentive_v4_seq11_v2_replay',
            architecture='k1_retentive_causal_phase_refiner_v3',
            initialized_from_base_v3_source_promoted_epoch=9,
            base_v3_checkpoint_sha256=base_v3_checkpoint_sha256,
            base_v3_promotion_report_sha256=promotion_report_sha256,
            base_v3_teacher_retention=True,
            base_v3_teacher_retention_loss_weight=0.25,
            source_train_frames=2984,
            original_source_train_frames=2781,
            auxiliary_source_all_frames=251,
            auxiliary_source_train_frames=203,
            auxiliary_source_val_frames=48,
            source_val_frames=738,
            auxiliary_source_sequence='real_seq11',
            auxiliary_train_val_overlap=0,
            auxiliary_split_manifest_sha256=(
                split_report['split_manifest_sha256']),
            auxiliary_annotation_k0=1.9,
            auxiliary_target_geometry='top_beam_only',
            auxiliary_source_independent_sequence_claim=False,
            auxiliary_source_router_claim=False,
            auxiliary_validation_role='same_video_mechanism_only',
            appledouble_sidecars_are_samples=False,
            auxiliary_split_report_sha256=split_report_sha256,
            auxiliary_full_contract_sha256=full_contract_sha256,
            auxiliary_train_contract_sha256=train_contract_sha256,
            auxiliary_val_contract_sha256=val_contract_sha256,
            original_source_replay=True,
            original_batches_per_auxiliary_batch=(
                original_batches_per_auxiliary_batch),
            auxiliary_batches_per_cycle=1,
            fixed_optimizer_steps=True,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            training_epochs=training_epochs,
            total_optimizer_steps=total_optimizer_steps,
            checkpoint_interval_epochs=1,
            source_adjacent_pair_supervision=True,
            auxiliary_adjacent_pair_supervision=True,
            adjacent_pair_identity_model_input=False,
            target_data_read=False, fixed_test_read=False,
            source_gate_passed=False,
            detector_forward_during_training=True,
            dino_detector_forward_during_training=False,
            cached_dino_proposals_only=True,
            domain_routing=False, sequence_frame_routing=False,
            temporal_state=False, causal_history_input=True,
            history_horizon=4, history_identity_model_input=False,
            current_k1_geometry_anchor=True,
            native_dino_anchor_fallback=True,
            native_dino_current_conditioning=True,
            same_forward_all_domains=True,
            representation='six_delta_xywh_sin2a_cos2a_residual',
            angle_range='le90', edge_swap=True, proj_xy=True,
            refine_center=True, refine_size=True, refine_angle=True)))

custom_hooks = [
    dict(type='FixedRatioReplayEpochHook'),
    dict(type='GeometryRefinerContractHook'),
    dict(type='CudaPeakMemoryContractHook')]

seed = 3407
gpu_ids = [0]
load_from = None
resume_from = None
work_dir = (
    'work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_'
    'source_v4_seq11_v2_replay_seed3407')
