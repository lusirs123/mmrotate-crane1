"""Evaluate an E1 block-split checkpoint on 11 held-out seq11 frames only."""

_base_ = [
    './crane_symeood_dino_k1_retentive_causal_phase_refiner_'
    'source_v3_seq11_blocksplit.py']

aux_val_split = 'extra_source_real_seq11_pilot_k1p9_val_v1'
aux_val_audit = (
    'work_dirs/crane_symeood_dino_source_inventory_v1/'
    'real_seq11_pilot_k1p9/blocksplit_v1/val_all_lane_audit.json')
normalization = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375], to_rgb=True)

aux_val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDinoProposalFromAudit', audit_json=aux_val_audit,
         expected_frame_count=11, expected_split=aux_val_split),
    dict(type='LoadCausalHistoryFromAudit', audit_json=aux_val_audit,
         history_horizon=4, expected_frame_count=11,
         expected_split=aux_val_split),
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

aux_val_dataset = dict(
    type='CraneDataset', data_root='crane_project/data/crane_grab/',
    ann_file=aux_val_split + '/annfiles/',
    img_prefix=aux_val_split + '/images/',
    pipeline=aux_val_pipeline, version='le90')

data = dict(
    _delete_=True,
    val=aux_val_dataset,
    test=aux_val_dataset,
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

evaluation = dict(
    _delete_=True, metric='mAP', paper_temporal=False,
    thresh_sim=10.0, thresh_real=25.0)
load_from = None
resume_from = None

