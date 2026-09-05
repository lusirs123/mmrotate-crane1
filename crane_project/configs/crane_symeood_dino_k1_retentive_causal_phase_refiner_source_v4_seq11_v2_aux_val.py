"""Read-only V4 evaluation on the held-out 48-frame seq11-v2 block."""

import json
import os


_base_ = [
    './crane_symeood_dino_k1_retentive_causal_phase_refiner_'
    'source_v4_seq11_v2_replay.py']

data_root = 'crane_project/data/crane_grab/'
split_report_path = (
    'work_dirs/crane_symeood_dino_source_inventory_v2/'
    'real_seq11_k1p9_v2/blocksplit_v2/audited_split_materialization.json')
aux_val_audit = (
    'work_dirs/crane_symeood_dino_source_inventory_v2/'
    'real_seq11_k1p9_v2/blocksplit_v2/aux_val_all_lane_audit.json')


def _data_root_child(value):
    raw = os.fspath(value).strip()
    root = os.path.abspath(os.path.normpath(data_root))
    if os.path.isabs(raw):
        candidate = os.path.abspath(os.path.normpath(raw))
        if os.path.commonpath([root, candidate]) != root:
            raise RuntimeError('Aux-val split is outside data root')
        relative = os.path.relpath(candidate, root)
    else:
        normalized = os.path.normpath(raw)
        normalized_root = os.path.normpath(data_root)
        relative = (os.path.relpath(normalized, normalized_root)
                    if normalized.startswith(normalized_root + os.sep)
                    else normalized)
    parts = [part for part in relative.split(os.sep) if part]
    if (len(parts) != 1 or parts[0] in {'.', '..', 'train', 'train_sim',
                                       'val', 'test'}):
        raise RuntimeError('Aux-val split is not a safe data-root child')
    return parts[0]


def _read_json(path):
    # A module-level file handle is not deepcopy/pickle-safe in MMCV 1.x.
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


split_report = _read_json(split_report_path)
if (split_report.get('decision') !=
        'ALLOW_SEQ11_BLOCKSPLIT_SOURCE_TRAINING'
        or split_report.get('aux_val_frame_count') != 48
        or split_report.get('train_val_overlap_count') != 0
        or split_report.get('filtered_audits_written') is not True):
    raise RuntimeError('The audited seq11-v2 split does not authorize aux-val')
aux_val_split = _data_root_child(split_report.get('val_split', ''))

normalization = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375], to_rgb=True)
aux_val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDinoProposalFromAudit', audit_json=aux_val_audit,
         expected_frame_count=48, expected_split=aux_val_split),
    dict(type='LoadCausalHistoryFromAudit', audit_json=aux_val_audit,
         history_horizon=4, expected_frame_count=48,
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
    type='CraneDataset', data_root=data_root,
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
