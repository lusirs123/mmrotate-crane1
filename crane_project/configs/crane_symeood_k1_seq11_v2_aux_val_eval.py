"""Read-only ordinary-K1 reference on the seq11-v2 48-frame aux block."""

import json
import os


_base_ = ['./crane_symeood_k1.py']

data_root = 'crane_project/data/crane_grab/'
split_report_path = (
    'work_dirs/crane_symeood_dino_source_inventory_v2/'
    'real_seq11_k1p9_v2/blocksplit_v2/audited_split_materialization.json')


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


with open(split_report_path, 'r', encoding='utf-8') as _handle:
    split_report = json.load(_handle)
if (split_report.get('decision') !=
        'ALLOW_SEQ11_BLOCKSPLIT_SOURCE_TRAINING'
        or split_report.get('aux_val_frame_count') != 48
        or split_report.get('train_val_overlap_count') != 0):
    raise RuntimeError('The audited seq11-v2 split does not authorize aux-val')
aux_val_split = _data_root_child(split_report.get('val_split', ''))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug', img_scale=(1024, 1024), flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size=(1024, 1024),
                 pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])])]
aux_val_dataset = dict(
    type='CraneDataset', data_root=data_root,
    ann_file=aux_val_split + '/annfiles/',
    img_prefix=aux_val_split + '/images/',
    pipeline=test_pipeline, version='le90')

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

