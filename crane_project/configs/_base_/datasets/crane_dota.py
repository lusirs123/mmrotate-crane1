# configs/_base_/datasetes/crane_dota.py
# 港口抓斗数据集基础配置
# 设计原则：一份配置，baseline 和自研模型共用
#
# 继承方式：
#   crane_baseline.py 和 crane_symeood.py 均在 _base_ 里引用本文件
#   _base_ = ['../_base_/datasetes/crane_dota.py', ...]

# ─────────────────────────────────────────────────────────
# 数据集基本信息
# ─────────────────────────────────────────────────────────
dataset_type = 'DOTADataset'
# data_root    = 'data/crane_grab/'
# 绝对路径，之后要根据服务器路径进行修改
data_root = '/root/mmrotate/crane_project/data/crane_grab/' # 结尾建议加斜杠

metainfo = dict(
    classes=('grab',),
    palette=[(255, 128, 0)],
)

# ─────────────────────────────────────────────────────────
# Pipeline 定义
# ─────────────────────────────────────────────────────────
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(704, 576), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical', 'diagonal'],
    ),
    dict(
        type='mmdet.RandomPhotometricDistort',
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10,
        prob=0.5,
    ),
    dict(type='mmdet.PackDetInputs'),
]

# val 和 test 共用同一个 pipeline
# 关键：meta_keys 必须包含 img_path
# CraneOBBMetric 从 img_path 的文件名解析 domain/seq_id/frame_id
# baseline 的 DOTAMetric 也会正常使用这些 meta_keys，不受影响
val_test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(704, 576), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',        # ← CraneOBBMetric 时序解析的唯一依赖
            'ori_shape',
            'img_shape',
            'scale_factor',
            'plc_rope_length', # ← DEP 指标可选字段，无此字段时自动跳过
        ),
    ),
]

# ─────────────────────────────────────────────────────────
# DataLoader 定义
# ─────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        img_suffix='jpg',
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,       # 时序评估必须为 1，baseline 的 DOTAMetric 不受影响
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        img_suffix='jpg',
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=val_test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        img_suffix='jpg',
        ann_file='test/annfiles/',
        data_prefix=dict(img_path='test/images/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=val_test_pipeline,
    ),
)

# ─────────────────────────────────────────────────────────
# 评估器（所有对比模型共用）
# ─────────────────────────────────────────────────────────
custom_imports = dict(
    imports=['mmrotate_crane.crane_metrics'],
    allow_failed_imports=False,
)

val_evaluator = [
    # 学术对齐指标：所有模型都报告 mAP
    dict(
        type='DOTAMetric',
        metric='mAP',
        iou_thrs=[0.5],
    ),
    # 场景专项指标：所有对比模型都用同一套
    dict(
        type='CraneOBBMetric',
        mode='val',
        center_thresh_px=15.0,
        angle_limit_deg=35.0,
        iou_thresh=0.5,
        collect_device='cpu',
        prefix='crane',
    ),
]

test_evaluator = [
    dict(
        type='DOTAMetric',
        metric='mAP',
        iou_thrs=[0.5, 0.75],
    ),
    dict(
        type='CraneOBBMetric',
        mode='test',
        center_thresh_px=15.0,
        ekf_window=10,
        mcml_limit=5,
        angle_limit_deg=35.0,
        depth_k=1000.0,
        depth_alpha=-1.5,
        iou_thresh=0.5,
        collect_device='cpu',
        prefix='crane',
    ),
]