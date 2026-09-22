"""Source-only DINO-to-SymEOOD semantic feature distillation.

DINO is read from an offline feature cache during training.  Validation,
fixed TEST, and deployment execute only the unchanged SymEOOD student.
"""

_base_ = ['./crane_symeood_k1.py']

angle_version = 'le90'
dino_feature_cache = (
    'work_dirs/dino_teacher_scoped_lowlight_v1_formal8/feature_cache')

model = dict(
    semantic_distillation=dict(
        enabled=True,
        mode='feature',
        # Restrict transfer to GT object regions.  The OBB regression target,
        # SymKLD loss, angle convention, and inference heads stay unchanged.
        scope='foreground',
        protect_geometry=True,
        student_channels=256,
        teacher_channels=1024,
        feature_level=0,
        loss_weight=0.05,
        max_tokens=4096))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip',
         flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'],
         version=angle_version),
    # Load after geometric transforms so the cached patch grid receives the
    # exact same flip.  Fixed 64x64 half precision bounds host/GPU memory.
    dict(type='LoadDinoFeatureFromCache',
         cache_dir=dino_feature_cache,
         output_size=(64, 64),
         expected_channels=1024,
         expected_model='dinov2_vitl14',
         strict_image_identity=True),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Pad',
         size=(1024, 1024),
         pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'teacher_features']),
]

dataset_type = 'CraneDataset'
data_root = 'crane_project/data/crane_grab/'
data = dict(
    # Start with one sample per GPU.  Offline 64x64 FP16 teacher features add
    # about 8 MiB/sample before activations and avoid any live DINO graph.
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=[
        dict(type=dataset_type, data_root=data_root,
             ann_file='train/annfiles/', img_prefix='train/images/',
             pipeline=train_pipeline, version=angle_version),
        dict(type=dataset_type, data_root=data_root,
             ann_file='train_sim/annfiles/', img_prefix='train/images/',
             pipeline=train_pipeline, version=angle_version),
    ])

work_dir = 'work_dirs/crane_symeood_k1_dino_semantic_distill_v1'
