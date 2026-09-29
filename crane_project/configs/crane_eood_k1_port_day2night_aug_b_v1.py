"""EOOD control with exactly the same training scale augmentation as SymEOOD B.
Only train pipelines, module import and work_dir differ from EOOD A.
"""
_base_ = ['./crane_eood_k1_port_day2night_v1.py']

custom_imports = {'imports': ['mmrotate.datasets.crane_custom_dota',
             'mmrotate.core.hooks.set_epoch_info_hook',
             'mmrotate.datasets.pipelines.port_train_augment'],
 'allow_failed_imports': False}

data = {'train': [{'type': 'CraneDataset',
            'data_root': 'crane_project/data/crane_grab_port_day2night_v1/',
            'ann_file': 'train/annfiles/',
            'img_prefix': 'train/images/',
            'pipeline': [{'type': 'LoadImageFromFile'},
                         {'type': 'LoadAnnotations', 'with_bbox': True},
                         {'type': 'RResize', 'img_scale': (1024, 1024)},
                         {'type': 'PortIsotropicShrink',
                          'prob': 0.5,
                          'scale_range': (0.5, 1.0)},
                         {'type': 'RRandomFlip',
                          'flip_ratio': [0.25, 0.25, 0.25],
                          'direction': ['horizontal', 'vertical', 'diagonal'],
                          'version': 'le90'},
                         {'type': 'Normalize',
                          'mean': [123.675, 116.28, 103.53],
                          'std': [58.395, 57.12, 57.375],
                          'to_rgb': True},
                         {'type': 'Pad',
                          'size': (1024, 1024),
                          'pad_val': {'img': (114.0, 114.0, 114.0)}},
                         {'type': 'DefaultFormatBundle'},
                         {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels']}],
            'version': 'le90'},
           {'type': 'CraneDataset',
            'data_root': 'crane_project/data/crane_grab_port_day2night_v1/',
            'ann_file': 'train_sim/annfiles/',
            'img_prefix': 'train_sim/images/',
            'pipeline': [{'type': 'LoadImageFromFile'},
                         {'type': 'LoadAnnotations', 'with_bbox': True},
                         {'type': 'RResize', 'img_scale': (1024, 1024)},
                         {'type': 'PortIsotropicShrink',
                          'prob': 0.5,
                          'scale_range': (0.5, 1.0)},
                         {'type': 'RRandomFlip',
                          'flip_ratio': [0.25, 0.25, 0.25],
                          'direction': ['horizontal', 'vertical', 'diagonal'],
                          'version': 'le90'},
                         {'type': 'Normalize',
                          'mean': [123.675, 116.28, 103.53],
                          'std': [58.395, 57.12, 57.375],
                          'to_rgb': True},
                         {'type': 'Pad',
                          'size': (1024, 1024),
                          'pad_val': {'img': (114.0, 114.0, 114.0)}},
                         {'type': 'DefaultFormatBundle'},
                         {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels']}],
            'version': 'le90'}]}

work_dir = 'work_dirs/crane_eood_k1_port_day2night_aug_b_v1'

# Keep epoch 16/18 for the predeclared offline sweep after 24 epochs.
checkpoint_config = dict(max_keep_ckpts=24)
