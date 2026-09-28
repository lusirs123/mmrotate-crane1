"""SymEOOD K=1 on the port dataset; override dataset paths only.

Launch with a separate --work-dir. Keep the base initialization (ImageNet
backbone, no project checkpoint) and all training/evaluation parameters.
MMCV replaces lists, so explicitly preserve the inherited train pipelines.
"""

_base_ = ['./crane_symeood_k1.py']

data_root = 'crane_project/data/crane_grab_port_day2night_v1/'
data = dict(
    train=[
        dict(
            type={{_base_.dataset_type}},
            data_root=data_root,
            ann_file='train/annfiles/',
            img_prefix='train/images/',
            pipeline={{_base_.train_pipeline}},
            version={{_base_.angle_version}}),
        dict(
            type={{_base_.dataset_type}},
            data_root=data_root,
            ann_file='train_sim/annfiles/',
            img_prefix='train_sim/images/',
            pipeline={{_base_.train_pipeline}},
            version={{_base_.angle_version}}),
    ],
    val=dict(data_root=data_root),
    test=dict(data_root=data_root),
)
