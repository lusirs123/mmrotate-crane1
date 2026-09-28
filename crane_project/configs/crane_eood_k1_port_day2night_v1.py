"""EOOD K=1 on the new dataset; only dataset paths differ from the base.

Use a separate --work-dir at launch: the base work_dir is intentionally
inherited to keep this configuration override limited to dataset paths.
MMCV replaces train lists wholesale, so retain each entry's inherited
type, pipeline and angle convention explicitly through base references.
"""

_base_ = ['./crane_eood_k1.py']

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
