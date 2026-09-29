"""K=1 on the frozen seq06-restored port split.

Train real 1810 + sim 748; VAL 887; TEST 1440.
Inherit original model/training settings; override dataset paths and work_dir.
Historical config bytes are archived in dataset provenance/config_retirement_20260929.
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

work_dir = 'work_dirs/crane_eood_k1_port_day2night_seq06_v1'
