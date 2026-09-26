"""C: 24-epoch K1-initialized object/background relation training."""

_base_ = ['./crane_symeood_k1_dino_object_background_relation_common_v5.py']

model = dict(object_background_relation=dict(
    enabled=True,
    loss_weight=0.05,
    feature_level=0,
    # Keep geometry protected from this auxiliary relation term.  The K1
    # detection loss still trains FPN and detection heads in both arms.
    protect_geometry=True))
work_dir = 'work_dirs/crane_symeood_k1_dino_object_background_relation_c_v5'
