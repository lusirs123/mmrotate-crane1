"""Protocol record for the one-shot source-only static S7 ranker.

The locked launcher in ``tools/dino_teacher_s7_static_domain_ranker_train.py``
is the executable source of truth.  This config documents the intended
experiment and keeps the stage separate from temporal association, EKF, and
target-dev routing.
"""

experiment_name = 'crane_symeood_s7_static_domain_ranker_v1'
train_components = 's7_static_domain_ranker'
base_epoch = 4
base_protocol = 's7_temporal_association_relative_quality'

s7_static_domain_ranker = dict(
    hidden=128,
    score_weight=1.0,
    quality_loss_weight=1.0,
    relative_loss_weight=0.5,
    relative_margin=0.25,
    relative_min_gap=0.10,
    relative_max_pairs=128,
    rank_margin=0.25,
    retention_weight=2.0,
    gain_weight=1.0,
    prior_weight=0.01,
    max_candidates=100,
    feature_domain_augmentation=dict(
        probability=0.75,
        strength=0.15,
        operations=['brightness', 'blur', 'scale']),
    source_only=True,
    target_read=False,
    temporal_association=False,
    inference_slice_routing=False,
    gain_replay=False,
    exact_old_correct_retention=True,
)

source_gate = dict(
    min_full_top1=688,
    min_small_top1=311,
    max_mcml=3,
    lost=0,
)

training = dict(
    epochs=4,
    selection_epochs=[1, 2, 3, 4],
    lr=0.001,
    lr_steps=[2, 3],
    seed=0,
    skip_target_eval=True,
)
