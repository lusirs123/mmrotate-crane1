"""Source-only non-positive quality suppression on affine S7 epoch 1.

The learned module emits one shared delta in [-2, 0] for every S7 candidate
in a frame.  It cannot promote S7 or change its internal ordering.  The best
checkpoint remains the native S14 fallback unless source validation reaches
lost=0, full>=688, small>=311, and MCML<=3.
"""

_base_ = ['./crane_symeood_scoped_dino_lowlight_s7_retention_merge_v1.py']

_quality_head = dict(
    s7_protected_merge=True,
    s7_lane_arbitration=False,
    s7_quality_suppression=True,
    s7_quality_hidden=32,
    s7_quality_max_suppression=2.0,
    s7_quality_init_risk_bias=0.0)

model = dict(
    dino_rescue=dict(head=_quality_head),
    dino_head_checkpoint=(
        'work_dirs/dino_teacher_s7_quality_suppression_v1/'
        'labeller_best_source_only.pth'))

dino_rescue = dict(head=_quality_head)

s7_quality_training = dict(
    base_checkpoint=(
        'work_dirs/dino_teacher_s7_retention_merge_v1/'
        'labeller_epoch_01_source_only.pth'),
    base_epoch=1,
    source_train_only=True,
    source_gate=dict(
        exact_retention=True,
        min_full_top1=688,
        min_small_top1=311,
        max_mcml=3),
    lane_wide=True,
    adjustment_range=[-2.0, 0.0],
    source_support_preflight=dict(
        exact_training_risk_miner=True,
        minimum_risk_pairs=1,
        zero_risk_action='skip_optimization_and_keep_epoch_0',
        report_all_s7_wrong_frames=True),
    positive_promotion=False,
    gain_replay=False,
    target_gate='formal_source_gate_only')
