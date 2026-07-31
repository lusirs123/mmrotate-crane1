"""Source-only lane-aware arbitration on the fixed epoch-1 S7 merge.

This is an experimental training configuration.  It does not authorize target
selection or deployment; the resulting checkpoint must pass the source exact
retention gate before any target-dev comparison.
"""

_base_ = ['./crane_symeood_scoped_dino_lowlight_s7_retention_merge_v1.py']

_lane_head = dict(
    s7_protected_merge=True,
    s7_lane_arbitration=True,
    s7_lane_hidden=32,
    s7_lane_max_adjustment=2.0,
    s7_lane_base_epoch=1)

model = dict(
    dino_rescue=dict(head=_lane_head),
    dino_head_checkpoint=(
        'work_dirs/dino_teacher_s7_lane_arbitration_v1/'
        'labeller_best_source_only.pth'))

dino_rescue = dict(head=_lane_head)

s7_lane_training = dict(
    base_checkpoint=(
        'work_dirs/dino_teacher_s7_retention_merge_v1/'
        'labeller_epoch_01_source_only.pth'),
    source_gate='exact_retention')
