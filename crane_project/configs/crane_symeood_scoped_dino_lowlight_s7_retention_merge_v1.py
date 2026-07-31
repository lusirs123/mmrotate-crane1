"""Source-gated native S14 plus frozen S7 retention-aware merge.

The checkpoint referenced here is deployable only when its stored source gate
selected an epoch greater than zero.  Epoch zero remains the native-only
fallback and does not constitute an S7 improvement.
"""

_base_ = ['./crane_symeood_scoped_dino_lowlight_s7_v1.py']

_merge_head = dict(
    s7_protected_merge=True,
    s7_merge_init_bias=-2.0)

model = dict(
    dino_rescue=dict(head=_merge_head),
    dino_head_checkpoint=(
        'work_dirs/dino_teacher_s7_retention_merge_v1/'
        'labeller_best_source_only.pth'))

dino_rescue = dict(head=_merge_head)
