"""Dynamic-hard-negative source-only S7 lane arbitration.

The detector checkpoint is usable only when the stored source gate selected an
epoch greater than zero.  Training remains anchored to the audited epoch-1
affine merge and must not read target data.
"""

_base_ = ['./crane_symeood_scoped_dino_lowlight_s7_lane_arbitration_v1.py']

model = dict(
    dino_head_checkpoint=(
        'work_dirs/dino_teacher_s7_lane_arbitration_v2/'
        'labeller_best_source_only.pth'))

s7_lane_training = dict(
    base_checkpoint=(
        'work_dirs/dino_teacher_s7_retention_merge_v1/'
        'labeller_epoch_01_source_only.pth'),
    source_gate='exact_retention',
    hard_negative_ranking='current_adjusted_s7_log_odds',
    hard_negatives=4,
    gain_repeat=8,
    gain_competitors=['native_s14', 'wrong_supplement_s7'],
    source_train_only=True)
