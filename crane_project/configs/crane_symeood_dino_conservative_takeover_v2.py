"""Fixed-test entry for source-calibrated conservative takeover V2.

The calibration JSON must be produced exclusively from the 738-frame source
validation split.  Detector construction rejects target-read, failed-gate, or
missing calibration payloads.
"""

_base_ = ['./crane_symeood_dino_unified_v1.py']

model = dict(
    conservative_takeover=dict(
        enabled=True,
        calibration_json=(
            'work_dirs/crane_symeood_dino_conservative_takeover_v2/'
            'source_calibration.json')))

formal_conservative_takeover_contract = dict(
    protocol='source_calibrated_conservative_takeover_v2',
    selection_split='val',
    target_data_read=False,
    default_lane='symeood_k1_brightaug',
    rescue_lane='frozen_dino_native_s14',
    asymmetric_hysteresis=True,
    causal_confirmation=True,
    source_owned_geometry=True,
    test_parameter_search=False)

formal_detection_contract = dict(
    conservative_takeover=True,
    conservative_takeover_calibration='source_val_only',
    lane_hysteresis=True,
    test_parameter_search=False)

work_dir = (
    'work_dirs/crane_symeood_dino_conservative_takeover_v2/full_test')
fusion_audit_file = 'conservative_takeover_fusion_audit.json'
