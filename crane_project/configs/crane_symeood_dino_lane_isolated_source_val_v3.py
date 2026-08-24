"""Collect the source-val lane audit required by conditional DINO V3.

This stage runs both frozen lanes on all 738 source-validation frames.  It does
not enable V3 routing; its only purpose is to record raw SymEOOD geometry and
the native-DINO lane for CPU-only source calibration.
"""

_base_ = ['./crane_symeood_dino_unified_source_val_v1.py']

work_dir = (
    'work_dirs/crane_symeood_dino_lane_isolated_v3/source_val_collect')
fusion_audit_file = 'source_val_lane_audit.json'

formal_lane_isolated_source_contract = dict(
    protocol='lane_isolated_conditional_dino_v3_source_collection',
    split='val',
    expected_frames=738,
    target_data_read=False,
    both_frozen_lanes_run=True,
    raw_symeood_box_required=True,
    parameter_search=False)
