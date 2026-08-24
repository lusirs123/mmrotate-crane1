"""Fixed-test entry for source-calibrated conditional DINO rescue V3."""

_base_ = ['./crane_symeood_dino_unified_v1.py']

model = dict(
    conditional_dino=dict(
        enabled=True,
        calibration_json=(
            'work_dirs/crane_symeood_dino_lane_isolated_v3/'
            'source_calibration.json')))

formal_lane_isolated_conditional_contract = dict(
    protocol='source_calibrated_lane_isolated_conditional_dino_v3',
    metric_protocol_version=2,
    selection_split='val',
    target_data_read=False,
    pre_dino_signals=[
        'sym_eood_missing',
        'sym_eood_normalized_diagonal',
        'sym_eood_self_geometry_change'],
    independent_lane_state=True,
    dino_self_geometry_only=True,
    cross_lane_geometry_rejection=False,
    target_scope=False,
    sequence_identity_routing=False,
    test_parameter_search=False,
    measurement_validity_output=True)

work_dir = 'work_dirs/crane_symeood_dino_lane_isolated_v3/fixed_test'
fusion_audit_file = 'lane_isolated_conditional_fusion_audit.json'
