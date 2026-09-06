"""Official 738-frame source-val check of frozen Base-V3 current-only mode.

This configuration changes only the uniformly applied inference component
mode of the promoted epoch-9 refiner.  It performs no training or epoch
selection and contains no fixed-TEST dataset.
"""

_base_ = ['./crane_symeood_dino_k1_retentive_component_source_val.py']

source_val_audit = (
    'work_dirs/crane_symeood_dino_conservative_takeover_v2/'
    'source_calibration_collect/source_val_fusion_source_audit.json')
runtime_input_files = dict(dino_source_val_audit=source_val_audit)

model = dict(
    geometry_refiner=dict(inference_component_mode='current_only'),
    evaluation_only=True)

# MMCV merges the parent config dictionary but does not inject the parent's
# temporary Python variables into this child module's execution namespace.
expected_checkpoint = (
    'work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_'
    'source_v3_seed3407/k1_retentive_v3_epoch9_promoted.pth')
expected_checkpoint_protocol = (
    'source_gated_k1_retentive_causal_phase_refiner_v3')
expected_checkpoint_source_train_frames = 2781
expected_checkpoint_target_data_read = False
expected_checkpoint_fixed_test_read = False

source_only_result_contract = dict(
    protocol='base_v3_epoch9_current_only_official_source_val_v1',
    runtime_audit_protocol='mmdet_runtime_inference_resource_audit_v2',
    evidence_boundary='official_source_val_738_pass_fail_only',
    expected_frame_count=738,
    checkpoint_role='base_v3_promoted_epoch9',
    source_training_frame_count=2781,
    source_split='val',
    inference_component_mode='current_only',
    history_tensors_loaded=True,
    history_output_contribution=False,
    same_setting_real_sim=True,
    domain_routing=False,
    sequence_frame_routing=False,
    optimizer_steps=0,
    training=False,
    epoch_selection=False,
    target_data_read=False,
    fixed_test_read=False,
    eligible_for_checkpoint_promotion=False,
    eligible_for_fixed_test=False)

work_dir = (
    'work_dirs/crane_symeood_dino_k1_retentive_component_'
    'current_only_source_val')
