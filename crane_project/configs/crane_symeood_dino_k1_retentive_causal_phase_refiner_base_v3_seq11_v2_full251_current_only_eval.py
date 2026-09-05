"""Frozen Base-V3 epoch9 current-only ablation on seq11-v2 full251.

The checkpoint, current image, K1 anchor, and cached DINO conditioning are
identical to the full Base-V3 diagnostic.  ``current_only`` uniformly removes
the history residual from every output.  History tensors remain loadable only
to keep the public forward signature and preprocessing graph unchanged.
"""

_base_ = [
    './crane_symeood_dino_k1_retentive_causal_phase_refiner_'
    'base_v3_seq11_v2_full251_eval.py']

model = dict(
    geometry_refiner=dict(inference_component_mode='current_only'))

source_only_result_contract = dict(
    _delete_=True,
    protocol='base_v3_epoch9_seq11_v2_full251_current_only_inference_v1',
    expected_frame_count=251,
    checkpoint_role='base_v3_promoted_epoch9',
    source_training_frame_count=2781,
    source_split='extra_source_real_seq11_pilot_k1p9_v2',
    prediction_coordinate_system='original_image_pixels',
    obb_convention='le90',
    annotation_k0=1.9,
    target_geometry='top_beam_only',
    inference_component_mode='current_only',
    history_tensors_loaded=True,
    history_output_contribution=False,
    same_setting_all_frames=True,
    domain_routing=False,
    sequence_frame_routing=False,
    optimizer_steps=0,
    target_data_read=False,
    fixed_test_read=False,
    eligible_for_epoch_selection=False,
    eligible_for_checkpoint_promotion=False)

work_dir = (
    'work_dirs/crane_symeood_dino_source_inventory_v2/'
    'real_seq11_k1p9_v2/base_v3_epoch9_current_only_full251')
