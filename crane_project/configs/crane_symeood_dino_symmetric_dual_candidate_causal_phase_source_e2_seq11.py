"""E2: source-only symmetric K1/DINO candidate fusion plus seq11.

This experiment removes the V3 hard K1-present lock.  Both candidates share
one domain-agnostic, parameter-free equal-weight OBB anchor; a common causal
phase residual head is then trained from all source GT.  Candidate absence is
handled uniformly and is not a learned/discrete takeover decision.
"""

_base_ = [
    './crane_symeood_dino_k1_retentive_causal_phase_refiner_'
    'source_v3_seq11.py']

geometry_refiner = dict(
    _delete_=True,
    type='SymmetricDualCandidateCausalPhaseGeometryRefiner',
    roi_output_size=7, in_channels=256, fc_channels=256, num_fcs=2,
    refine_center=True, refine_size=True, refine_angle=True,
    zero_init_output=True,
    center_loss_weight=1.0, size_loss_weight=1.0, angle_loss_weight=1.0,
    decoded_geometry_loss_weight=0.10,
    temporal_size_loss_weight=0.20,
    retention_loss_weight=0.0,
    history_horizon=4,
    max_current_center_delta=0.35,
    max_current_log_size_delta=0.45,
    max_current_angle_delta_deg=20.0,
    max_history_center_delta=0.08,
    max_history_log_size_delta=0.12,
    max_history_angle_delta_deg=8.0,
    history_gate_bias=-4.0,
    conditioning_gate_bias=-2.0,
    bbox_coder=dict(
        type='DeltaXYWHAOBBoxCoder', angle_range='le90',
        edge_swap=True, proj_xy=True,
        target_means=(0., 0., 0., 0., 0.),
        target_stds=(1., 1., 1., 1., 1.)))

evidence_contract = dict(
    source_train_frames=2840,
    current_k1_geometry_anchor=False,
    native_dino_anchor_fallback=False,
    symmetric_dual_candidate_anchor=True,
    candidate_blend_weight=0.5,
    candidate_blend_learned=False,
    candidate_selection_router=False,
    candidate_presence_fallback_only=True,
    same_forward_all_domains=True,
    continuous_k1_retention=False,
    fixed_target_parameter_selection=False)

model = dict(
    geometry_refiner=geometry_refiner,
    evidence_contract=evidence_contract,
    geometry_refiner_checkpoint=None,
    geometry_refiner_checkpoint_sha256=None,
    geometry_refiner_checkpoint_contract=None,
    evaluation_only=False)

checkpoint_config = dict(
    interval=1, max_keep_ckpts=10,
    meta=dict(
        geometry_refiner_checkpoint_contract=dict(
            protocol='source_only_symmetric_dual_candidate_e2_v1_seq11',
            architecture=(
                'symmetric_dual_candidate_causal_phase_refiner_e2_v1'),
            frozen_baseline_variant='symeood_k1_epoch24',
            frozen_baseline_config='crane_project/configs/crane_symeood_k1.py',
            frozen_baseline_checkpoint='work_dirs/crane_symeood_k1/epoch_24.pth',
            source_train_frames=2840, original_source_train_frames=2781,
            auxiliary_source_frames=59,
            auxiliary_source_sequence='real_seq11',
            source_val_frames=738,
            target_data_read=False, fixed_test_read=False,
            source_gate_passed=False,
            detector_forward_during_training=True,
            dino_detector_forward_during_training=False,
            frozen_symeood_feature_forward=True,
            frozen_symeood_detection_head_forward=True,
            frozen_symeood_detection_from_shared_features=True,
            cached_dino_proposals_only=True,
            domain_routing=False, sequence_frame_routing=False,
            temporal_state=False, causal_history_input=True,
            history_horizon=4, history_identity_model_input=False,
            current_k1_geometry_anchor=False,
            native_dino_anchor_fallback=False,
            native_dino_current_conditioning=True,
            symmetric_dual_candidate_anchor=True,
            candidate_blend_weight=0.5,
            candidate_blend_learned=False,
            candidate_selection_router=False,
            candidate_presence_fallback_only=True,
            same_forward_all_domains=True,
            bounded_current_residual=True,
            bounded_history_residual=True,
            rejectable_history_gate=True,
            exact_current_only_when_no_history=True,
            continuous_double_angle_phase=True,
            zero_phase_is_exact_identity=True,
            source_only_proposal_corruption=True,
            fixed_target_parameter_selection=False,
            representation='six_delta_xywh_sin2a_cos2a_residual',
            continuous_k1_retention=False, retention_loss_weight=0.0,
            source_adjacent_pair_supervision=True,
            adjacent_pair_identity_model_input=False,
            temporal_size_error_consistency=True,
            temporal_size_loss_weight=0.20,
            inference_sequence_input=False,
            single_gpu_adjacent_pair_training=True,
            train_samples_per_gpu=2, train_shuffle=False,
            auxiliary_source_independent_sequence_claim=False,
            auxiliary_source_router_claim=False,
            auxiliary_source_sparse_history=True,
            appledouble_sidecars_are_samples=False,
            angle_range='le90', edge_swap=True, proj_xy=True,
            refine_center=True, refine_size=True, refine_angle=True)))

seed = 3407
gpu_ids = [0]
load_from = None
resume_from = None
work_dir = (
    'work_dirs/crane_symeood_dino_symmetric_dual_candidate_'
    'causal_phase_source_e2_v1_seq11_seed3407')
