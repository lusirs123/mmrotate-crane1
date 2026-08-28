"""Source-only Dual-Tower V2.1 size refinement.

The audited V2 recomposition initializes both towers.  Center and angle stay
bitwise frozen; only the size tower is optimized with single-frame residual,
decoded SymKLD geometry, and source-train adjacent-pair error consistency.
Sequence/frame metadata forms training pairs only and is never a model input,
router feature, or inference-time state.
"""

_base_ = ['./crane_symeood_dino_geometry_refiner_full_source_v1.py']

dual_v2_checkpoint = (
    'work_dirs/crane_symeood_dino_geometry_refiner_'
    'dual_tower_source_val_v2/dual_tower_v2_recomposed.pth')
dual_v2_sha256 = (
    'c9ad3077f3dee761f0900c91263bf0962a0693f915fb0e0c9551c0a0c128336a')

geometry_refiner = dict(
    _delete_=True,
    type='DinoConditionedDualTowerGeometryRefiner',
    roi_output_size=7,
    in_channels=256,
    fc_channels=256,
    num_fcs=2,
    refine_center=True,
    refine_size=True,
    refine_angle=True,
    zero_init_output=False,
    train_size_tower=True,
    train_pose_tower=False,
    train_roi_extractor=False,
    center_loss_weight=0.0,
    size_loss_weight=1.0,
    angle_loss_weight=0.0,
    decoded_geometry_loss_weight=0.25,
    temporal_size_loss_weight=0.20,
    bbox_coder=dict(
        type='DeltaXYWHAOBBoxCoder',
        angle_range='le90', edge_swap=True, proj_xy=True,
        target_means=(0., 0., 0., 0., 0.),
        target_stds=(1., 1., 1., 1., 1.)))

evidence_contract = dict(
    source_train_frames=2781,
    source_val_frames=738,
    target_data_read=False,
    detector_forward_during_training=False,
    domain_routing=False,
    sequence_frame_routing=False,
    temporal_state=False,
    source_adjacent_pair_supervision=True,
    inference_sequence_input=False,
    sequential_source_sampler=True)

model = dict(
    geometry_refiner=geometry_refiner,
    geometry_refiner_checkpoint=dual_v2_checkpoint,
    geometry_refiner_checkpoint_sha256=dual_v2_sha256,
    evidence_contract=evidence_contract,
    evaluation_only=False)

# Sequential batches are required only to expose adjacent source frames to the
# training loss.  The trainer verifies sequence continuity and discards every
# cross-sequence or non-consecutive pair.
data = dict(
    train_dataloader=dict(
        samples_per_gpu=2, workers_per_gpu=2, shuffle=False),
    val_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    constructor='GeometryRefinerOptimizerConstructor',
    lr=2e-5,
    weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step', warmup='linear', warmup_iters=100,
    warmup_ratio=0.1, step=[5, 7])
runner = dict(type='EpochBasedRunner', max_epochs=8)

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=8,
    meta=dict(
        geometry_refiner_checkpoint_contract=dict(
            protocol='source_only_dual_tower_size_refinement_v21',
            architecture='dual_tower_size_pose_v2',
            source_train_frames=2781,
            source_val_frames=738,
            target_data_read=False,
            fixed_test_read=False,
            source_gate_passed=False,
            domain_routing=False,
            sequence_frame_routing=False,
            temporal_state=False,
            source_adjacent_pair_supervision=True,
            inference_sequence_input=False,
            sequential_source_sampler=True,
            representation='five_delta_xywha',
            angle_range='le90', edge_swap=True, proj_xy=True,
            refine_center=True, refine_size=True, refine_angle=True,
            train_size_tower=True,
            train_pose_tower=False,
            train_roi_extractor=False,
            decoded_geometry_loss='symmetric_kld',
            decoded_geometry_loss_weight=0.25,
            temporal_size_error_consistency_weight=0.20,
            initialized_from_dual_v2_sha256=dual_v2_sha256)))

seed = 3407
gpu_ids = [0]
load_from = None
resume_from = None
work_dir = (
    'work_dirs/crane_symeood_dino_geometry_refiner_'
    'dual_tower_size_source_v21_seed3407')
