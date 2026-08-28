"""Source-val-only evaluation of the recomposed Dual-Tower Refiner V2."""

_base_ = ['./crane_symeood_dino_geometry_refiner_full_source_v1.py']

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
    zero_init_output=True,
    center_loss_weight=1.0,
    size_loss_weight=1.0,
    angle_loss_weight=1.0,
    bbox_coder=dict(
        type='DeltaXYWHAOBBoxCoder',
        angle_range='le90', edge_swap=True, proj_xy=True,
        target_means=(0., 0., 0., 0., 0.),
        target_stds=(1., 1., 1., 1., 1.)))

model = dict(
    geometry_refiner=geometry_refiner,
    evaluation_only=True)

checkpoint_config = dict(
    meta=dict(
        geometry_refiner_checkpoint_contract=dict(
            protocol='source_only_dual_tower_component_recomposition_v2',
            architecture='dual_tower_size_pose_v2',
            source_train_frames=2781,
            source_val_frames=738,
            target_data_read=False,
            fixed_test_read=False,
            source_gate_passed=False,
            domain_routing=False,
            sequence_frame_routing=False,
            temporal_state=False,
            representation='five_delta_xywha',
            angle_range='le90', edge_swap=True, proj_xy=True,
            refine_center=True, refine_size=True, refine_angle=True)))

work_dir = (
    'work_dirs/crane_symeood_dino_geometry_refiner_'
    'dual_tower_source_val_v2')
