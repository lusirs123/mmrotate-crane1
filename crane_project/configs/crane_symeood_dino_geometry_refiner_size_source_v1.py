"""Size-only mechanism ablation for the shared geometry-refiner code path."""

_base_ = ['./crane_symeood_dino_geometry_refiner_full_source_v1.py']

model = dict(
    geometry_refiner=dict(
        refine_center=False,
        refine_size=True,
        refine_angle=False))

checkpoint_config = dict(
    meta=dict(
        geometry_refiner_checkpoint_contract=dict(
            refine_center=False,
            refine_size=True,
            refine_angle=False)))

work_dir = 'work_dirs/crane_symeood_dino_geometry_refiner_size_source_v1'

