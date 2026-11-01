"""Current R4 detector entry. Expand _base_ for the complete configuration.
Model/data/training recipe match the archived R4; old initialization/RNG replay is separate."""

_base_ = ['./_base_/crane_sgar_detector_v1.py']

model = {'bbox_head': {'type': 'SymEOODRegTowerHeadV1'}}

work_dir = 'work_dirs/crane_sgar_r4_v1'


