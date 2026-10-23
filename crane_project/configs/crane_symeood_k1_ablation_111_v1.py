"""S4x ablation group 111: A=1 B=1 C=1.
Engineering definition only; no training launched by this file.
Inherits the historical SymEOOD+scale-B config. Overrides are limited
to the three ablation switches and work_dir.
"""
_base_ = ['./crane_symeood_k1_ablation_v1_base.py']

work_dir = 'work_dirs/symeood_ablation_v1/111'

# No switch override: A=1, B=1, C=1 -> historical SymEOOD + scale B (af8307c).
