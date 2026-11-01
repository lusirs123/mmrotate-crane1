"""S4x three-factor ablation base. Inherits the historical SymEOOD+scale-B
configuration unchanged; children override ONLY the three ablation switches and
work_dir. Fixed augmentation: PortIsotropicShrink(p=.5, [0.5,1.0]).
"""
_base_ = ['./crane_symeood_k1_port_day2night_aug_b_v1.py']

work_dir = 'work_dirs/symeood_ablation_v1/base'
