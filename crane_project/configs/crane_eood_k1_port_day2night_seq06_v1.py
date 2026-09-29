"""EOOD K=1 control on the existing dataset with seq06 restored.

Train: real 1810 + sim 748; VAL 887; TEST 1440.
Inherit the original EOOD training settings and ImageNet initialization.
"""

_base_ = ['./crane_eood_k1_port_day2night_v1.py']

work_dir = 'work_dirs/crane_eood_k1_port_day2night_seq06_v1'
