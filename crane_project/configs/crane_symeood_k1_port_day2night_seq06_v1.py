"""Original K1 training settings on the in-place seq06-restored dataset.

Expected counts: train 1810 + train_sim 748, VAL 887, TEST 1440.
The parent supplies dataset paths and inherits all original K1 parameters.
Use a new output directory to preserve the earlier no-seq06 experiment.
"""

_base_ = ['./crane_symeood_k1_port_day2night_v1.py']

work_dir = 'work_dirs/crane_symeood_k1_port_day2night_seq06_v1'
