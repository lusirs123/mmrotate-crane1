"""Fixed F-S v1 formal comparison, authorized after the bounded preflight.

Inherits the reviewed candidate verbatim: beta=.1, lambda=.1, original
ImageNet initialization/seed0/24 epochs. No B resume or D/H stacking.
Separate entry preserves the preflight config and its historical SHA.
"""
_base_ = ['./crane_symeood_k1_port_day2night_size_f_s_v1.py']
