"""Inference-only repair for the already trained PQA v1 checkpoints.

All candidates are evaluated against the same highest-resolution P3 heatmap,
which removes cross-FPN quality calibration drift.  The 25x25 integration
grid matches the successful oracle probe instead of the v1 9x9 shortcut.

Do not train this config.  Use it to re-evaluate an existing v1 checkpoint in
a fresh output directory before spending compute on v2.
"""

_base_ = ['./crane_symeood_k1_pqa_quality_primary.py']

model = dict(
    pqa_canonical_heatmap_level=0,
    pqa_grid_size=25)

load_from = None
resume_from = None
work_dir = 'work_dirs/crane_symeood_k1_pqa_quality_primary_p3g25_eval'

