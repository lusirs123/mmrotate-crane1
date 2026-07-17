"""Stabilized PQA v2 for the unified 24-epoch experiment protocol.

Changes from v1 are evidence-driven:
  * a private two-conv localization tower compensates for this fork's Retina
    head having no hidden regression subnet;
  * one canonical P3 heatmap gives all FPN candidates a common quality scale;
  * 25x25 Volume-IoU matches the oracle gate and restores angle sensitivity;
  * dark supervision starts after two epochs and ramps over two more epochs;
  * dark weights are reduced so clean spatial precision remains primary.
"""

_base_ = ['./crane_symeood_k1_pqa_quality_primary.py']

model = dict(
    pqa_head=dict(
        _delete_=True,
        type='PQAHeatmapHead',
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        prior_prob=0.01),
    pqa_canonical_heatmap_level=0,
    pqa_grid_size=25,
    pqa_dark_supervision_weight=0.25,
    pqa_dark_consistency_weight=0.05,
    # The log has 650 iterations/epoch.  Learn clean H for two epochs, then
    # introduce the OOD objective smoothly over epochs 3-4.
    pqa_dark_warmup_iters=1300,
    pqa_dark_ramp_iters=1300)

load_from = None
resume_from = None
work_dir = 'work_dirs/crane_symeood_k1_pqa_quality_primary_v2'

