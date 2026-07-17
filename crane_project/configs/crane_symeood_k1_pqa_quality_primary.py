"""Full PQA heatmap + illumination consistency + quality-primary ranking.

This experiment differs from the failed scalar RegQualityHead in two ways:
  1. it predicts a dense GT-relative position heatmap instead of one scalar
     IoU per anchor;
  2. every decoded candidate contributes its explicit OBB geometry through
     PQA Volume-IoU during inference.

The main classification target, bbox loss, backbone, and FPN training remain
the BrightAug baseline.  PQA receives detached features.  Its paired dark-view
supervision and consistency loss therefore update only PQAHeatmapHead.

Unified protocol:
  * initialize from K1 epoch_24 through the command-line load_from override;
  * train 24 epochs with root tools/dist_train.sh (seed=0);
  * evaluate only epochs 16/18/20/22/24;
  * never reuse another branch's ckpt_sweep cache.
"""

_base_ = ['./crane_symeood_k1_brightaug.py']

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.dense_heads.pqa_heatmap_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola',
    ],
    allow_failed_imports=False)

model = dict(
    pqa_head=dict(
        type='PQAHeatmapHead',
        in_channels=256,
        prior_prob=0.01),
    # Hard isolation: neither clean nor dark PQA loss updates the main model.
    pqa_detach=True,
    # Paper-style dense localization-distribution supervision.
    pqa_ld_loss_weight=1.5,
    pqa_ld_gamma=2.0,
    # The oracle showed usable candidates can rank as deep as 9661 by cls.
    pqa_pre_topk=10000,
    # Task adaptation: quality directly selects top-1.  For the faithful PQA
    # baseline, evaluate the same checkpoint with
    #   --cfg-options model.pqa_score_mode=cls_x_quality
    pqa_score_mode='quality',
    pqa_grid_size=9,
    pqa_quality_batch_size=512,
    # Innovation for unseen dark frames: supervise the same spatial heatmap
    # on a stronger photometric view and align it to a clean-view teacher.
    pqa_dark_supervision_weight=0.5,
    pqa_dark_consistency_weight=0.1,
    # This transform is applied after pipeline BrightAug.  A moderate second
    # gamma avoids turning already-dark samples into textureless black frames;
    # composed minima still reach an effective gamma near 0.2.
    pqa_dark_gamma_range=(0.5, 0.9),
    pqa_dark_contrast_range=(0.7, 1.1),
    pqa_dark_noise_std_range=(0.0, 10.0),
    test_cfg=dict(
        nms_pre=10000,
        min_bbox_size=0,
        score_thr=0.0,
        nms=dict(iou_thr=0.1),
        max_per_img=1))

load_from = None
resume_from = None
work_dir = 'work_dirs/crane_symeood_k1_pqa_quality_primary'
