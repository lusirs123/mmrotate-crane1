"""BrightAug + gradient-isolated reg-quality-primary ranking.

Unified protocol:
  * initialize from K1 epoch_24 via the root tools/dist_train.sh command;
  * train for 24 epochs;
  * evaluate only epochs 16/18/20/22/24;
  * use a fresh work_dir/cache.

The main classification target and bbox losses are inherited unchanged from
BrightAug.  Classification only retains a broad pre-threshold top-10000 pool;
the independent quality score selects the final top-1 without cls×quality
fusion.
"""

_base_ = ['./crane_symeood_k1_brightaug.py']

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.dense_heads.reg_quality_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola',
    ],
    allow_failed_imports=False)

model = dict(
    reg_quality_head=dict(
        type='RegQualityHead',
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        num_anchors=3,
        prior_prob=0.01),
    # Hard isolation: quality loss updates only RegQualityHead.
    reg_quality_detach=True,
    reg_quality_loss_weight=1.0,
    reg_quality_focal_gamma=2.0,
    reg_quality_min_target_iou=0.1,
    # Keep low-cls candidates before any absolute score threshold.
    reg_quality_pre_topk=10000,
    test_cfg=dict(
        nms_pre=10000,
        min_bbox_size=0,
        score_thr=0.0,
        nms=dict(iou_thr=0.1),
        max_per_img=1))

load_from = None
resume_from = None
work_dir = 'work_dirs/crane_symeood_k1_regquality_primary'
