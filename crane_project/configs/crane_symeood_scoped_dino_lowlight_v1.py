"""Paper-method config for scope-gated frozen-DINO low-light rescue.

BrightAug and DINOv2 stay frozen.  Only the oriented RPN/ROI rescue heads are
trained on source data.  The bundled seq02 scope is target-dev diagnosis only;
replace it with acquisition metadata and an unseen low-light sequence for the
final paper test.
"""

_base_ = ['./crane_symeood_k1_brightaug.py']

# The normal MMRotate test entry builds the complete method below.  The
# dino_rescue dictionary remains the single source of training/provenance
# parameters used by the source-only head trainer.
custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.detectors.scoped_dino_lowlight_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola',
    ],
    allow_failed_imports=False)

dino_rescue = dict(
    protocol_name='Scope-Gated Frozen DINO Semantic Rescue V1',
    baseline=dict(
        config='crane_project/configs/crane_symeood_k1_brightaug.py',
        checkpoint='work_dirs/crane_symeood_k1_brightaug/epoch_20.pth',
        selection_tool='crane_project/tools/ckpt_sweep.py',
        selection_split='val',
        selection_epochs=[16, 18, 20, 22, 24],
        selection_file=(
            'work_dirs/crane_symeood_k1_brightaug/ckpt_sweep/'
            'selected_checkpoint.txt'),
        gpu=0),
    data=dict(
        root='crane_project/data/crane_grab/',
        source_train_datasets=['train:train', 'train_sim:train'],
        source_val_datasets=['val:val'],
        # The following sequence is a read-only source control for the
        # composed-method test, not the DINO-head training set.
        source_split='val', source_seq='real_seq07',
        source_val_modulus=5,
        target_dev_split='test', target_dev_seq='real_seq02',
        target_dev_start=137, target_dev_end=169),
    dinov2=dict(
        repo='third_party/dinov2',
        checkpoint='pretrained/dinov2_vitl14_pretrain.pth',
        model='dinov2_vitl14', gpus=[1, 2],
        legacy_sdpa_query_chunk=512,
        height=600, max_long_side=1333, patch_size=14),
    head=dict(
        gpu=0, rpn_feat_channels=256, roi_fc_channels=1024,
        roi_samples=256, proposal_count=2000, max_detections=2000,
        # DINO rescue has its own ROI NMS; BrightAug's main path remains
        # the custom no-NMS SymEOOD top-1 path.
        roi_nms_iou_thr=0.1),
    # DINO head has an independent schedule.  Its epoch numbers are unrelated
    # to the frozen BrightAug detector checkpoint selected above.
    train=dict(
        epochs=8, lr=0.001, momentum=0.9, weight_decay=0.0001,
        max_grad_norm=10.0,
        warmup_iters=1000, warmup_ratio=0.001,
        lr_steps=[5, 7], lr_gamma=0.1,
        checkpoint_interval=1,
        selection_epochs=[1, 2, 3, 4, 5, 6, 7, 8],
        seed=0,
        work_dir='work_dirs/dino_teacher_scoped_lowlight_v1_formal8',
        feature_cache_dir=(
            'work_dirs/dino_teacher_scoped_lowlight_v1_formal8/feature_cache'),
        out_json=(
            'work_dirs/dino_teacher_scoped_lowlight_v1_formal8/'
            'train_result.json')),
    test=dict(
        labeller_checkpoint=(
            'work_dirs/dino_teacher_scoped_lowlight_v1_formal8/'
            'labeller_best_source_only.pth'),
        scope_manifest=(
            'crane_project/configs/scopes/'
            'seq02_lowlight_target_dev_diagnosis.json'),
        feature_cache_dir=(
            'work_dirs/dino_teacher_scoped_lowlight_v1_formal8/'
            'feature_cache'),
        out_json=(
            'work_dirs/dino_teacher_scoped_lowlight_v1_formal8/'
            'target_dev_test_result.json')),
    full_test=dict(
        labeller_checkpoint=(
            'work_dirs/dino_teacher_scoped_lowlight_v1_formal8/'
            'labeller_best_source_only.pth'),
        scope_manifest=(
            'crane_project/configs/scopes/'
            'full_test_seq02_lowlight_diagnosis.json'),
        confirm_diagnosis_scope=True,
        feature_cache_dir=(
            'work_dirs/dino_teacher_scoped_lowlight_v1_formal8/'
            'feature_cache'),
        out_dir=(
            'work_dirs/dino_teacher_scoped_lowlight_v1_formal8/'
            'full_test_diagnosis_v2'),
        out_json=(
            'work_dirs/dino_teacher_scoped_lowlight_v1_formal8/'
            'full_test_diagnosis_v2/result.json')))

# Deployable inference composition.  The positional checkpoint passed to
# tools/test.py is the unchanged BrightAug checkpoint; the detector loads the
# source-selected frozen-DINO head separately and keeps it out of the module
# tree so its GPU-1/GPU-2 sharding survives MMDataParallel.
model = dict(
    _delete_=True,
    type='ScopedDinoLowlightDetector',
    baseline_config='crane_project/configs/crane_symeood_k1_brightaug.py',
    dino_rescue=dino_rescue,
    dino_head_checkpoint=(
        'work_dirs/dino_teacher_scoped_lowlight_v1_formal8/'
        'labeller_best_source_only.pth'),
    scope_manifest=(
        'crane_project/configs/scopes/full_test_seq02_lowlight_diagnosis.json'),
    scope_split='test',
    stabilizer=dict(
        enabled=True,
        alpha=0.25,
        selection_split='val',
        candidates=[0.25, 0.5, 0.75, 1.0],
        max_source_riou_drop=0.005,
        target_used_for_selection=False),
    test_cfg=dict(score_thr=0.05, max_per_img=1),
)

evaluation = dict(
    interval=1,
    metric='mAP',
    thresh_sim=10.0,
    thresh_real=25.0,
    weight_sim=0.7,
    weight_real=0.3,
    paper_temporal=True,
    temporal_center_thresh_px=15.0,
    temporal_ekf_window=10,
    temporal_mcml_limit=5,
    temporal_iou_thresh=0.5,
)
