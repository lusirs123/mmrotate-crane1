"""Fixed-target diagnostic for E1 trained with all 59 seq11 source frames.

This entry point answers one bounded ablation question: whether adding all 59
labelled seq11 frames to the original 2781-frame source training set changes
the fixed 992-frame target benchmark.  Epoch 10 is the predeclared training
endpoint; TEST must not select another epoch.  Because the benchmark has
already been inspected and may inform later data collection, this run is not
an untouched final-test claim.
"""

_base_ = [
    './crane_symeood_dino_k1_retentive_causal_phase_refiner_'
    'source_v3_seq11.py']

evidence_role = 'fixed-target-development-diagnostic'
candidate_epoch_policy = 'fixed_training_endpoint_epoch10'
source_training_frame_count = 2840
auxiliary_source_frame_count = 59
test_used_for_epoch_selection = False
eligible_for_unbiased_final_test_claim = False
eligible_for_unknown_sequence_claim = False

geometry_refiner = dict(
    zero_init_output=False,
    inference_component_mode='full',
    center_loss_weight=0.0,
    size_loss_weight=0.0,
    angle_loss_weight=0.0,
    decoded_geometry_loss_weight=0.0,
    temporal_size_loss_weight=0.0,
    retention_loss_weight=0.0)

model = dict(
    geometry_refiner=geometry_refiner,
    geometry_refiner_checkpoint=None,
    geometry_refiner_checkpoint_sha256=None,
    geometry_refiner_checkpoint_contract=None,
    evaluation_only=True)

fixed_target_audit = (
    'work_dirs/crane_symeood_dino_conservative_takeover_v2/'
    'full_test_metric_v2/conservative_takeover_fusion_audit.json')
normalization = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375], to_rgb=True)
fixed_target_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDinoProposalFromAudit', audit_json=fixed_target_audit,
         expected_frame_count=992, expected_split='test'),
    dict(type='LoadCausalHistoryFromAudit', audit_json=fixed_target_audit,
         history_horizon=4, expected_frame_count=992,
         expected_split='test'),
    dict(type='MultiScaleFlipAug', img_scale=(1024, 1024), flip=False,
         transforms=[
             dict(type='RResize'),
             dict(type='Normalize', **normalization),
             dict(type='Pad', size=(1024, 1024),
                  pad_val=dict(img=(114.0, 114.0, 114.0))),
             dict(type='PrepareCausalHistoryInputs', **normalization),
             dict(type='DefaultFormatBundle'),
             dict(type='FormatDinoProposal'),
             dict(type='FormatCausalHistoryInputs'),
             dict(type='Collect', keys=[
                 'img', 'dino_proposals', 'causal_history_images',
                 'causal_history_proposals', 'causal_history_valid_mask',
                 'causal_history_ages'])])]

fixed_target_dataset = dict(
    type='CraneDataset', data_root='crane_project/data/crane_grab/',
    ann_file='test/annfiles/', img_prefix='test/images/',
    pipeline=fixed_target_pipeline, version='le90')
data = dict(
    _delete_=True,
    val=fixed_target_dataset,
    test=fixed_target_dataset,
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False))
evaluation = dict(
    _delete_=True, metric='mAP', thresh_sim=10.0, thresh_real=25.0,
    weight_sim=0.7, weight_real=0.3, paper_temporal=True,
    temporal_center_thresh_px=15.0, temporal_ekf_window=10,
    temporal_mcml_limit=5, temporal_iou_thresh=0.5)

load_from = None
resume_from = None
work_dir = (
    'work_dirs/crane_symeood_dino_k1_retentive_causal_phase_refiner_'
    'source_v3_seq11_e1_epoch10_fixed_target_diagnostic')
