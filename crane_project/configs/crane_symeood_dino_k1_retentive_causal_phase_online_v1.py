"""Online Base V3 epoch-9 inference without cached DINO/audit loaders.

The command-line integration runner supplies current native-DINO proposals and
strictly previous raw images/proposals in memory.  This config deliberately
inherits the source-promoted Base V3 checkpoint identity from the fixed-test
config while replacing only the input assembly path.
"""

_base_ = [
    './crane_symeood_dino_k1_retentive_causal_phase_fixed_test.py']

normalization = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375], to_rgb=True)

online_test_pipeline = [
    dict(type='LoadImageFromFile'),
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

online_pipeline_contract = dict(
    protocol='base_v3_epoch9_true_online_input_v1',
    detector='SymEOODDinoGeometryRefinerTrainer',
    dino_source='FrozenDinoNativeS14Detector',
    cached_dino_loader=False,
    cached_history_loader=False,
    history_horizon=4,
    history_source='strictly_previous_online_native_dino',
    history_identity_model_input=False,
    same_forward_all_domains=True,
    domain_routing=False,
    sequence_frame_output_routing=False,
    future_frames_used=False,
    ground_truth_used=False,
    output_coordinate_space='original_image',
    top1_only=True)
