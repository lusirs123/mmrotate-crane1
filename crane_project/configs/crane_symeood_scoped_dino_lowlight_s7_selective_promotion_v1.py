"""Protocol record for source-only native-protected S7 promotion V1."""

experiment_name = 'crane_symeood_s7_selective_promotion_v1'
train_components = 's7_selective_promotion'
base_epoch = 4
base_protocol = 's7_temporal_association_relative_quality'

s7_selective_promotion = dict(
    hidden=128,
    initial_uncertainty=0.5,
    advantage_gap=0.10,
    promotion_margin=0.10,
    uncertainty_multiplier=1.0,
    quality_loss_weight=1.0,
    classification_loss_weight=1.0,
    retention_weight=2.0,
    gain_weight=1.0,
    prior_weight=0.01,
    max_candidates=100,
    min_gain_sequences=2,
    feature_domain_augmentation=dict(
        probability=0.75,
        strength=0.15,
        operations=['brightness', 'blur', 'scale']),
    frozen_detector=True,
    frozen_phase2_quality_teacher=True,
    source_only=True,
    target_read=False,
    temporal_association=False,
    inference_slice_routing=False,
    sequence_identity_feature=False,
    inference='lower_confidence_bound_selective_promotion',
    uncertain_action='exact_native_fallback',
)

source_gate = dict(
    min_full_top1=688,
    min_small_top1=311,
    max_mcml=3,
    lost=0,
    dfr_nonregression=True,
    aci_nonregression=True,
    min_gain_sequences=2,
)

training = dict(
    epochs=4,
    selection_epochs=[1, 2, 3, 4],
    lr=0.001,
    lr_steps=[2, 3],
    seed=0,
    skip_target_eval=True,
)
