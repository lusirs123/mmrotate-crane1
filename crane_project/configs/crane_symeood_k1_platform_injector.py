# =========================================================
# SymEOOD(K=1) + Platform Context Feature Injector (§19 stage2)
#
# Relative to crane_symeood_k1.py:
#   - adds a PlatformContextInjector, not a box-producing head
#   - train/test both modulate selected FPN features before the main bbox head
#   - final output remains beam OBBs from the main SymEOOD bbox head only
#   - no platform-to-beam box inversion, no candidate filtering/reranking
#
# This is intentionally separate from crane_symeood_k1_platform_ctx.py:
#   - platform_ctx      = stage1 training-only auxiliary supervision
#   - platform_injector = stage2 feature-level train/test modulation
# =========================================================

_base_ = ['./crane_symeood_k1.py']

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.dense_heads.platform_context_injector',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola',
    ],
    allow_failed_imports=False)

seq_platform_k = dict(
    real_seq01=dict(
        width_k=0.98732591,
        height_k=1.62260842,
        offset_long_k=0.00934639,
        offset_short_k=0.00164862,
        dtheta=0.0),
    real_seq04=dict(
        width_k=0.77059758,
        height_k=1.53177023,
        offset_long_k=0.01926015,
        offset_short_k=0.03946324,
        dtheta=0.0),
    real_seq05=dict(
        width_k=0.99586868,
        height_k=1.62817121,
        offset_long_k=0.02955090,
        offset_short_k=-0.02368786,
        dtheta=0.0),
    real_seq06=dict(
        width_k=0.76033056,
        height_k=1.49701762,
        offset_long_k=0.00316936,
        offset_short_k=0.02095947,
        dtheta=0.0),
    sim_seq08=dict(
        width_k=0.69372499,
        height_k=1.74030745,
        offset_long_k=0.01328498,
        offset_short_k=0.03850464,
        dtheta=0.0),
)

model = dict(
    platform_context_injector=dict(
        type='PlatformContextInjector',
        in_channels=256,
        feat_channels=128,
        stacked_convs=2,
        levels=(0, 1, 2),
        inject_levels=(0, 1, 2),
        seq_platform_k=seq_platform_k,
        loss_weight=0.02,
        pos_weight=5.0,
        neg_weight=0.02,
        min_pos_pixels=1,
        use_dtheta=False,
        gate_scale=0.15,
        init_gate_alpha=0.0,
        detach_modulation=False,
        apply_at_train=True,
        apply_at_test=True,
    )
)

work_dir = 'work_dirs/crane_symeood_k1_platform_injector'
