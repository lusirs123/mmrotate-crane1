# =========================================================
# SymEOOD(K=1) + Platform Context Auxiliary Supervision (§19)
#
# Relative to crane_symeood_k1.py:
#   - adds a training-only PlatformContextHead
#   - generates platform ROI masks from beam GT using per-sequence K
#   - keeps inference unchanged: simple_test still uses only the main head
#
# The seq-level K values below are fitted from
# work_dirs/crane_symeood_k1/manual_platform_polygons_train_v2.json.
# Test annotations are intentionally not used for training.
# =========================================================

_base_ = ['../crane_symeood_k1.py']

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.dense_heads.platform_context_head',
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
    platform_context_head=dict(
        type='PlatformContextHead',
        in_channels=256,
        feat_channels=128,
        stacked_convs=2,
        levels=(0, 1, 2),
        seq_platform_k=seq_platform_k,
        loss_weight=0.02,
        pos_weight=5.0,
        neg_weight=0.02,
        min_pos_pixels=1,
        use_dtheta=False,
    )
)

work_dir = 'work_dirs/crane_symeood_k1_platform_ctx'
