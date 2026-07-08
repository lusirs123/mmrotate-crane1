# =========================================================
# SymEOOD(K=1) + stronger Platform Context Feature Injector
#
# This config keeps the stage2 boundary:
#   - platform context modulates FPN features in train/test
#   - final output is still only the beam OBB from the main bbox head
#   - no platform box inversion, no candidate filtering/reranking
#
# Difference from crane_symeood_k1_platform_injector.py:
#   - stronger gate_scale
#   - non-zero initial gate_alpha, so inference modulation is active from the
#     beginning instead of waiting for alpha to move away from zero
#   - slightly stronger context loss to keep the platform map informative
# =========================================================

_base_ = ['./crane_symeood_k1_platform_injector.py']

model = dict(
    platform_context_injector=dict(
        gate_scale=0.30,
        init_gate_alpha=0.50,
        loss_weight=0.03,
    )
)

work_dir = 'work_dirs/crane_symeood_k1_platform_injector_strong'
