"""Matched K1 warm start for the A/B/C FPN-gradient experiment.

All three arms freeze ResNet-50 and train the FPN, detection heads, and
classification adapter.  The source cache pipeline and four-epoch budget are
inherited unchanged from warmstart V2.
"""

_base_ = ['./crane_symeood_k1_dino_warmstart_common_v2.py']

# MMDetection ResNet freezes the stem and stages 1..4 when frozen_stages=4.
# Keep the setting identical in A, B, and C; the server preflight checks the
# resulting requires_grad flags after model.train().
model = dict(backbone=dict(frozen_stages=4))

