"""Student-only inference config after semantic distillation."""

_base_ = ['./crane_symeood_k1.py']

# The classification adapter is part of the learned student.  The DINO cache
# and 256->1024 training projection are deliberately absent.
model = dict(bbox_head=dict(use_semantic_cls_adapter=True))

load_from = None
resume_from = None
