"""Regression-only tower control; inherited S classification stays on FPN."""
import torch
import torch.nn as nn
from mmrotate.models.builder import ROTATED_HEADS
from .sym_eood_tower_head_v1 import SymEOODTowerHeadV1


@ROTATED_HEADS.register_module()
class SymEOODRegTowerHeadV1(SymEOODTowerHeadV1):
    def __init__(self, *args, **kwargs):
        # Resource policy, not a training hyperparameter. This also runs when
        # ckpt_sweep builds the model in an independent inference process.
        self.memory_limit_mib = 6144
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            limit = self.memory_limit_mib * 2**20
            total = torch.cuda.get_device_properties(device).total_memory
            if limit >= total:
                raise RuntimeError('R4 requires a GPU with more than 6 GiB physical VRAM')
            torch.cuda.set_per_process_memory_fraction(limit / total, device)
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        # Keep only regression layers as trainable modules. The inherited
        # forward traverses an empty classifier list, then the original
        # retina_cls, exactly matching S's classification computation.
        self.cls_convs = nn.ModuleList()
