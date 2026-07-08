import torch
import torch.nn as nn

from .platform_context_head import PlatformContextHead
from mmrotate.models.builder import ROTATED_HEADS


@ROTATED_HEADS.register_module(force=True)
class PlatformContextInjector(PlatformContextHead):
    """Platform context feature-level injector.

    The module predicts the same platform context map as PlatformContextHead,
    but uses that map to modulate selected FPN levels in both training and
    inference. It never decodes boxes and never changes the output contract:
    the detector still returns only beam OBBs from the main bbox head.
    """

    def __init__(self,
                 inject_levels=None,
                 gate_scale=0.15,
                 init_gate_alpha=0.0,
                 detach_modulation=False,
                 apply_at_train=True,
                 apply_at_test=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.inject_levels = (
            tuple(int(x) for x in inject_levels)
            if inject_levels is not None else self.levels)
        self.gate_scale = float(gate_scale)
        self.gate_alpha = nn.Parameter(
            torch.tensor(float(init_gate_alpha), dtype=torch.float32))
        self.detach_modulation = bool(detach_modulation)
        self.apply_at_train = bool(apply_at_train)
        self.apply_at_test = bool(apply_at_test)

    def _modulate_with_preds(self, feats, preds, apply=True):
        if not apply:
            return feats

        out_feats = list(feats)
        pred_by_level = {
            int(level): pred
            for level, pred in zip(self.levels, preds)
        }
        for level in self.inject_levels:
            if level < 0 or level >= len(out_feats):
                continue
            if level not in pred_by_level:
                continue
            logit = pred_by_level[level]
            if self.detach_modulation:
                logit = logit.detach()
            gate = logit.tanh()
            scale = self.gate_scale * torch.tanh(self.gate_alpha)
            out_feats[level] = out_feats[level] * (1.0 + scale * gate)

        return tuple(out_feats) if isinstance(feats, tuple) else out_feats

    def forward_inject(self, feats, train=True):
        preds = self(feats)
        apply = self.apply_at_train if train else self.apply_at_test
        return self._modulate_with_preds(feats, preds, apply=apply)

    def forward_train_features(self, feats, img_metas, gt_bboxes):
        preds = self(feats)
        modulated_feats = self._modulate_with_preds(
            feats, preds, apply=self.apply_at_train)
        losses = self.loss(preds, img_metas, gt_bboxes)
        losses.update(dict(
            platform_gate_alpha=self.gate_alpha.detach(),
            platform_gate_scale_eff=(
                self.gate_scale * torch.tanh(self.gate_alpha.detach())),
        ))
        return modulated_feats, losses

    def forward_test_features(self, feats):
        preds = self(feats)
        return self._modulate_with_preds(
            feats, preds, apply=self.apply_at_test)
