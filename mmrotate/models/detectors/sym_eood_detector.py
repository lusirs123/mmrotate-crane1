# mmrotate/models/detectors/sym_eood_detector.py
import copy
import inspect
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmrotate.models.builder import ROTATED_DETECTORS, build_head
from mmrotate.models.dense_heads.rotated_atss_head import RotatedATSSHead
from mmrotate.core import rbbox2result

@ROTATED_DETECTORS.register_module(force=True)
class SymEOOD(SingleStageDetector):
    """
    Symmetric EOOD Detector（MMRotate 0.x 兼容版）
    支持三种辅助头模式（互斥）：
      mode A: aux_bbox_head — Anchor-based 辅助头（如 RotatedATSS）
      mode B: gaussian_head — Anchor-free 高斯热图辅助头
      mode C: uadh_head   — 不确定性感知对角线辅助头（UADH）
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 aux_bbox_head=None,
                 gaussian_head=None,
                 uadh_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
            # 如果 detector 级别没有 train_cfg/test_cfg，从 bbox_head 内部提取
        if train_cfg is None and bbox_head is not None:
            train_cfg = bbox_head.get('train_cfg', None)
        if test_cfg is None and bbox_head is not None:
            test_cfg = bbox_head.get('test_cfg', None)
            
        super(SymEOOD, self).__init__(
            backbone, neck, bbox_head,
            train_cfg, test_cfg, pretrained, init_cfg
        )

        # Mode A: Anchor-based 辅助头
        if aux_bbox_head is not None:
            self.aux_heads = nn.ModuleList()
            for head_cfg in aux_bbox_head:
                head_cfg = copy.deepcopy(head_cfg)
                head_cfg.setdefault('train_cfg', train_cfg)
                head_cfg.setdefault('test_cfg', test_cfg)
                self.aux_heads.append(build_head(head_cfg))
        else:
            self.aux_heads = None

        # Mode B: Anchor-free 高斯热图辅助头
        if gaussian_head is not None:
            self.gaussian_head = build_head(gaussian_head)
        else:
            self.gaussian_head = None

        # Mode C: UADH 不确定性对角线辅助头
        if uadh_head is not None:
            self.uadh_head = build_head(uadh_head)
        else:
            self.uadh_head = None

    def _build_aux_feats(self, feats, aux_head):
        if isinstance(aux_head, RotatedATSSHead):
            return [(feat, feat) for feat in feats]
        return feats

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """0.x 标准联合训练入口 + 可选 L_equi / L_invar / degraded-cls.

        L_equi: flip(img) 取角度预测, 不参与检测损失.
        L_invar: T_photo(img) 取角度预测, 不参与检测损失.
        degraded-cls: T_degrade(img) 只参与分类损失, 不参与 bbox/angle/equi.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)

        x = self.extract_feat(img)
        losses = dict()

        # 主头损失
        main_outs = self.bbox_head(x)

        # --- L_equi 路径 (翻转) ---
        equi_flip_outs = None
        equi_flip_dirs = None
        if (hasattr(self.bbox_head, 'use_equi_loss')
                and self.bbox_head.use_equi_loss):
            flip_img, equi_flip_dirs = self._build_equi_flip_view(img)
            flip_x = self.extract_feat(flip_img)
            equi_flip_outs = self.bbox_head(flip_x)

        # --- L_invar 路径 (光度) ---
        photo_outs = None
        if (hasattr(self.bbox_head, 'use_invar_loss')
                and self.bbox_head.use_invar_loss):
            photo_img = self._build_photo_view(img, img_metas)
            photo_x = self.extract_feat(photo_img)
            photo_outs = self.bbox_head(photo_x)

        # --- degraded 路径 ---
        # 旧 degraded-cls: 退化图复用主头分类塔, 仅做兼容既有实验.
        # 新 degraded-aux2: 退化图进入独立 aux2 分类塔, 并做幅度域一致性.
        degraded_outs = None
        degraded_aux2_cls_scores = None
        degraded_aux2_amp_loss = None
        use_degraded_cls = (
            hasattr(self.bbox_head, 'use_degraded_cls_loss')
            and self.bbox_head.use_degraded_cls_loss)
        use_degraded_aux2 = (
            hasattr(self.bbox_head, 'use_degraded_aux2_loss')
            and self.bbox_head.use_degraded_aux2_loss)
        use_degraded_aux_head = (
            self.aux_heads is not None and any(
                getattr(aux_head, 'use_degraded_view', False)
                for aux_head in self.aux_heads))
        if use_degraded_cls or use_degraded_aux2 or use_degraded_aux_head:
            degraded_img = self._build_degraded_cls_view(img, img_metas)
            degraded_x = self.extract_feat(degraded_img)
            if use_degraded_cls:
                degraded_outs = self.bbox_head(degraded_x)
            if use_degraded_aux2:
                degraded_aux2_cls_scores = (
                    self.bbox_head.forward_degraded_aux2(degraded_x))
                degraded_aux2_amp_loss = (
                    self._compute_degraded_aux2_amp_loss(x, degraded_x))

        main_losses = self.bbox_head.loss(
            *main_outs,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore,
            equi_flip_outs=equi_flip_outs,
            equi_flip_dirs=equi_flip_dirs,
            photo_outs=photo_outs,
            degraded_outs=degraded_outs,
            degraded_aux2_cls_scores=degraded_aux2_cls_scores)
        losses.update(main_losses)
        if degraded_aux2_amp_loss is not None:
            losses['loss_degraded_aux2_amp'] = torch.nan_to_num(
                degraded_aux2_amp_loss, nan=0.0, posinf=0.0, neginf=0.0)

        # Mode A: Anchor-based 辅助头损失
        if self.aux_heads is not None:
            for i, aux_head in enumerate(self.aux_heads):
                use_aux_degraded = getattr(aux_head, 'use_degraded_view', False)
                aux_base_feats = degraded_x if use_aux_degraded else x
                aux_feats = self._build_aux_feats(aux_base_feats, aux_head)
                aux_kwargs = {}
                aux_sig_params = inspect.signature(
                    aux_head.forward_train).parameters
                if 'main_outs' in aux_sig_params:
                    aux_kwargs['main_outs'] = main_outs
                if 'main_bbox_head' in aux_sig_params:
                    aux_kwargs['main_bbox_head'] = self.bbox_head
                aux_losses = aux_head.forward_train(
                    aux_feats, img_metas, gt_bboxes, gt_labels,
                    gt_bboxes_ignore, **aux_kwargs)
                for k, v in aux_losses.items():
                    if isinstance(v, torch.Tensor):
                        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    elif isinstance(v, list):
                        v = [torch.nan_to_num(vi, nan=0.0, posinf=0.0, neginf=0.0)
                             if isinstance(vi, torch.Tensor) else vi for vi in v]
                    losses['aux{:d}_{:s}'.format(i, k)] = v
                if use_aux_degraded:
                    aux_amp = self._compute_degraded_aux2_amp_loss(
                        x, aux_base_feats, aux_head=aux_head)
                    losses['aux{:d}_loss_amp'.format(i)] = torch.nan_to_num(
                        aux_amp, nan=0.0, posinf=0.0, neginf=0.0)

        # Mode B: 高斯热图辅助头损失
        if self.gaussian_head is not None:
            heatmap = self.gaussian_head(x[0])  # 仅用 P3
            losses['loss_heatmap'] = self.gaussian_head.loss(
                heatmap, gt_bboxes, img_metas)

        # Mode C: UADH 辅助头损失
        if self.uadh_head is not None:
            uadh_pred = self.uadh_head(x[0])  # 仅用 P3, [N, 2, H, W]
            uadh_losses = self.uadh_head.loss(
                uadh_pred, gt_bboxes, img_metas, main_outs=main_outs,
                bbox_head=self.bbox_head)
            for k, v in uadh_losses.items():
                if isinstance(v, torch.Tensor):
                    v = torch.nan_to_num(v, nan=0.0, posinf=0.0,
                                         neginf=0.0)
                losses[k] = v

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """
        0.x 标准推理入口。
        辅助头天然不参与推理，零额外开销。
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def _build_equi_flip_view(self, img):
        """构建 L_equi 使用的翻转视图.

        输入已经经过 Normalize/Pad, 这里对网络输入张量做整数翻转；
        不生成翻转 GT, 因为 flip 图只用于角度一致性。
        """
        head = self.bbox_head
        probs = getattr(head, 'equi_flip_probs', (0.25, 0.25, 0.25))
        directions = ['horizontal', 'vertical', 'diagonal']

        flip_imgs = []
        flip_dirs = []

        B = img.size(0)
        for i in range(B):
            direction = random.choices(directions, weights=probs, k=1)[0]
            if direction == 'horizontal':
                flip_imgs.append(torch.flip(img[i:i + 1], dims=[3]))
            elif direction == 'vertical':
                flip_imgs.append(torch.flip(img[i:i + 1], dims=[2]))
            else:
                flip_imgs.append(torch.flip(img[i:i + 1], dims=[2, 3]))
            flip_dirs.append(direction)

        return torch.cat(flip_imgs, dim=0), flip_dirs

    def _build_photo_view(self, img, img_metas=None):
        """构建 L_invar 使用的 photometric 扰动视图 (GPU torch op).

        T_photo 参数基于 train+val 分布 P5-P95 (Route B: in-distribution).
        Spatial gradient 是主对抗源, gamma/ch_gain 是辅助.
        """
        from mmrotate.models.losses.angle_equi import build_photo_params, apply_t_photo

        # 安全检查: 确认 pipeline 的 Normalize 配置与 T_photo 硬编码值一致
        if img_metas is not None:
            norm_cfg = img_metas[0].get('img_norm_cfg', {})
            if norm_cfg:
                mean = list(norm_cfg.get('mean', []))
                std = list(norm_cfg.get('std', []))
                expected_mean = [123.675, 116.28, 103.53]
                expected_std = [58.395, 57.12, 57.375]
                if mean and std:
                    for i, (a, b) in enumerate(zip(mean, expected_mean)):
                        assert abs(a - b) < 0.01, \
                            f"T_photo Normalize mean[{i}]: pipeline={a}, expected={b}"
                    for i, (a, b) in enumerate(zip(std, expected_std)):
                        assert abs(a - b) < 0.01, \
                            f"T_photo Normalize std[{i}]: pipeline={a}, expected={b}"

        head = self.bbox_head
        params = build_photo_params(
            img.size(0), img.device,
            gamma_range=getattr(head, 'invar_gamma_range', (0.7, 1.5)),
            rg_range=getattr(head, 'invar_rg_range', (0.95, 1.40)),
            bg_range=getattr(head, 'invar_bg_range', (0.75, 1.05)),
            grad_lr_range=getattr(head, 'invar_grad_lr_range', (0.5, 2.0)),
            grad_ud_range=getattr(head, 'invar_grad_ud_range', (0.7, 1.5)),
            contrast_range=getattr(head, 'invar_contrast_range', (0.5, 2.0)),
        )
        return apply_t_photo(img, params)

    def _build_degraded_cls_view(self, img, img_metas=None):
        """构建低光/OOD 退化视图.

        输入是 Normalize 后的 RGB tensor。这里反归一化到 [0, 1]，
        执行 gamma 变暗、垂直照度梯度、通道色偏、对比度和噪声，再
        归一化回网络输入空间。该视图只用于 degraded-cls / aux2 前景
        鲁棒监督，不计算主头 bbox/angle/equi。
        """
        # 安全检查: 确认 pipeline 的 Normalize 配置与硬编码值一致
        if img_metas is not None:
            norm_cfg = img_metas[0].get('img_norm_cfg', {})
            if norm_cfg:
                mean = list(norm_cfg.get('mean', []))
                std = list(norm_cfg.get('std', []))
                expected_mean = [123.675, 116.28, 103.53]
                expected_std = [58.395, 57.12, 57.375]
                if mean and std:
                    for i, (a, b) in enumerate(zip(mean, expected_mean)):
                        assert abs(a - b) < 0.01, \
                            f"T_degrade Normalize mean[{i}]: pipeline={a}, expected={b}"
                    for i, (a, b) in enumerate(zip(std, expected_std)):
                        assert abs(a - b) < 0.01, \
                            f"T_degrade Normalize std[{i}]: pipeline={a}, expected={b}"

        head = self.bbox_head
        brightness_range = getattr(
            head, 'degraded_brightness_range', (0.4, 1.0))
        contrast_range = getattr(
            head, 'degraded_contrast_range', (0.6, 1.2))
        noise_std_range = getattr(
            head, 'degraded_noise_std_range', (0.0, 20.0))
        vertical_grad_range = getattr(
            head, 'degraded_vertical_grad_range', (1.0, 1.0))
        rg_range = getattr(head, 'degraded_rg_range', (1.0, 1.0))
        bg_range = getattr(head, 'degraded_bg_range', (1.0, 1.0))
        prob = float(getattr(head, 'degraded_prob', 0.5))

        B = img.size(0)
        mean = img.new_tensor([123.675, 116.28, 103.53]).view(
            1, 3, 1, 1) / 255.0
        std = img.new_tensor([58.395, 57.12, 57.375]).view(
            1, 3, 1, 1) / 255.0

        x = (img * std + mean).clamp(0.0, 1.0)

        on_mask = (torch.rand(B, 1, 1, 1, device=img.device) < prob).float()

        gamma = torch.empty(B, 1, 1, 1, device=img.device).uniform_(
            float(brightness_range[0]), float(brightness_range[1]))
        exponent = 1.0 / gamma.clamp(min=1e-6)
        x_gamma = x.clamp(0.0, 1.0).pow(exponent)

        # 物理先验 1: 上/下方向照度不均。默认范围为 (1,1) 时不生效,
        # 保持旧 degraded-cls 配置逐位兼容。
        grad_ratio = torch.empty(B, 1, 1, 1, device=img.device).uniform_(
            float(vertical_grad_range[0]), float(vertical_grad_range[1]))
        y = torch.linspace(0.0, 1.0, x.size(2), device=img.device).view(
            1, 1, x.size(2), 1)
        # ratio > 1 表示图像上侧更亮、下侧更暗；均值归一避免整体曝光重复计入。
        illum = grad_ratio + (1.0 - grad_ratio) * y
        illum = illum / illum.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        x_gamma = (x_gamma * illum).clamp(0.0, 1.0)

        # 物理先验 2: 港口夜间灯光/水面反射导致的 RGB 通道偏色。
        # 输入为 RGB, rg/bg 分别控制 R/G、B/G 的相对通道增益。
        rg = torch.empty(B, 1, 1, 1, device=img.device).uniform_(
            float(rg_range[0]), float(rg_range[1]))
        bg = torch.empty(B, 1, 1, 1, device=img.device).uniform_(
            float(bg_range[0]), float(bg_range[1]))
        ch_gain = torch.cat([rg, torch.ones_like(rg), bg], dim=1)
        x_gamma = (x_gamma * ch_gain).clamp(0.0, 1.0)

        contrast = torch.empty(B, 1, 1, 1, device=img.device).uniform_(
            float(contrast_range[0]), float(contrast_range[1]))
        mean_val = x_gamma.mean(dim=(1, 2, 3), keepdim=True)
        x_contrast = (x_gamma - mean_val) * contrast + mean_val

        noise_std = torch.empty(B, 1, 1, 1, device=img.device).uniform_(
            float(noise_std_range[0]), float(noise_std_range[1])) / 255.0
        if float(noise_std_range[1]) > 0:
            x_contrast = x_contrast + torch.randn_like(x_contrast) * noise_std

        x_degraded = x_contrast.clamp(0.0, 1.0)
        x = x_degraded * on_mask + x * (1.0 - on_mask)
        return (x - mean) / std

    def _compute_degraded_aux2_amp_loss(self, clean_feats, degraded_feats,
                                        aux_head=None):
        """退化特征到 clean teacher 的幅度域一致性.

        clean_feats 使用 detach, 只把退化视图特征拉向 clean 幅度谱, 不让
        clean 分支为了迁就退化样本而反向移动。这里不约束相位, 以降低
        对结构/角度敏感信息的干扰风险。
        """
        head = self.bbox_head if aux_head is None else aux_head
        weight = float(getattr(head, 'degraded_aux2_amp_loss_weight', 0.0))
        if weight <= 0:
            return clean_feats[0].sum() * 0.0

        levels = getattr(head, 'degraded_aux2_amp_levels', (0, 1, 2))
        losses = []
        for lvl in levels:
            lvl = int(lvl)
            if lvl < 0 or lvl >= len(clean_feats) or lvl >= len(degraded_feats):
                continue
            clean_amp = self._rfft_amplitude(clean_feats[lvl].detach().float())
            degraded_amp = self._rfft_amplitude(degraded_feats[lvl].float())
            denom = clean_amp.detach().abs().mean().clamp(min=1.0)
            losses.append(F.l1_loss(degraded_amp / denom, clean_amp / denom))

        if not losses:
            return clean_feats[0].sum() * 0.0

        return torch.stack(losses).mean() * weight

    @staticmethod
    def _rfft_amplitude(feat):
        """兼容新旧 PyTorch 的 2D FFT 幅度谱."""
        if hasattr(torch, 'fft') and hasattr(torch.fft, 'rfft2'):
            fft = torch.fft.rfft2(feat, norm='ortho')
            amp = torch.abs(fft)
        else:
            fft = torch.rfft(feat, signal_ndim=2, normalized=True,
                             onesided=True)
            amp = torch.sqrt(fft[..., 0].pow(2) + fft[..., 1].pow(2) + 1e-12)
        return torch.log1p(amp)
