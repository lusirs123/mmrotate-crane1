# mmrotate/models/detectors/sym_eood_detector.py
import copy
import inspect
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmrotate.models.builder import ROTATED_DETECTORS, build_head, build_loss
from mmrotate.core import build_assigner
from mmrotate.models.dense_heads.rotated_atss_head import RotatedATSSHead
from mmrotate.core import rbbox2result
from mmrotate.core.bbox.iou_calculators import RBboxOverlaps2D

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
                 platform_context_head=None,
                 platform_context_injector=None,
                 inject_aux_only=False,
                 aux_detach_cls_head=None,
                 reg_quality_head=None,
                 reg_quality_detach=True,
                 reg_quality_loss_weight=1.0,
                 reg_quality_focal_gamma=2.0,
                 reg_quality_min_target_iou=0.1,
                 reg_quality_pre_topk=10000,
                 pqa_head=None,
                 pqa_detach=True,
                 pqa_ld_loss_weight=1.5,
                 pqa_ld_gamma=2.0,
                 pqa_pre_topk=10000,
                 pqa_score_mode='quality',
                 pqa_grid_size=9,
                 pqa_quality_batch_size=512,
                 pqa_canonical_heatmap_level=None,
                 pqa_dark_supervision_weight=0.5,
                 pqa_dark_consistency_weight=0.1,
                 pqa_dark_warmup_iters=0,
                 pqa_dark_ramp_iters=0,
                 pqa_rank_loss_weight=0.0,
                 pqa_dark_rank_loss_weight=0.0,
                 pqa_rank_samples=128,
                 pqa_rank_mining_grid_size=5,
                 pqa_rank_min_iou_gap=0.10,
                 pqa_rank_score_margin=0.05,
                 pqa_rank_temperature=0.10,
                 pqa_dark_gamma_range=(0.5, 0.9),
                 pqa_dark_contrast_range=(0.7, 1.1),
                 pqa_dark_noise_std_range=(0.0, 10.0),
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

        # Platform context auxiliary supervision. Training-only; inference
        # remains the main head path in simple_test().
        if platform_context_head is not None:
            self.platform_context_head = build_head(platform_context_head)
        else:
            self.platform_context_head = None

        # Stage2 platform context feature modulation. This branch is allowed
        # to run in both train and test, but it only modulates FPN features.
        # Final outputs still come exclusively from the main beam bbox head.
        if platform_context_injector is not None:
            self.platform_context_injector = build_head(
                platform_context_injector)
        else:
            self.platform_context_injector = None

        # When True, context injection only applies to aux head features;
        # main head and all its variants (equi/photo/degraded) see clean
        # FPN features, and simple_test skips injection entirely.
        self.inject_aux_only = bool(inject_aux_only)

        # Detachable auxiliary classification head.
        # Features are detach()'d before entering this head during training,
        # so gradients NEVER flow back to backbone/FPN. Combined with strong
        # augmentations, it learns to output high cls confidence on frames
        # where the main head collapses (dark/blurry dead segments).
        # At inference: cls_fused = max(main_cls_logits, aux_cls_logits).
        self.aux_detach_cls_head = None
        self.aux_detach_loss_cls = None
        self.aux_detach_assigner = None
        self.aux_detach_pos_iou_thr = 0.5
        self.aux_detach_neg_iou_thr = 0.4
        self.aux_detach_min_pos_iou = 0.0
        # Strong augmentation params (configurable via detector attributes)
        self.aux_detach_gamma_range = (0.1, 0.8)
        self.aux_detach_blur_sigma_range = (0.5, 3.0)
        self.aux_detach_blur_kernel = 7
        self.aux_detach_noise_std_range = (0.0, 30.0)
        self.aux_detach_downscale_range = (0.5, 0.8)
        self.aux_detach_contrast_range = (0.5, 1.5)
        self.aux_detach_rg_range = (0.7, 1.3)
        self.aux_detach_bg_range = (0.7, 1.3)

        if aux_detach_cls_head is not None:
            cfg = copy.deepcopy(aux_detach_cls_head)
            loss_cfg = cfg.pop('loss_cls', None)
            assigner_cfg = cfg.pop('assigner', None)
            self._parse_aux_detach_params(cfg)
            self.aux_detach_cls_head = build_head(cfg)
            if loss_cfg is not None:
                self.aux_detach_loss_cls = build_loss(loss_cfg)
            if assigner_cfg is not None:
                self.aux_detach_assigner = build_assigner(assigner_cfg)

        # Independent localization-quality branch.  During training this head
        # receives detached FPN features and detached decoded main boxes.  Its
        # loss therefore updates only the quality-head parameters.  During
        # inference cls keeps a broad top-K pool and quality alone ranks it.
        self.reg_quality_head = None
        self.reg_quality_detach = bool(reg_quality_detach)
        self.reg_quality_loss_weight = float(reg_quality_loss_weight)
        self.reg_quality_focal_gamma = float(reg_quality_focal_gamma)
        self.reg_quality_min_target_iou = float(
            reg_quality_min_target_iou)
        self.reg_quality_pre_topk = int(reg_quality_pre_topk)
        if reg_quality_head is not None:
            if self.aux_detach_cls_head is not None:
                raise ValueError(
                    'reg_quality_head and aux_detach_cls_head are mutually '
                    'exclusive inference branches')
            self.reg_quality_head = build_head(
                copy.deepcopy(reg_quality_head))
            if self.reg_quality_head.num_anchors != self.bbox_head.num_anchors:
                raise ValueError(
                    'reg-quality/main anchor mismatch: quality={} main={}'
                    .format(self.reg_quality_head.num_anchors,
                            self.bbox_head.num_anchors))
            if self.reg_quality_pre_topk <= 0:
                raise ValueError('reg_quality_pre_topk must be positive')
            if self.reg_quality_loss_weight <= 0.0:
                raise ValueError('reg_quality_loss_weight must be positive')
            if self.reg_quality_focal_gamma < 0.0:
                raise ValueError('reg_quality_focal_gamma must be non-negative')
            if not 0.0 <= self.reg_quality_min_target_iou <= 1.0:
                raise ValueError(
                    'reg_quality_min_target_iou must be in [0, 1]')

        # Full pixel-level quality assessment (PQA).  Unlike the scalar
        # reg-quality head, this branch predicts a dense GT-relative heatmap;
        # every decoded candidate contributes its own OBB geometry when
        # Volume-IoU quality is calculated.  Classification remains unchanged.
        self.pqa_head = None
        self.pqa_detach = bool(pqa_detach)
        self.pqa_ld_loss_weight = float(pqa_ld_loss_weight)
        self.pqa_ld_gamma = float(pqa_ld_gamma)
        self.pqa_pre_topk = int(pqa_pre_topk)
        self.pqa_score_mode = str(pqa_score_mode)
        self.pqa_grid_size = int(pqa_grid_size)
        self.pqa_quality_batch_size = int(pqa_quality_batch_size)
        self.pqa_canonical_heatmap_level = (
            None if pqa_canonical_heatmap_level is None
            else int(pqa_canonical_heatmap_level))
        self.pqa_dark_supervision_weight = float(
            pqa_dark_supervision_weight)
        self.pqa_dark_consistency_weight = float(
            pqa_dark_consistency_weight)
        self.pqa_dark_warmup_iters = int(pqa_dark_warmup_iters)
        self.pqa_dark_ramp_iters = int(pqa_dark_ramp_iters)
        self.pqa_rank_loss_weight = float(pqa_rank_loss_weight)
        self.pqa_dark_rank_loss_weight = float(pqa_dark_rank_loss_weight)
        self.pqa_rank_samples = int(pqa_rank_samples)
        self.pqa_rank_mining_grid_size = int(pqa_rank_mining_grid_size)
        self.pqa_rank_min_iou_gap = float(pqa_rank_min_iou_gap)
        self.pqa_rank_score_margin = float(pqa_rank_score_margin)
        self.pqa_rank_temperature = float(pqa_rank_temperature)
        self.pqa_dark_gamma_range = tuple(pqa_dark_gamma_range)
        self.pqa_dark_contrast_range = tuple(pqa_dark_contrast_range)
        self.pqa_dark_noise_std_range = tuple(pqa_dark_noise_std_range)
        if pqa_head is not None:
            if self.reg_quality_head is not None:
                raise ValueError(
                    'pqa_head and reg_quality_head are mutually exclusive')
            if self.aux_detach_cls_head is not None:
                raise ValueError(
                    'pqa_head and aux_detach_cls_head are mutually exclusive')
            self.pqa_head = build_head(copy.deepcopy(pqa_head))
            if self.pqa_pre_topk <= 0:
                raise ValueError('pqa_pre_topk must be positive')
            if self.pqa_ld_loss_weight <= 0.0:
                raise ValueError('pqa_ld_loss_weight must be positive')
            if self.pqa_ld_gamma < 0.0:
                raise ValueError('pqa_ld_gamma must be non-negative')
            if self.pqa_score_mode not in ('quality', 'cls_x_quality'):
                raise ValueError(
                    'pqa_score_mode must be quality or cls_x_quality')
            if self.pqa_grid_size < 3:
                raise ValueError('pqa_grid_size must be at least 3')
            if self.pqa_quality_batch_size <= 0:
                raise ValueError('pqa_quality_batch_size must be positive')
            if (self.pqa_canonical_heatmap_level is not None
                    and self.pqa_canonical_heatmap_level < 0):
                raise ValueError(
                    'pqa_canonical_heatmap_level must be non-negative')
            if self.pqa_dark_supervision_weight < 0.0:
                raise ValueError(
                    'pqa_dark_supervision_weight must be non-negative')
            if self.pqa_dark_consistency_weight < 0.0:
                raise ValueError(
                    'pqa_dark_consistency_weight must be non-negative')
            if self.pqa_dark_warmup_iters < 0 or self.pqa_dark_ramp_iters < 0:
                raise ValueError('PQA dark warmup/ramp must be non-negative')
            if (self.pqa_rank_loss_weight < 0.0
                    or self.pqa_dark_rank_loss_weight < 0.0):
                raise ValueError('PQA rank weights must be non-negative')
            if self.pqa_rank_samples < 2:
                raise ValueError('pqa_rank_samples must be at least 2')
            if self.pqa_rank_mining_grid_size < 3:
                raise ValueError(
                    'pqa_rank_mining_grid_size must be at least 3')
            if self.pqa_rank_min_iou_gap < 0.0:
                raise ValueError('pqa_rank_min_iou_gap must be non-negative')
            if self.pqa_rank_score_margin < 0.0:
                raise ValueError('pqa_rank_score_margin must be non-negative')
            if self.pqa_rank_temperature <= 0.0:
                raise ValueError('pqa_rank_temperature must be positive')
            self.register_buffer(
                '_pqa_train_step', torch.zeros((), dtype=torch.long))

    def _build_aux_feats(self, feats, aux_head):
        if isinstance(aux_head, RotatedATSSHead):
            return [(feat, feat) for feat in feats]
        return feats

    def _parse_aux_detach_params(self, cfg):
        """Extract strong-augmentation and assignment params from config."""
        for key in ('pos_iou_thr', 'neg_iou_thr', 'min_pos_iou',
                     'gamma_range', 'blur_sigma_range', 'blur_kernel',
                     'noise_std_range', 'downscale_range',
                     'contrast_range', 'rg_range', 'bg_range'):
            cfg_key = 'aux_detach_' + key
            val = cfg.pop(cfg_key, None)
            if val is not None:
                setattr(self, cfg_key, val)

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

        # --- Score-level platform context modulation ---
        # Context head predicts platform activation maps on selected FPN
        # levels, trained independently via BCE+Dice. The maps are then
        # detached & sigmoided → additive spatial bias on cls logits
        # (before sigmoid) in the main bbox head. Feature modulation is
        # zero — geometry stays safe.
        platform_context_map = None
        if (self.platform_context_head is not None
                and self.bbox_head.use_score_context_modulation):
            context_logits = self.platform_context_head(x)
            # Compute context loss independently
            platform_losses = self.platform_context_head.loss(
                context_logits, img_metas, gt_bboxes)
            for k, v in platform_losses.items():
                if isinstance(v, torch.Tensor):
                    v = torch.nan_to_num(v, nan=0.0, posinf=0.0,
                                         neginf=0.0)
                losses[k] = v
            # Build platform_context_map: [B,1,H,W] tensors for modulated
            # levels, None for unmodulated levels (rest of FPN).
            ctx_levels = self.platform_context_head.levels
            platform_context_map = [None] * len(x)
            for idx, lvl in enumerate(ctx_levels):
                if lvl < len(x):
                    platform_context_map[lvl] = torch.sigmoid(
                        context_logits[idx].detach())

        # --- Context injection (legacy, mutually exclusive with score mod) ---
        # Old mode: inject x before main head (corrupts main head geometry)
        # Aux-only mode: inject only for aux head, main head gets clean x
        x_aux = x
        if self.platform_context_injector is not None:
            if self.inject_aux_only:
                x_aux, injector_losses = (
                    self.platform_context_injector.forward_train_features(
                        x, img_metas, gt_bboxes))
            else:
                x, injector_losses = (
                    self.platform_context_injector.forward_train_features(
                        x, img_metas, gt_bboxes))
                x_aux = x
            for k, v in injector_losses.items():
                if isinstance(v, torch.Tensor):
                    v = torch.nan_to_num(v, nan=0.0, posinf=0.0,
                                         neginf=0.0)
                losses[k] = v

        # 主头损失
        main_outs = self.bbox_head(x)

        # --- L_equi 路径 (翻转) ---
        equi_flip_outs = None
        equi_flip_dirs = None
        if (hasattr(self.bbox_head, 'use_equi_loss')
                and self.bbox_head.use_equi_loss):
            flip_img, equi_flip_dirs = self._build_equi_flip_view(img)
            flip_x = self.extract_feat(flip_img)
            if (self.platform_context_injector is not None
                    and not self.inject_aux_only):
                flip_x = self.platform_context_injector.forward_inject(
                    flip_x, train=True)
            equi_flip_outs = self.bbox_head(flip_x)

        # --- L_invar 路径 (光度) ---
        photo_outs = None
        if (hasattr(self.bbox_head, 'use_invar_loss')
                and self.bbox_head.use_invar_loss):
            photo_img = self._build_photo_view(img, img_metas)
            photo_x = self.extract_feat(photo_img)
            if (self.platform_context_injector is not None
                    and not self.inject_aux_only):
                photo_x = self.platform_context_injector.forward_inject(
                    photo_x, train=True)
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
            if (self.platform_context_injector is not None
                    and not self.inject_aux_only):
                degraded_x = self.platform_context_injector.forward_inject(
                    degraded_x, train=True)
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
            degraded_aux2_cls_scores=degraded_aux2_cls_scores,
            platform_context_map=platform_context_map)
        losses.update(main_losses)

        # --- Independent reg-quality training ---
        # Hard detach prevents the quality target/loss from weakening cls or
        # perturbing the already useful geometry branch.
        if self.reg_quality_head is not None:
            quality_feats = ([feat.detach() for feat in x]
                             if self.reg_quality_detach else x)
            quality_logits = self.reg_quality_head(quality_feats)
            quality_loss, quality_stats = self._compute_reg_quality_loss(
                quality_logits, main_outs[0], main_outs[1],
                img_metas, gt_bboxes)
            losses['loss_reg_quality'] = torch.nan_to_num(
                quality_loss, nan=0.0, posinf=0.0, neginf=0.0)
            for name, value in quality_stats.items():
                losses[name] = value

        # --- Full PQA heatmap training ---
        # Clean and dark PQA paths update only PQAHeatmapHead when detach is
        # enabled.  The paired dark view is spatially identical, so it reuses
        # exactly the same Gaussian labels without transforming any OBB.
        if self.pqa_head is not None:
            pqa_feats = ([feat.detach() for feat in x]
                         if self.pqa_detach else x)
            pqa_logits = self.pqa_head(pqa_feats)
            pqa_strides = self.bbox_head.anchor_generator.strides
            if self.pqa_canonical_heatmap_level is not None:
                level = self.pqa_canonical_heatmap_level
                if not 0 <= level < len(pqa_logits):
                    raise RuntimeError(
                        'pqa_canonical_heatmap_level is out of range')
                pqa_train_logits = (pqa_logits[level],)
                pqa_train_strides = (pqa_strides[level],)
            else:
                pqa_train_logits = pqa_logits
                pqa_train_strides = pqa_strides
            pqa_targets, pqa_valid = self.pqa_head.build_targets(
                pqa_train_logits, img_metas, gt_bboxes, pqa_train_strides)
            pqa_loss, pqa_stats = self.pqa_head.ld_loss(
                pqa_train_logits, pqa_targets, pqa_valid,
                gamma=self.pqa_ld_gamma,
                loss_weight=self.pqa_ld_loss_weight)
            losses['loss_pqa_ld'] = torch.nan_to_num(
                pqa_loss, nan=0.0, posinf=0.0, neginf=0.0)
            for name, value in pqa_stats.items():
                losses[name] = value

            # Dense heatmap fitting alone does not supervise the actual
            # top-1 decision.  Build detached decoded candidates once and
            # directly optimize their PQA quality ordering against RIoU.
            pqa_rank_batches = None
            if (self.pqa_rank_loss_weight > 0.0
                    or self.pqa_dark_rank_loss_weight > 0.0):
                pqa_rank_batches = self._build_pqa_rank_batches(
                    main_outs[0], main_outs[1], pqa_logits,
                    img_metas, gt_bboxes)
            if self.pqa_rank_loss_weight > 0.0:
                rank_loss, rank_stats = self._compute_pqa_rank_loss(
                    pqa_logits, pqa_rank_batches, img_metas,
                    loss_weight=self.pqa_rank_loss_weight)
                losses['loss_pqa_rank'] = torch.nan_to_num(
                    rank_loss, nan=0.0, posinf=0.0, neginf=0.0)
                for name, value in rank_stats.items():
                    losses[name] = value

            train_step = int(self._pqa_train_step.item())
            self._pqa_train_step.add_(1)
            if train_step < self.pqa_dark_warmup_iters:
                dark_factor = 0.0
            elif self.pqa_dark_ramp_iters > 0:
                dark_factor = min(
                    1.0,
                    float(train_step - self.pqa_dark_warmup_iters + 1)
                    / float(self.pqa_dark_ramp_iters))
            else:
                dark_factor = 1.0
            losses['pqa_dark_factor'] = pqa_loss.new_tensor(dark_factor)

            use_dark = (
                dark_factor > 0.0
                and (self.pqa_dark_supervision_weight > 0.0
                     or self.pqa_dark_consistency_weight > 0.0
                     or self.pqa_dark_rank_loss_weight > 0.0))
            if use_dark:
                pqa_dark_img = self._build_pqa_dark_view(img, img_metas)
                # No dark-view gradient may alter backbone/FPN geometry.
                with torch.no_grad():
                    pqa_dark_feats = self.extract_feat(pqa_dark_img)
                pqa_dark_logits = self.pqa_head(
                    [feat.detach() for feat in pqa_dark_feats])
                if self.pqa_canonical_heatmap_level is not None:
                    pqa_dark_train_logits = (
                        pqa_dark_logits[self.pqa_canonical_heatmap_level],)
                else:
                    pqa_dark_train_logits = pqa_dark_logits
                if self.pqa_dark_supervision_weight > 0.0:
                    dark_loss, _ = self.pqa_head.ld_loss(
                        pqa_dark_train_logits, pqa_targets, pqa_valid,
                        gamma=self.pqa_ld_gamma,
                        loss_weight=(self.pqa_ld_loss_weight
                                     * self.pqa_dark_supervision_weight
                                     * dark_factor))
                    losses['loss_pqa_dark_ld'] = torch.nan_to_num(
                        dark_loss, nan=0.0, posinf=0.0, neginf=0.0)
                if self.pqa_dark_consistency_weight > 0.0:
                    consistency = self.pqa_head.consistency_loss(
                        pqa_train_logits, pqa_dark_train_logits,
                        pqa_targets, pqa_valid,
                        loss_weight=(self.pqa_dark_consistency_weight
                                     * dark_factor))
                    losses['loss_pqa_dark_consistency'] = torch.nan_to_num(
                        consistency, nan=0.0, posinf=0.0, neginf=0.0)
                if self.pqa_dark_rank_loss_weight > 0.0:
                    dark_rank_loss, dark_rank_stats = (
                        self._compute_pqa_rank_loss(
                            pqa_dark_logits, pqa_rank_batches, img_metas,
                            loss_weight=(self.pqa_dark_rank_loss_weight
                                         * dark_factor)))
                    losses['loss_pqa_dark_rank'] = torch.nan_to_num(
                        dark_rank_loss, nan=0.0, posinf=0.0, neginf=0.0)
                    for name, value in dark_rank_stats.items():
                        losses['dark_' + name] = value
            else:
                zero_dark = pqa_train_logits[0].sum() * 0.0
                if self.pqa_dark_supervision_weight > 0.0:
                    losses['loss_pqa_dark_ld'] = zero_dark
                if self.pqa_dark_consistency_weight > 0.0:
                    losses['loss_pqa_dark_consistency'] = zero_dark
                if self.pqa_dark_rank_loss_weight > 0.0:
                    losses['loss_pqa_dark_rank'] = zero_dark
        if degraded_aux2_amp_loss is not None:
            losses['loss_degraded_aux2_amp'] = torch.nan_to_num(
                degraded_aux2_amp_loss, nan=0.0, posinf=0.0, neginf=0.0)

        # Mode A: Anchor-based 辅助头损失
        if self.aux_heads is not None:
            for i, aux_head in enumerate(self.aux_heads):
                use_aux_degraded = getattr(aux_head, 'use_degraded_view', False)
                # In aux_only mode, aux head gets injected features (x_aux);
                # otherwise it inherits the same (possibly injected) x.
                if self.inject_aux_only:
                    aux_base_feats = x_aux
                else:
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

        # (platform_context_head loss is now computed inline before
        # bbox_head.loss, with logits also used for score modulation.)

        # --- aux_detach_cls_head training ---
        # Strongly augmented view → backbone → FPN → detach() → aux_cls_head.
        # detach() is the HARD isolation: gradients from aux head NEVER flow
        # back to backbone/FPN. Main head continues normal training.
        if self.aux_detach_cls_head is not None:
            aug_img = self._build_strong_aug_view(img, img_metas)
            aug_x = self.extract_feat(aug_img)
            # HARD ISOLATION: cut gradient flow to backbone/FPN
            aug_x_detached = [feat.detach() for feat in aug_x]
            aux_cls_scores = self.aux_detach_cls_head(aug_x_detached)
            aux_loss = self._compute_aux_detach_loss(
                aux_cls_scores, img_metas, gt_bboxes, gt_labels)
            losses['loss_aux_detach_cls'] = torch.nan_to_num(
                aux_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """
        0.x 标准推理入口。
        辅助头天然不参与推理，零额外开销。
        Score-level context modulation: 在 cls logit 上施加加性空间偏置，
        FPN 特征不被调制 → 几何不受影响。

        AuxDetachClsHead: 推理时对同一张干净图 forward aux head，
        cls 分数做 element-wise max fusion: cls_fused = max(main, aux)。
        Bbox 始终来自主头，几何完全不受影响。
        """
        feat = self.extract_feat(img)

        # Legacy: feature-level injection (only if NOT inject_aux_only)
        if (self.platform_context_injector is not None
                and not self.inject_aux_only):
            feat = self.platform_context_injector.forward_test_features(feat)

        # Score-level context modulation: run context head, build
        # platform activation maps, pass to bbox head for cls logit bias.
        platform_context_map = None
        if (self.platform_context_head is not None
                and self.bbox_head.use_score_context_modulation):
            context_logits = self.platform_context_head(feat)
            ctx_levels = self.platform_context_head.levels
            platform_context_map = [None] * len(feat)
            for idx, lvl in enumerate(ctx_levels):
                if lvl < len(feat):
                    platform_context_map[lvl] = torch.sigmoid(
                        context_logits[idx])

        if self.pqa_head is not None:
            outs = self.bbox_head(feat)
            pqa_logits = self.pqa_head(feat)
            results_list = self._pqa_get_bboxes(
                outs[0], outs[1], pqa_logits, img_metas,
                rescale=rescale)
        elif self.reg_quality_head is not None:
            outs = self.bbox_head(feat)
            quality_logits = self.reg_quality_head(feat)
            results_list = self._reg_quality_primary_get_bboxes(
                outs[0], outs[1], quality_logits, img_metas,
                rescale=rescale)
        elif self.aux_detach_cls_head is not None:
            # Forward main head and aux head on the SAME clean features
            outs = self.bbox_head(feat)
            aux_cls_scores = self.aux_detach_cls_head(feat)

            # Score-level context modulation (if active)
            if (platform_context_map is not None
                    and self.bbox_head.use_score_context_modulation):
                gate = (self.bbox_head.score_context_gate_scale
                        * torch.sigmoid(self.bbox_head.score_context_gate_alpha))
                outs_cls = tuple(
                    cs + gate * pcm if pcm is not None else cs
                    for cs, pcm in zip(outs[0], platform_context_map))
            else:
                outs_cls = outs[0]

            # CLS FUSION: element-wise max in logit space
            #  max(logit_A, logit_B) respects monotonicity of sigmoid
            cls_fused = tuple(
                torch.maximum(cs, acs)
                for cs, acs in zip(outs_cls, aux_cls_scores))
            outs = (cls_fused, outs[1])

            results_list = self.bbox_head.get_bboxes(
                *outs, img_metas=img_metas, rescale=rescale, with_nms=False)
        else:
            results_list = self.bbox_head.simple_test(
                feat, img_metas, rescale=rescale,
                platform_context_map=platform_context_map)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def _compute_reg_quality_loss(self, quality_logits, cls_scores,
                                  bbox_preds, img_metas, gt_bboxes):
        """Supervise per-anchor quality with detached decoded-box RIoU.

        This is a separate Quality-Focal-style objective.  It does not alter
        the main classification target and, with ``reg_quality_detach=True``,
        cannot send gradients into FPN or the main bbox predictor.
        """
        if not (len(quality_logits) == len(cls_scores) == len(bbox_preds)):
            raise RuntimeError(
                'reg-quality level mismatch: quality={} cls={} bbox={}'.format(
                    len(quality_logits), len(cls_scores), len(bbox_preds)))

        featmap_sizes = [item.shape[-2:] for item in quality_logits]
        device = quality_logits[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, img_metas, device=device)

        zero = quality_logits[0].sum() * 0.0
        loss_sum = zero
        total_positive = 0
        target_sum = zero.detach()
        pred_positive_sum = zero.detach()

        for img_index, img_meta in enumerate(img_metas):
            anchors = torch.cat(anchor_list[img_index])
            valid = torch.cat(valid_flag_list[img_index]).bool()
            image_quality_logits = torch.cat([
                level[img_index].permute(1, 2, 0).reshape(-1)
                for level in quality_logits
            ])
            image_cls_logits = torch.cat([
                level[img_index].permute(1, 2, 0).reshape(
                    -1, self.bbox_head.cls_out_channels)
                for level in cls_scores
            ])
            image_bbox_preds = torch.cat([
                level[img_index].permute(1, 2, 0).reshape(-1, 5)
                for level in bbox_preds
            ])
            if not (anchors.shape[0] == image_quality_logits.numel()
                    == image_cls_logits.shape[0]
                    == image_bbox_preds.shape[0]):
                raise RuntimeError(
                    'reg-quality anchor alignment mismatch: anchors={} '
                    'quality={} cls={} bbox={}'.format(
                        anchors.shape[0], image_quality_logits.numel(),
                        image_cls_logits.shape[0], image_bbox_preds.shape[0]))

            with torch.no_grad():
                if self.bbox_head.use_sigmoid_cls:
                    pool_scores = image_cls_logits.sigmoid().max(dim=1).values
                else:
                    pool_scores = image_cls_logits.softmax(
                        dim=1)[:, :-1].max(dim=1).values
                pool_size = min(
                    self.reg_quality_pre_topk, int(pool_scores.numel()))
                pool_indices = pool_scores.topk(
                    pool_size, largest=True, sorted=False).indices
                supervision = torch.zeros_like(valid)
                supervision[pool_indices] = True
                supervision &= valid

                decoded = self.bbox_head.bbox_coder.decode(
                    anchors, image_bbox_preds.detach(),
                    max_shape=img_meta['img_shape'])
                max_size = float(max(img_meta.get(
                    'pad_shape', img_meta['img_shape'])[:2]))
                decoded = torch.nan_to_num(
                    decoded, nan=0.0, posinf=max_size, neginf=0.0)
                decoded[:, 0].clamp_(0.0, max_size)
                decoded[:, 1].clamp_(0.0, max_size)
                decoded[:, 2].clamp_(1.0, max_size)
                decoded[:, 3].clamp_(1.0, max_size)

                targets = decoded.new_zeros(decoded.shape[0])
                if (gt_bboxes[img_index].numel() > 0
                        and supervision.any()):
                    overlaps = RBboxOverlaps2D()(
                        decoded[supervision], gt_bboxes[img_index])
                    pool_targets = overlaps.max(dim=1).values.clamp_(0.0, 1.0)
                    pool_targets[pool_targets <
                                 self.reg_quality_min_target_iou] = 0.0
                    targets[supervision] = pool_targets

            probabilities = image_quality_logits.sigmoid()
            element_loss = F.binary_cross_entropy_with_logits(
                image_quality_logits, targets, reduction='none')
            focal_weight = (targets - probabilities).abs().pow(
                self.reg_quality_focal_gamma)
            element_loss = (
                element_loss * focal_weight * supervision.float())
            loss_sum = loss_sum + element_loss.sum()

            positive = (targets > 0.0) & supervision
            positive_count = int(positive.sum().item())
            total_positive += positive_count
            if positive_count:
                target_sum = target_sum + targets[positive].sum()
                pred_positive_sum = (
                    pred_positive_sum + probabilities[positive].detach().sum())

        avg_factor = max(total_positive, 1)
        loss = (loss_sum / float(avg_factor)
                * self.reg_quality_loss_weight)
        stats = dict(
            reg_quality_positive=zero.new_tensor(float(total_positive)),
            reg_quality_target_mean=(target_sum / float(avg_factor)),
            reg_quality_pred_positive_mean=(
                pred_positive_sum / float(avg_factor)),
        )
        return loss, stats

    def _build_pqa_rank_batches(self, cls_scores, bbox_preds, heatmap_logits,
                                img_metas, gt_bboxes):
        """Build a small, detached ranking pool from inference candidates.

        Each image mixes the best-RIoU candidates, current PQA false-peak
        candidates, high-classification candidates, and low-RIoU negatives
        from the same cls-topK pool used by inference.  PQA hard mining uses
        a cheap coarse grid over all candidates; gradients use the configured
        full grid only on the selected subset.
        """
        if not (len(cls_scores) == len(bbox_preds) == len(heatmap_logits)):
            raise RuntimeError('PQA rank cls/bbox/heatmap level mismatch')
        featmap_sizes = [item.shape[-2:] for item in cls_scores]
        anchors_per_level = self.bbox_head.anchor_generator.grid_priors(
            featmap_sizes, device=cls_scores[0].device)
        batches = []

        with torch.no_grad():
            for img_index, img_meta in enumerate(img_metas):
                all_boxes = []
                all_scores = []
                all_levels = []
                for level_index, (cls_level, bbox_level, anchors) in enumerate(
                        zip(cls_scores, bbox_preds, anchors_per_level)):
                    cls_flat = cls_level[img_index].detach().permute(
                        1, 2, 0).reshape(
                            -1, self.bbox_head.cls_out_channels)
                    if self.bbox_head.use_sigmoid_cls:
                        scores = cls_flat.sigmoid().max(dim=1).values
                    else:
                        scores = cls_flat.softmax(
                            dim=-1)[:, :-1].max(dim=1).values
                    scores = torch.nan_to_num(
                        scores, nan=0.0, posinf=1.0, neginf=0.0)
                    bbox_flat = bbox_level[img_index].detach().permute(
                        1, 2, 0).reshape(-1, 5)
                    if not (anchors.shape[0] == bbox_flat.shape[0]
                            == scores.numel()):
                        raise RuntimeError(
                            'PQA rank anchor alignment mismatch')
                    decoded = self.bbox_head.bbox_coder.decode(
                        anchors, bbox_flat, max_shape=img_meta['img_shape'])
                    max_size = float(max(img_meta.get(
                        'pad_shape', img_meta['img_shape'])[:2]))
                    decoded = torch.nan_to_num(
                        decoded, nan=0.0, posinf=max_size, neginf=0.0)
                    decoded[:, 0].clamp_(0.0, max_size)
                    decoded[:, 1].clamp_(0.0, max_size)
                    decoded[:, 2].clamp_(1.0, max_size)
                    decoded[:, 3].clamp_(1.0, max_size)
                    all_boxes.append(decoded)
                    all_scores.append(scores)
                    all_levels.append(torch.full(
                        (decoded.shape[0],), int(level_index),
                        dtype=torch.long, device=decoded.device))

                boxes = torch.cat(all_boxes)
                scores = torch.cat(all_scores)
                levels = torch.cat(all_levels)
                pool_size = min(self.pqa_pre_topk, int(scores.numel()))
                pool_indices = scores.topk(
                    pool_size, largest=True, sorted=False).indices
                pool_boxes = boxes[pool_indices]
                pool_scores = scores[pool_indices]
                pool_levels = levels[pool_indices]

                if gt_bboxes[img_index].numel() == 0 or pool_size < 2:
                    batches.append(dict(
                        boxes=pool_boxes[:0], levels=pool_levels[:0],
                        target_ious=pool_scores[:0]))
                    continue
                overlaps = RBboxOverlaps2D()(
                    pool_boxes, gt_bboxes[img_index].detach())
                target_ious = overlaps.max(dim=1).values.clamp(0.0, 1.0)

                image_heatmaps = tuple(
                    level[img_index:img_index + 1]
                    for level in heatmap_logits)
                pad_shape = img_meta.get('pad_shape', img_meta['img_shape'])
                mining_quality = self.pqa_head.quality_from_boxes(
                    image_heatmaps, pool_boxes, pool_levels, pad_shape,
                    grid_size=self.pqa_rank_mining_grid_size,
                    batch_size=self.pqa_quality_batch_size,
                    canonical_level=self.pqa_canonical_heatmap_level)
                mining_quality = torch.nan_to_num(
                    mining_quality, nan=0.0, posinf=1.0, neginf=0.0)

                sample_count = min(self.pqa_rank_samples, pool_size)
                iou_order = target_ious.argsort(descending=True)
                pqa_order = mining_quality.argsort(descending=True)
                cls_order = pool_scores.argsort(descending=True)
                high_budget = max(sample_count // 2, 1)
                pqa_budget = max(sample_count // 4, 1)
                cls_budget = max(sample_count // 8, 1)
                low_budget = max(
                    sample_count - high_budget - pqa_budget - cls_budget, 1)
                used = torch.zeros(
                    pool_size, dtype=torch.bool, device=pool_boxes.device)
                selected_parts = []

                def add_unique(order, budget):
                    fresh = order[~used[order]][:int(budget)]
                    if fresh.numel() > 0:
                        used[fresh] = True
                        selected_parts.append(fresh)

                add_unique(iou_order, high_budget)
                add_unique(pqa_order, pqa_budget)
                add_unique(cls_order, cls_budget)
                add_unique(iou_order.flip(0), low_budget)
                already = sum(int(part.numel()) for part in selected_parts)
                if already < sample_count:
                    add_unique(iou_order, sample_count - already)
                selected = torch.cat(selected_parts)[:sample_count]
                batches.append(dict(
                    boxes=pool_boxes[selected].detach(),
                    levels=pool_levels[selected].detach(),
                    target_ious=target_ious[selected].detach()))
        return batches

    def _compute_pqa_rank_loss(self, heatmap_logits, rank_batches, img_metas,
                               loss_weight):
        """Apply decoded-candidate ordering loss to clean or dark heatmaps."""
        if rank_batches is None or len(rank_batches) != len(img_metas):
            raise RuntimeError('PQA rank batches must align with image metas')
        zero = heatmap_logits[0].sum() * 0.0
        image_losses = []
        pair_total = 0.0
        accuracy_sum = zero.detach()
        for img_index, (batch, img_meta) in enumerate(
                zip(rank_batches, img_metas)):
            if batch['boxes'].shape[0] < 2:
                continue
            image_heatmaps = tuple(
                level[img_index:img_index + 1]
                for level in heatmap_logits)
            pad_shape = img_meta.get('pad_shape', img_meta['img_shape'])
            qualities = self.pqa_head.quality_from_boxes(
                image_heatmaps, batch['boxes'], batch['levels'], pad_shape,
                grid_size=self.pqa_grid_size,
                batch_size=self.pqa_quality_batch_size,
                canonical_level=self.pqa_canonical_heatmap_level)
            image_loss, image_stats = self.pqa_head.pairwise_rank_loss(
                qualities, batch['target_ious'],
                min_iou_gap=self.pqa_rank_min_iou_gap,
                score_margin=self.pqa_rank_score_margin,
                temperature=self.pqa_rank_temperature,
                loss_weight=loss_weight)
            pairs = float(image_stats['pqa_rank_pairs'].item())
            if pairs > 0.0:
                image_losses.append(image_loss)
                pair_total += pairs
                accuracy_sum = (accuracy_sum
                                + image_stats['pqa_rank_accuracy'] * pairs)
        if image_losses:
            loss = torch.stack(image_losses).mean()
            accuracy = accuracy_sum / max(pair_total, 1.0)
        else:
            loss = zero
            accuracy = zero.detach()
        return loss, dict(
            pqa_rank_pairs=zero.new_tensor(pair_total),
            pqa_rank_accuracy=accuracy)

    def _pqa_get_bboxes(self, cls_scores, bbox_preds, heatmap_logits,
                        img_metas, rescale=False):
        """Rank cls-topK decoded candidates by PQA Volume-IoU.

        ``quality`` is the task-adapted quality-primary mode selected by the
        oracle gate. ``cls_x_quality`` is the faithful paper baseline and can
        be evaluated from the same checkpoint without retraining.
        """
        if not (len(cls_scores) == len(bbox_preds) == len(heatmap_logits)):
            raise RuntimeError('PQA inference level mismatch')
        featmap_sizes = [item.shape[-2:] for item in cls_scores]
        anchors_per_level = self.bbox_head.anchor_generator.grid_priors(
            featmap_sizes, device=cls_scores[0].device)
        num_imgs = cls_scores[0].shape[0]
        results = []

        for img_index in range(num_imgs):
            all_boxes = []
            all_cls_scores = []
            all_labels = []
            all_levels = []
            for level_index, (cls_level, bbox_level, anchors) in enumerate(
                    zip(cls_scores, bbox_preds, anchors_per_level)):
                cls_flat = cls_level[img_index].permute(
                    1, 2, 0).reshape(-1, self.bbox_head.cls_out_channels)
                if self.bbox_head.use_sigmoid_cls:
                    class_prob = cls_flat.sigmoid()
                else:
                    class_prob = cls_flat.softmax(dim=-1)[:, :-1]
                class_prob = torch.nan_to_num(
                    class_prob, nan=0.0, posinf=1.0, neginf=0.0)
                max_cls, labels = class_prob.max(dim=1)
                bbox_flat = bbox_level[img_index].permute(
                    1, 2, 0).reshape(-1, 5)
                if not (anchors.shape[0] == bbox_flat.shape[0]
                        == max_cls.numel()):
                    raise RuntimeError('PQA anchor alignment mismatch')
                decoded = self.bbox_head.bbox_coder.decode(
                    anchors, bbox_flat,
                    max_shape=img_metas[img_index]['img_shape'])
                max_size = float(max(img_metas[img_index].get(
                    'pad_shape', img_metas[img_index]['img_shape'])[:2]))
                decoded = torch.nan_to_num(
                    decoded, nan=0.0, posinf=max_size, neginf=0.0)
                decoded[:, 0].clamp_(0.0, max_size)
                decoded[:, 1].clamp_(0.0, max_size)
                decoded[:, 2].clamp_(1.0, max_size)
                decoded[:, 3].clamp_(1.0, max_size)
                all_boxes.append(decoded)
                all_cls_scores.append(max_cls)
                all_labels.append(labels)
                all_levels.append(torch.full(
                    (decoded.shape[0],), int(level_index),
                    dtype=torch.long, device=decoded.device))

            boxes = torch.cat(all_boxes)
            class_scores = torch.cat(all_cls_scores)
            labels = torch.cat(all_labels)
            levels = torch.cat(all_levels)
            pre_topk = min(self.pqa_pre_topk, class_scores.numel())
            _, pre_indices = class_scores.topk(
                pre_topk, largest=True, sorted=False)
            pool_boxes = boxes[pre_indices]
            pool_cls = class_scores[pre_indices]
            pool_labels = labels[pre_indices]
            pool_levels = levels[pre_indices]
            image_heatmaps = tuple(
                level[img_index:img_index + 1]
                for level in heatmap_logits)
            pad_shape = img_metas[img_index].get(
                'pad_shape', img_metas[img_index]['img_shape'])
            quality = self.pqa_head.quality_from_boxes(
                image_heatmaps, pool_boxes, pool_levels, pad_shape,
                grid_size=self.pqa_grid_size,
                batch_size=self.pqa_quality_batch_size,
                canonical_level=self.pqa_canonical_heatmap_level)
            quality = torch.nan_to_num(
                quality, nan=0.0, posinf=1.0, neginf=0.0)
            if self.pqa_score_mode == 'quality':
                ranking_scores = quality
            elif self.pqa_score_mode == 'cls_x_quality':
                ranking_scores = pool_cls * quality
            else:
                raise RuntimeError(
                    'Unsupported pqa_score_mode: ' + self.pqa_score_mode)

            max_per_img = int(self.test_cfg.get('max_per_img', 1))
            keep_count = min(max(max_per_img, 1), pre_topk)
            selected_scores, selected = ranking_scores.topk(
                keep_count, largest=True, sorted=True)
            selected_boxes = pool_boxes[selected]
            selected_labels = pool_labels[selected]
            if rescale and selected_boxes.numel() > 0:
                scale_factor = selected_boxes.new_tensor(
                    img_metas[img_index]['scale_factor'])
                selected_boxes[:, :4] /= scale_factor
            det_bboxes = torch.cat(
                [selected_boxes, selected_scores[:, None]], dim=1)
            results.append((det_bboxes, selected_labels))
        return results

    def _reg_quality_primary_get_bboxes(self, cls_scores, bbox_preds,
                                        quality_logits, img_metas,
                                        rescale=False):
        """Select quality top-1 inside the pre-threshold cls-topK pool."""
        if not (len(cls_scores) == len(bbox_preds) == len(quality_logits)):
            raise RuntimeError('reg-quality inference level mismatch')
        featmap_sizes = [item.shape[-2:] for item in cls_scores]
        anchors_per_level = self.bbox_head.anchor_generator.grid_priors(
            featmap_sizes, device=cls_scores[0].device)
        num_imgs = cls_scores[0].shape[0]
        results = []

        for img_index in range(num_imgs):
            all_boxes = []
            all_cls_scores = []
            all_labels = []
            all_quality = []
            for cls_level, bbox_level, quality_level, anchors in zip(
                    cls_scores, bbox_preds, quality_logits,
                    anchors_per_level):
                cls_flat = cls_level[img_index].permute(
                    1, 2, 0).reshape(-1, self.bbox_head.cls_out_channels)
                if self.bbox_head.use_sigmoid_cls:
                    class_prob = cls_flat.sigmoid()
                else:
                    class_prob = cls_flat.softmax(dim=-1)[:, :-1]
                class_prob = torch.nan_to_num(
                    class_prob, nan=0.0, posinf=1.0, neginf=0.0)
                max_cls, labels = class_prob.max(dim=1)
                bbox_flat = bbox_level[img_index].permute(
                    1, 2, 0).reshape(-1, 5)
                quality_flat = quality_level[img_index].permute(
                    1, 2, 0).reshape(-1).sigmoid()
                quality_flat = torch.nan_to_num(
                    quality_flat, nan=0.0, posinf=1.0, neginf=0.0)
                if not (anchors.shape[0] == bbox_flat.shape[0]
                        == max_cls.numel() == quality_flat.numel()):
                    raise RuntimeError(
                        'reg-quality inference anchor alignment mismatch')
                decoded = self.bbox_head.bbox_coder.decode(
                    anchors, bbox_flat,
                    max_shape=img_metas[img_index]['img_shape'])
                max_size = float(max(img_metas[img_index].get(
                    'pad_shape', img_metas[img_index]['img_shape'])[:2]))
                decoded = torch.nan_to_num(
                    decoded, nan=0.0, posinf=max_size, neginf=0.0)
                decoded[:, 0].clamp_(0.0, max_size)
                decoded[:, 1].clamp_(0.0, max_size)
                decoded[:, 2].clamp_(1.0, max_size)
                decoded[:, 3].clamp_(1.0, max_size)
                all_boxes.append(decoded)
                all_cls_scores.append(max_cls)
                all_labels.append(labels)
                all_quality.append(quality_flat)

            boxes = torch.cat(all_boxes)
            class_scores = torch.cat(all_cls_scores)
            labels = torch.cat(all_labels)
            quality = torch.cat(all_quality)

            pre_topk = min(self.reg_quality_pre_topk, class_scores.numel())
            _, pre_indices = class_scores.topk(
                pre_topk, largest=True, sorted=False)
            pool_quality = quality[pre_indices]
            max_per_img = int(self.test_cfg.get('max_per_img', 1))
            keep_count = min(max(max_per_img, 1), pre_topk)
            selected_quality, selected_in_pool = pool_quality.topk(
                keep_count, largest=True, sorted=True)
            selected = pre_indices[selected_in_pool]
            selected_boxes = boxes[selected]
            selected_labels = labels[selected]

            if rescale and selected_boxes.numel() > 0:
                scale_factor = selected_boxes.new_tensor(
                    img_metas[img_index]['scale_factor'])
                selected_boxes[:, :4] /= scale_factor
            det_bboxes = torch.cat(
                [selected_boxes, selected_quality[:, None]], dim=1)
            results.append((det_bboxes, selected_labels))
        return results

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

    def _build_pqa_dark_view(self, img, img_metas=None):
        """Build a geometry-preserving dark view for PQA consistency.

        Only illumination, contrast, and sensor noise are changed.  There is
        no crop, resize, blur, or flip, so clean-view Gaussian labels remain
        exactly aligned with the dark view.  The caller extracts dark FPN
        features under ``torch.no_grad`` to preserve the main detector.
        """
        if img_metas is not None:
            norm_cfg = img_metas[0].get('img_norm_cfg', {})
            if norm_cfg:
                expected_mean = [123.675, 116.28, 103.53]
                expected_std = [58.395, 57.12, 57.375]
                mean_cfg = list(norm_cfg.get('mean', []))
                std_cfg = list(norm_cfg.get('std', []))
                if mean_cfg and std_cfg:
                    for actual, expected in zip(mean_cfg, expected_mean):
                        assert abs(actual - expected) < 0.01
                    for actual, expected in zip(std_cfg, expected_std):
                        assert abs(actual - expected) < 0.01

        mean = img.new_tensor([123.675, 116.28, 103.53]).view(
            1, 3, 1, 1) / 255.0
        std = img.new_tensor([58.395, 57.12, 57.375]).view(
            1, 3, 1, 1) / 255.0
        x = (img * std + mean).clamp(0.0, 1.0)
        batch = x.shape[0]
        gamma = torch.empty(
            batch, 1, 1, 1, device=x.device).uniform_(
                float(self.pqa_dark_gamma_range[0]),
                float(self.pqa_dark_gamma_range[1]))
        x = x.clamp(min=1e-6).pow(1.0 / gamma.clamp(min=1e-6))

        contrast = torch.empty(
            batch, 1, 1, 1, device=x.device).uniform_(
                float(self.pqa_dark_contrast_range[0]),
                float(self.pqa_dark_contrast_range[1]))
        image_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - image_mean) * contrast + image_mean

        noise_std = torch.empty(
            batch, 1, 1, 1, device=x.device).uniform_(
                float(self.pqa_dark_noise_std_range[0]),
                float(self.pqa_dark_noise_std_range[1])) / 255.0
        if float(self.pqa_dark_noise_std_range[1]) > 0.0:
            x = x + torch.randn_like(x) * noise_std
        x = x.clamp(0.0, 1.0)
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

    # ================================================================
    # aux_detach_cls_head: training + inference support
    # ================================================================

    def _build_strong_aug_view(self, img, img_metas=None):
        """Build strongly augmented view for aux_detach_cls_head training.

        Input is a Normalize'd RGB tensor. We undo normalization, apply
        aggressive augmentations on [0,1] range, then re-normalize.

        Pipeline:
          1. Gamma dimming (brightness 0.1-0.8) → simulates very dark frames
          2. Gaussian blur (sigma 0.5-3.0) → simulates motion blur / defocus
          3. Downscale + upsample (0.5-0.8×) → simulates small-target appearance
          4. Gaussian noise (0-30/255) → simulates sensor noise in low light
          5. Contrast jitter (0.5-1.5)
          6. Channel gain (R/G, B/G jitter)

        All ops run on GPU. No augmentation on validation behaves as identity.
        """
        mean = img.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1) / 255.0
        std = img.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1) / 255.0

        # Denormalize to [0, 1]
        x = (img * std + mean).clamp(0.0, 1.0)
        B, C, H, W = x.shape

        # 1. Gamma dimming
        gamma = torch.empty(B, 1, 1, 1, device=img.device).uniform_(
            float(self.aux_detach_gamma_range[0]),
            float(self.aux_detach_gamma_range[1]))
        x = x.clamp(min=1e-6).pow(1.0 / gamma.clamp(min=1e-6))

        # 2. Gaussian blur (applied to all images with prob=1.0 since we
        #    explicitly want the aux head to handle blurry frames)
        blur_sigma = torch.empty(B, device=img.device).uniform_(
            float(self.aux_detach_blur_sigma_range[0]),
            float(self.aux_detach_blur_sigma_range[1]))
        for i in range(B):
            if blur_sigma[i] > 0.1:
                xi = x[i:i + 1]
                kernel_size = int(self.aux_detach_blur_kernel)
                x[i:i + 1] = self._gaussian_blur_2d(
                    xi, kernel_size, blur_sigma[i].item())

        # 3. Downscale + upsample
        downscale = torch.empty(B, 1, 1, 1, device=img.device).uniform_(
            float(self.aux_detach_downscale_range[0]),
            float(self.aux_detach_downscale_range[1]))
        for i in range(B):
            scale = downscale[i].item()
            if scale < 0.95:
                new_h = max(int(H * scale), 16)
                new_w = max(int(W * scale), 16)
                xi_down = F.interpolate(
                    x[i:i + 1], size=(new_h, new_w),
                    mode='bilinear', align_corners=False)
                x[i:i + 1] = F.interpolate(
                    xi_down, size=(H, W),
                    mode='bilinear', align_corners=False)

        # 4. Gaussian noise
        noise_std = torch.empty(B, 1, 1, 1, device=img.device).uniform_(
            float(self.aux_detach_noise_std_range[0]),
            float(self.aux_detach_noise_std_range[1])) / 255.0
        if float(self.aux_detach_noise_std_range[1]) > 0:
            x = x + torch.randn_like(x) * noise_std

        # 5. Contrast
        contrast = torch.empty(B, 1, 1, 1, device=img.device).uniform_(
            float(self.aux_detach_contrast_range[0]),
            float(self.aux_detach_contrast_range[1]))
        mean_val = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean_val) * contrast + mean_val

        # 6. Channel gain (R/G, B/G jitter)
        rg = torch.empty(B, 1, 1, 1, device=img.device).uniform_(
            float(self.aux_detach_rg_range[0]),
            float(self.aux_detach_rg_range[1]))
        bg = torch.empty(B, 1, 1, 1, device=img.device).uniform_(
            float(self.aux_detach_bg_range[0]),
            float(self.aux_detach_bg_range[1]))
        ch_gain = torch.cat([rg, torch.ones_like(rg), bg], dim=1)
        x = (x * ch_gain).clamp(0.0, 1.0)

        x = x.clamp(0.0, 1.0)
        return (x - mean) / std

    @staticmethod
    def _gaussian_blur_2d(x, kernel_size, sigma):
        """Apply 2D Gaussian blur on GPU using depthwise convolution.

        Args:
            x: [1, C, H, W] input tensor.
            kernel_size: int, odd kernel size.
            sigma: float, Gaussian sigma.

        Returns:
            [1, C, H, W] blurred tensor.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        # Build 1D Gaussian kernel
        ax = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
        ax = ax - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        # Build 2D kernel
        kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]  # [K, K]
        kernel_2d = kernel_2d.expand(x.size(1), 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        return F.conv2d(x, kernel_2d, padding=padding, groups=x.size(1))

    @staticmethod
    def _downscale_upscale(x, scale):
        """Downscale then bilinear upsample back to original size.

        Args:
            x: [1, C, H, W] input tensor.
            scale: float in (0, 1), scale factor.

        Returns:
            [1, C, H, W] processed tensor.
        """
        _, _, H_orig, W_orig = x.shape
        new_h = max(int(H_orig * scale), 16)
        new_w = max(int(W_orig * scale), 16)
        x_down = F.interpolate(x, size=(new_h, new_w),
                               mode='bilinear', align_corners=False)
        x_up = F.interpolate(x_down, size=(H_orig, W_orig),
                             mode='bilinear', align_corners=False)
        return x_up

    def _compute_aux_detach_loss(self, aux_cls_scores, img_metas,
                                  gt_bboxes, gt_labels):
        """Compute FocalLoss for aux_detach_cls_head.

        Uses simple IoU-based anchor assignment (independent of main head's
        prediction-aware SymPOLA). This is sufficient for binary FG/BG
        classification on the strongly augmented view.

        Args:
            aux_cls_scores: tuple of [B, C_anchors, H, W] tensors per level.
            img_metas, gt_bboxes, gt_labels: standard detection targets.

        Returns:
            Scalar loss tensor.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in aux_cls_scores]
        device = aux_cls_scores[0].device
        num_imgs = len(img_metas)

        # Generate anchors (same anchor generator as main head)
        anchor_list, _ = self.bbox_head.get_anchors(
            featmap_sizes, img_metas, device=device)

        num_levels = len(aux_cls_scores)
        cls_out_channels = self.aux_detach_cls_head.cls_out_channels

        # Flatten aux_cls_scores per image, per level
        flatten_scores = []
        for i in range(num_imgs):
            img_scores = []
            for lvl in range(num_levels):
                s = aux_cls_scores[lvl][i].permute(1, 2, 0).reshape(
                    -1, cls_out_channels)
                img_scores.append(s)
            flatten_scores.append(torch.cat(img_scores))

        # IoU-based binary assignment per image
        iou_calc = RBboxOverlaps2D()
        pos_iou = float(self.aux_detach_pos_iou_thr)
        neg_iou = float(self.aux_detach_neg_iou_thr)
        num_classes = self.bbox_head.num_classes

        total_loss = 0.0
        total_pos = 0

        for i in range(num_imgs):
            flat_anchors = torch.cat(anchor_list[i])
            scores_i = flatten_scores[i]
            gt_i = gt_bboxes[i]
            gt_labels_i = gt_labels[i]

            if gt_i.size(0) == 0:
                # No GT: all anchors are negative
                labels = flat_anchors.new_full(
                    (flat_anchors.size(0),), num_classes, dtype=torch.long)
                label_weights = flat_anchors.new_zeros(flat_anchors.size(0))
                pos_count = 0
            else:
                overlaps = iou_calc(flat_anchors, gt_i)  # [N_anchors, N_gt]
                max_overlaps, argmax_overlaps = overlaps.max(dim=1)

                pos_mask = max_overlaps >= pos_iou
                neg_mask = max_overlaps < neg_iou

                labels = flat_anchors.new_full(
                    (flat_anchors.size(0),), num_classes, dtype=torch.long)
                labels[pos_mask] = gt_labels_i[argmax_overlaps[pos_mask]]

                label_weights = max_overlaps.new_zeros(flat_anchors.size(0))
                label_weights[pos_mask] = 1.0
                label_weights[neg_mask] = 1.0

                pos_count = int(pos_mask.sum().item())
                total_pos += pos_count

            loss_i = self.aux_detach_loss_cls(
                scores_i,
                labels,
                weight=label_weights,
                avg_factor=max(pos_count, 1))
            total_loss = total_loss + loss_i

        return total_loss / max(num_imgs, 1)
