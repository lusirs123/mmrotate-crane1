# mmrotate/models/dense_heads/sym_eood_head.py
# #先从 backbone 和 FPN 得到多层特征，对每一层输出分类分数和回归偏移。
# 在 loss 里先生成整张图对应的 anchors。
# 把每层的 cls 和 bbox 预测展平，按 image-first 的方式拼起来。
# 进入 get_targets，再按图像调用 _get_targets_single。
# _get_targets_single 里先把 anchors 过滤成有效区域，然后把 bbox_pred 解码成物理框。
# 接着调用 assigner 做分配，得到每个 anchor 的正负样本、目标框和标签。
# 再用 sampler 取正负样本，构造 labels、label_weights、bbox_targets、bbox_weights。
# 回到 loss_single，逐层把 cls_score 和 bbox_pred 展平。
# 回归分支把预测框和目标框都解码到物理空间。
# 分类分支用 SymNFLLoss 计算分类损失，回归分支用 SymKLDLoss 计算回归损失。
# 所有层的 loss 汇总成 loss_cls 和 loss_bbox 返回。

import torch
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import images_to_levels, multi_apply, unmap
from mmrotate.models.builder import ROTATED_HEADS
from mmrotate.models.dense_heads.rotated_retina_head import RotatedRetinaHead


@ROTATED_HEADS.register_module(force=True)
class SymEOODHead(RotatedRetinaHead):
    """
    SymEOOD 检测头.

    可选 L_equi:
      use_equi_loss (bool): 是否启用翻转等变一致性损失. 默认 False.
      equi_loss_weight (float): L_equi 权重 λ1. 默认 0.2.
      equi_flip_probs (tuple): 翻转方向概率. 默认 (0.25, 0.25, 0.25).

    可选 L_invar:
      use_invar_loss (bool): 是否启用光度不变性损失. 默认 False.
      invar_loss_weight (float): L_invar 权重 λ2. 默认 0.05.
      invar_*_range (tuple): T_photo 各分量范围 (Route B in-distribution).

    可选 degraded-cls:
      use_degraded_cls_loss (bool): 是否启用低光退化分支分类损失. 默认 False.
      degraded_cls_loss_weight (float): degraded 分支 cls loss 的额外权重.
      degraded_*: 低光退化参数, 复刻 SimpleAug 的 gamma/sRGB 风格.
    """

    def __init__(self, *args,
                 use_equi_loss: bool = False,
                 equi_loss_weight: float = 0.2,
                 equi_flip_probs: tuple = (0.25, 0.25, 0.25),
                 use_invar_loss: bool = False,
                 invar_loss_weight: float = 0.05,
                 invar_gamma_range: tuple = (0.7, 1.5),
                 invar_rg_range: tuple = (0.95, 1.40),
                 invar_bg_range: tuple = (0.75, 1.05),
                 invar_grad_lr_range: tuple = (0.5, 2.0),
                 invar_grad_ud_range: tuple = (0.7, 1.5),
                 invar_contrast_range: tuple = (0.5, 2.0),
                 use_degraded_cls_loss: bool = False,
                 degraded_cls_loss_weight: float = 0.25,
                 degraded_brightness_range: tuple = (0.4, 1.0),
                 degraded_contrast_range: tuple = (0.6, 1.2),
                 degraded_noise_std_range: tuple = (0.0, 20.0),
                 degraded_prob: float = 0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # L_equi
        self.use_equi_loss = use_equi_loss
        self.equi_loss_weight = equi_loss_weight
        self.equi_flip_probs = equi_flip_probs
        # L_invar
        self.use_invar_loss = use_invar_loss
        self.invar_loss_weight = invar_loss_weight
        self.invar_gamma_range = invar_gamma_range
        self.invar_rg_range = invar_rg_range
        self.invar_bg_range = invar_bg_range
        self.invar_grad_lr_range = invar_grad_lr_range
        self.invar_grad_ud_range = invar_grad_ud_range
        self.invar_contrast_range = invar_contrast_range
        # degraded cls-only branch
        self.use_degraded_cls_loss = use_degraded_cls_loss
        self.degraded_cls_loss_weight = degraded_cls_loss_weight
        self.degraded_brightness_range = degraded_brightness_range
        self.degraded_contrast_range = degraded_contrast_range
        self.degraded_noise_std_range = degraded_noise_std_range
        self.degraded_prob = degraded_prob
        # 惰性导入避免循环依赖
        self._equi_fn = None

    def _get_equi_fn(self):
        if self._equi_fn is None:
            from mmrotate.models.losses.angle_equi import (
                angle_to_emb, equi_flip_loss, mirror_grid_indices,
                valid_feat_mask)
            self._equi_fn = (angle_to_emb, equi_flip_loss,
                             mirror_grid_indices, valid_feat_mask)
        return self._equi_fn

    def init_weights(self):
        super().init_weights()
        if self.use_sigmoid_cls and hasattr(self, 'retina_cls') and self.retina_cls.bias is not None:
            bias_cls = bias_init_with_prob(0.01)
            torch.nn.init.constant_(self.retina_cls.bias, bias_cls)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             equi_flip_outs=None,
             equi_flip_dirs=None,
             photo_outs=None,
             degraded_outs=None):
        """
        覆写标准 loss()：
        1. 生成 Anchor
        2. 展平预测并解码
        3. 调用 SymPOLAAssigner（预测感知）
        4. Level-first 转置后调用 loss_single
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        num_imgs = cls_scores[0].size(0)

        # 1. 铺设全量基础 Anchor 先验
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        # 2. 展平网络输出：逐图拼接所有层级 → [num_anchors_total, C/5]
        flatten_cls_scores = []
        flatten_bbox_preds = []
        for i in range(num_imgs):
            img_cls_scores = []
            img_bbox_preds = []
            for lvl in range(len(cls_scores)):
                cls_score = cls_scores[lvl][i].permute(1, 2, 0).reshape(
                    -1, self.cls_out_channels)
                bbox_pred = bbox_preds[lvl][i].permute(1, 2, 0).reshape(-1, 5)
                img_cls_scores.append(cls_score)
                img_bbox_preds.append(bbox_pred)
            flatten_cls_scores.append(torch.cat(img_cls_scores))
            flatten_bbox_preds.append(torch.cat(img_bbox_preds))

        # 3. 调用 get_targets 执行预测感知分配
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            flatten_cls_scores,
            flatten_bbox_preds,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)

        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        # 强制采用正样本数作为归一化尺度，避免 O2O 场景下被海量背景稀释
        num_total_samples = max(num_total_pos, 1)

        # 4. 将 anchor_list 从 Image-first 转置为 Level-first
        num_levels = self.anchor_generator.num_levels
        decode_max_size = max(
            max(int(max(meta.get('pad_shape', meta['img_shape'])[:2]))
                for meta in img_metas),
            1)
        level_anchor_list = []
        for lvl in range(num_levels):
            level_anchor_list.append(
                torch.cat([anchors[lvl] for anchors in anchor_list]))

        # 5. 执行 loss_single（逐层计算），附带可选 L_equi / L_invar 信息。
        equi_infos = None
        equi_flip_bbox_preds = None
        if (self.use_equi_loss and equi_flip_outs is not None
                and equi_flip_dirs is not None):
            equi_infos = self._build_equi_infos(img_metas, equi_flip_dirs)
            equi_flip_bbox_preds = equi_flip_outs[1]

        photo_bbox_preds = None
        if self.use_invar_loss and photo_outs is not None:
            photo_bbox_preds = photo_outs[1]

        degraded_cls_scores = None
        if self.use_degraded_cls_loss and degraded_outs is not None:
            degraded_cls_scores = degraded_outs[0]

        # multi_apply 不支持每层不同额外参数 → 手动循环
        losses_cls, losses_bbox = [], []
        losses_equi, losses_invar, losses_degraded_cls = [], [], []
        for lvl in range(num_levels):
            lc, lb, le, li, ldc = self.loss_single(
                cls_scores[lvl], bbox_preds[lvl],
                level_anchor_list[lvl],
                labels_list[lvl], label_weights_list[lvl],
                bbox_targets_list[lvl], bbox_weights_list[lvl],
                num_total_samples=num_total_samples,
                decode_max_size=decode_max_size,
                equi_infos=equi_infos,
                equi_flip_bbox_pred=None if equi_flip_bbox_preds is None
                else equi_flip_bbox_preds[lvl],
                photo_bbox_pred=None if photo_bbox_preds is None
                else photo_bbox_preds[lvl],
                degraded_cls_score=None if degraded_cls_scores is None
                else degraded_cls_scores[lvl])
            losses_cls.append(lc)
            losses_bbox.append(lb)
            losses_equi.append(le)
            losses_invar.append(li)
            losses_degraded_cls.append(ldc)

        result = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        if self.use_equi_loss and any(le is not None and le.numel() > 0 for le in losses_equi):
            result['loss_equi'] = losses_equi
        if self.use_invar_loss and any(li is not None and li.numel() > 0 for li in losses_invar):
            result['loss_invar'] = losses_invar
        if (self.use_degraded_cls_loss and any(
                ldc is not None and ldc.numel() > 0
                for ldc in losses_degraded_cls)):
            result['loss_degraded_cls'] = losses_degraded_cls
        return result

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """管理 Batch 维度的拆分与映射"""
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_result_list) = results

        # 任何图像无有效 anchor 时返回 None
        if any([labels is None for labels in all_labels]):
            return None

        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)

        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            cls_scores,
                            bbox_preds,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """单图级别：预测解码 + SymPOLA 动态分配"""
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None,) * 7

        anchors = flat_anchors[inside_flags, :]
        cls_scores_valid = cls_scores[inside_flags, :]
        bbox_preds_valid = bbox_preds[inside_flags, :]

        # 1. 解码预测框（分配器需要物理坐标）
        decoded_bboxes = self.bbox_coder.decode(anchors, bbox_preds_valid)

        # 2. 调用 SymPOLAAssigner（预测感知一对一匹配）
        assign_result = self.assigner.assign(
            cls_scores_valid.detach(),
            decoded_bboxes.detach(),
            gt_labels,
            gt_bboxes,
            img_meta,
            is_training=self.training)

        # 3. PseudoSampler 采样
        sampling_result = self.sampler.sample(
            assign_result, anchors, gt_bboxes)

        # 4. 构建 targets 张量
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # 5. unmap 回全量 anchor 空间
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags,
                           fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples,
                    decode_max_size=None,
                    equi_infos=None,
                    equi_flip_bbox_pred=None,
                    photo_bbox_pred=None,
                    degraded_cls_score=None):
        """逐层 Loss 计算：SymNFLLoss + SymKLDLoss + 可选额外分支.

        Args:
            equi_infos: None 或 list[dict]，每张原图的翻转方向和有效区域.
            equi_flip_bbox_pred: flip(img) 分支在当前 FPN level 的 bbox 预测.
            photo_bbox_pred: T_photo(img) 分支在当前 FPN level 的 bbox 预测.
            degraded_cls_score: degraded(img) 分支在当前 FPN level 的 cls 预测.
        """
        # 1. 维度展平对齐
        # 记忆特征图空间尺寸以便 L_equi 使用 (在 permute 之前获取)
        H_feat = cls_score.size(2)   # input: (B, C_in, H_feat, W_feat)
        W_feat = cls_score.size(3)
        num_imgs_in = cls_score.size(0)

        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        anchors = anchors.reshape(-1, 5)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)

        # 2. 物理框解码
        decoded_pred_bboxes = self.bbox_coder.decode(anchors, bbox_pred)
        decoded_gt_bboxes = self.bbox_coder.decode(anchors, bbox_targets)

        if decode_max_size is not None:
            max_coord = float(decode_max_size - 1)
            decoded_pred_bboxes[:, 0].clamp_(0.0, max_coord)
            decoded_pred_bboxes[:, 1].clamp_(0.0, max_coord)
            decoded_pred_bboxes[:, 2].clamp_(1.0, float(decode_max_size))
            decoded_pred_bboxes[:, 3].clamp_(1.0, float(decode_max_size))

        # [数值安全] 对解码后的宽高做上界钳制，防止极端预测导致协方差矩阵爆炸
        max_wh = float(decode_max_size) if decode_max_size is not None else 2048.0
        decoded_pred_bboxes[:, 2].clamp_(1.0, max_wh)
        decoded_pred_bboxes[:, 3].clamp_(1.0, max_wh)

        # 3. 提取正样本对应的 GT 框（传给 SymNFLLoss 计算 mu_sym）
        pos_inds = (labels >= 0) & (labels < self.num_classes)
        pos_gt_bboxes = decoded_gt_bboxes[pos_inds]
        pos_pred_bboxes = decoded_pred_bboxes[pos_inds]

        # # ── 诊断探针 1：数据流形截获 ──────────────────────────────────
        # if not hasattr(self, '_diag_cnt'):
        #     self._diag_cnt = 0
            
        # if self._diag_cnt < 5:  # 只打印前 5 次，防止日志刷屏崩溃
        #     print(f'\n[DIAG-{self._diag_cnt}] pos数量={pos_inds.sum().item()}')
        #     if pos_inds.sum().item() > 0:
        #         print(f'[DIAG-{self._diag_cnt}] pred_bbox={pos_pred_bboxes[0].detach().cpu().numpy()}')
        #         print(f'[DIAG-{self._diag_cnt}] gt_bbox={pos_gt_bboxes[0].detach().cpu().numpy()}')
        #         print(f'[DIAG-{self._diag_cnt}] bbox_weights_sum={bbox_weights.sum().item()}')
        #     self._diag_cnt += 1
        # # ────────────────────────────────────────────────────────────

        if pos_gt_bboxes.numel() == 0:
            pos_gt_bboxes = decoded_gt_bboxes.new_zeros((0, 5))

        # 4. SymNFL 分类损失
        # =========================================================
        # 【归一化策略】：使用全局正样本数作为归一化因子，
        # 与回归损失保持一致的梯度尺度。SymNFLLoss 内部的 focal weight
        # 已经天然压制了远端背景的梯度贡献，不需要额外用 anchor 总数稀释。
        # =========================================================
        cls_avg_factor = max(float(num_total_samples), 1.0)
        loss_cls = self.loss_cls(
            cls_score,
            labels,
            pred_bboxes=decoded_pred_bboxes,
            gt_bboxes=pos_gt_bboxes,
            weight=label_weights,
            avg_factor=cls_avg_factor)

        # [数值安全] 防止 NaN/Inf 传播导致整个训练崩溃
        if torch.isnan(loss_cls) or torch.isinf(loss_cls):
            loss_cls = cls_score.sum() * 0.0

        # 5. SymKLD 回归损失（仅正样本）
        if pos_inds.numel() > 0:
            pos_pred_bboxes = decoded_pred_bboxes[pos_inds]
            pos_gt_bboxes = decoded_gt_bboxes[pos_inds]

            # OBB 5 自由度权重降维，避免无意义的 5 维广播
            pos_weights = bbox_weights[pos_inds][:, 0]

            loss_bbox = self.loss_bbox(
                pos_pred_bboxes,
                pos_gt_bboxes,
                weight=pos_weights,
                avg_factor=num_total_samples)
            # [数值安全] 防止 NaN/Inf 传播
            if torch.isnan(loss_bbox) or torch.isinf(loss_bbox):
                loss_bbox = bbox_pred.sum() * 0.0
            # # ── 诊断探针 2：回归梯度断点截获 ─────────────────────────────
            # if getattr(self, '_diag_cnt', 0) <= 5 and pos_inds.sum().item() > 0:
            #     print(f'[DIAG-{self._diag_cnt-1}] loss_bbox原始值={loss_bbox.item():.6f}')
            # # ────────────────────────────────────────────────────────────
        else:
            loss_bbox = bbox_pred.sum() * 0.0

        # ================================================================
        # 6. 翻转等变一致性损失 L_equi (可选)
        # ================================================================
        loss_equi = None
        if (self.use_equi_loss and equi_infos is not None
                and equi_flip_bbox_pred is not None
                and pos_inds.numel() > 0):
            loss_equi = self._compute_equi_loss(
                anchors, bbox_pred, labels, decoded_pred_bboxes,
                pos_inds, H_feat, W_feat, num_imgs_in,
                equi_infos, equi_flip_bbox_pred)
        if loss_equi is None:
            loss_equi = bbox_pred.new_zeros(())

        # ================================================================
        # 7. 光度不变性损失 L_invar (可选)
        # ================================================================
        loss_invar = None
        if (self.use_invar_loss and photo_bbox_pred is not None
                and pos_inds.numel() > 0):
            loss_invar = self._compute_invar_loss(
                anchors, bbox_pred, decoded_pred_bboxes,
                pos_inds, H_feat, W_feat, num_imgs_in,
                photo_bbox_pred)
        if loss_invar is None:
            loss_invar = bbox_pred.new_zeros(())

        # ================================================================
        # 8. 低光退化分支分类损失 (可选)
        # ================================================================
        loss_degraded_cls = None
        if self.use_degraded_cls_loss and degraded_cls_score is not None:
            loss_degraded_cls = self._compute_degraded_cls_loss(
                degraded_cls_score,
                labels,
                label_weights,
                decoded_pred_bboxes.detach(),
                pos_gt_bboxes.detach(),
                cls_avg_factor)
        if loss_degraded_cls is None:
            loss_degraded_cls = bbox_pred.new_zeros(())

        return (loss_cls, loss_bbox, loss_equi, loss_invar,
                loss_degraded_cls)

    def _compute_degraded_cls_loss(self, degraded_cls_score, labels,
                                   label_weights, decoded_pred_bboxes,
                                   pos_gt_bboxes, cls_avg_factor):
        """只在 degraded 分支上计算分类/objectness 损失.

        标签和正负样本沿用 clean 分支的分配结果；bbox/angle/equi 不在该
        分支计算。SymNFLLoss 仍需要 pred_bboxes 来构造空间排他权重，这里
        使用 clean 分支已 detach 的 decoded boxes，避免给 degraded bbox 分支
        引入回归/角度梯度。
        """
        degraded_cls_score = degraded_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        loss_degraded_cls = self.loss_cls(
            degraded_cls_score,
            labels,
            pred_bboxes=decoded_pred_bboxes,
            gt_bboxes=pos_gt_bboxes,
            weight=label_weights,
            avg_factor=cls_avg_factor,
            advance_iter=False)

        if torch.isnan(loss_degraded_cls) or torch.isinf(loss_degraded_cls):
            loss_degraded_cls = degraded_cls_score.sum() * 0.0

        return loss_degraded_cls * self.degraded_cls_loss_weight

    def _build_equi_infos(self, img_metas, equi_flip_dirs):
        """从 img_metas 和 detector 生成的方向构建 L_equi 配对信息."""
        equi_infos = []
        for i, meta in enumerate(img_metas):
            direction = equi_flip_dirs[i]
            img_shape = meta.get('img_shape', meta.get('pad_shape'))
            pad_shape = meta.get('pad_shape', img_shape)
            img_H, img_W = img_shape[:2]
            pad_H, pad_W = pad_shape[:2]
            equi_infos.append(dict(
                img_idx=i,
                direction=direction,
                img_H=int(img_H),
                img_W=int(img_W),
                pad_H=int(pad_H),
                pad_W=int(pad_W),
            ))
        return equi_infos

    def _compute_equi_loss(self, anchors, bbox_pred, labels, decoded_pred_bboxes,
                           pos_inds, H_feat, W_feat, num_imgs,
                           equi_infos, equi_flip_bbox_pred):
        """在当前 level 计算 L_equi: 用原图正样本作为 GT 桥接索引.

        只约束角度预测；flip 分支不参与分类/回归监督。
        """
        angle_to_emb, equi_flip_loss, mirror_grid_indices, valid_feat_mask = \
            self._get_equi_fn()

        # 从 anchor_generator 直接取每位置 anchor 数，避免整除不稳定
        num_anchors = self.anchor_generator.num_base_anchors[0]
        if num_anchors <= 0:
            return bbox_pred.new_zeros(())

        per_img_size = H_feat * W_feat * num_anchors
        flip_bbox_pred = equi_flip_bbox_pred.permute(0, 2, 3, 1).reshape(
            -1, 5)
        anchors_by_img = anchors.reshape(num_imgs, per_img_size, 5)

        losses = []
        for info in equi_infos:
            img_idx = info['img_idx']
            direction = info['direction']
            orig_start = img_idx * per_img_size
            orig_end = (img_idx + 1) * per_img_size
            pos_within_img = pos_inds[orig_start:orig_end]
            if not pos_within_img.any():
                continue

            idx_in_img = torch.nonzero(
                pos_within_img, as_tuple=False).squeeze(-1)
            row = idx_in_img // (W_feat * num_anchors)
            rem = idx_in_img % (W_feat * num_anchors)
            col = rem // num_anchors
            anchor_id = rem % num_anchors

            mirror_row, mirror_col = mirror_grid_indices(
                row, col, H_feat, W_feat, direction)

            # 原图位置 valid（padding 左/上锚定，右/下 padding）
            valid_mask = valid_feat_mask(
                row, col, H_feat, W_feat,
                info['img_H'], info['img_W'],
                info['pad_H'], info['pad_W'])
            # flip 图 padding 同样在右/下侧，镜像位置也需满足 flip 图 valid 区
            valid_mask_mir = valid_feat_mask(
                mirror_row, mirror_col, H_feat, W_feat,
                info['img_H'], info['img_W'],
                info['pad_H'], info['pad_W'])
            valid_mask = (valid_mask & valid_mask_mir &
                          (mirror_row >= 0) & (mirror_row < H_feat) &
                          (mirror_col >= 0) & (mirror_col < W_feat))
            if not valid_mask.any():
                continue

            row = row[valid_mask]
            col = col[valid_mask]
            anchor_id = anchor_id[valid_mask]
            mirror_row = mirror_row[valid_mask]
            mirror_col = mirror_col[valid_mask]

            orig_global_idx = (
                orig_start +
                (row * W_feat + col) * num_anchors +
                anchor_id)
            flip_global_idx = (
                img_idx * per_img_size +
                (mirror_row * W_feat + mirror_col) * num_anchors +
                anchor_id)

            theta_orig = decoded_pred_bboxes[orig_global_idx, 4]
            emb_orig = angle_to_emb(theta_orig)

            flip_anchor_idx = (
                (mirror_row * W_feat + mirror_col) * num_anchors +
                anchor_id)
            flip_anchors = anchors_by_img[img_idx, flip_anchor_idx, :]
            flip_decoded = self.bbox_coder.decode(
                flip_anchors, flip_bbox_pred[flip_global_idx, :])
            emb_flip = angle_to_emb(flip_decoded[:, 4])
            losses.append(equi_flip_loss(
                emb_orig, emb_flip, direction, valid_mask=None))

        if not losses:
            return bbox_pred.new_zeros(())

        loss_equi = torch.stack(losses).mean()

        # NaN guard
        if torch.isnan(loss_equi) or torch.isinf(loss_equi):
            loss_equi = bbox_pred.new_zeros(())

        return loss_equi * self.equi_loss_weight

    def _compute_invar_loss(self, anchors, bbox_pred, decoded_pred_bboxes,
                            pos_inds, H_feat, W_feat, num_imgs,
                            photo_bbox_pred):
        """在当前 level 计算 L_invar: 正样本位置的光度不变性约束.

        与 L_equi 不同: 两路图的空间位置完全一致, 不需要 mirror 映射.
        pos_idx 直接复用, 计算更简单.
        """
        angle_to_emb, _, _, _ = self._get_equi_fn()
        from mmrotate.models.losses.angle_equi import invar_photo_loss
        invar_fn = invar_photo_loss

        num_anchors = self.anchor_generator.num_base_anchors[0]
        if num_anchors <= 0:
            return bbox_pred.new_zeros(())

        per_img_size = H_feat * W_feat * num_anchors
        photo_bp = photo_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        anchors_by_img = anchors.reshape(num_imgs, per_img_size, 5)

        losses = []
        for img_idx in range(num_imgs):
            orig_start = img_idx * per_img_size
            pos_within_img = pos_inds[orig_start:orig_start + per_img_size]
            if not pos_within_img.any():
                continue

            idx_in_img = torch.nonzero(
                pos_within_img, as_tuple=False).squeeze(-1)
            row = idx_in_img // (W_feat * num_anchors)
            rem = idx_in_img % (W_feat * num_anchors)
            col = rem // num_anchors
            anchor_id = rem % num_anchors

            # 原图正样本的 global idx
            orig_global_idx = (
                orig_start +
                (row * W_feat + col) * num_anchors +
                anchor_id)

            # L_invar: photo 路径用完全相同的 (row, col, anchor_id)
            photo_global_idx = (
                img_idx * per_img_size +
                (row * W_feat + col) * num_anchors +
                anchor_id)

            # 原图 decoded θ → emb
            theta_orig = decoded_pred_bboxes[orig_global_idx, 4]
            emb_orig = angle_to_emb(theta_orig)

            # photo 路径 decoded θ → emb
            photo_anchor_idx = (row * W_feat + col) * num_anchors + anchor_id
            photo_anchors = anchors_by_img[img_idx, photo_anchor_idx, :]
            photo_decoded = self.bbox_coder.decode(
                photo_anchors, photo_bp[photo_global_idx, :])
            emb_photo = angle_to_emb(photo_decoded[:, 4])

            # L_invar: emb 应一致 (不需要 flip_emb)
            losses.append(invar_fn(emb_orig, emb_photo))

        if not losses:
            return bbox_pred.new_zeros(())

        loss_invar = torch.stack(losses).mean()

        if torch.isnan(loss_invar) or torch.isinf(loss_invar):
            loss_invar = bbox_pred.new_zeros(())

        return loss_invar * self.invar_loss_weight

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=False):
        """端到端 Top-K 推理（物理切断 NMS）"""
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)

        mlvl_bboxes = []
        mlvl_scores = []

        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)

            if cfg.get('score_thr', 0.0) > 0:
                max_scores, _ = scores.max(dim=1)
                valid_mask = max_scores > cfg.score_thr
                scores = scores[valid_mask]
                bbox_pred = bbox_pred[valid_mask]
                anchors = anchors[valid_mask]

            if scores.numel() == 0:
                continue

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        if not mlvl_bboxes:
            return torch.zeros((0, 6), device=cls_score_list[0].device), \
                   torch.zeros((0,), dtype=torch.long,
                               device=cls_score_list[0].device)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)

        scores, labels = mlvl_scores.max(dim=1)
        max_per_img = cfg.get('max_per_img', 1)

        if scores.numel() > max_per_img:
            _, topk_inds = scores.topk(max_per_img)
            bboxes = mlvl_bboxes[topk_inds]
            scores = scores[topk_inds]
            labels = labels[topk_inds]
        else:
            bboxes = mlvl_bboxes

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes[:, :4] /= scale_factor

        det_bboxes = torch.cat([bboxes, scores[:, None]], dim=1)
        return det_bboxes, labels

    def simple_test(self, feats, img_metas, rescale=False):
        """推理入口：兼容 BaseDenseHead.simple_test 调用链。

        直接覆写 simple_test，物理切断对 simple_test_bboxes 的依赖，
        端到端直通 FPN 与 O2O 解码器。
        """
        # 1. 前向传播提取密集特征
        outs = self.forward(feats)
        # 2. 端到端解码预测结果
        # 【物理防线】：显式传入 with_nms=False，彻底封死 NMS 后处理通道
        results_list = self.get_bboxes(
            *outs, img_metas=img_metas, rescale=rescale, with_nms=False)
        return results_list
