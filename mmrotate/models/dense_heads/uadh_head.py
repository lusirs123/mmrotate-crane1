# mmrotate/models/dense_heads/uadh_head.py
import torch
import torch.nn as nn
from mmrotate.models.builder import ROTATED_HEADS


@ROTATED_HEADS.register_module(force=True)
class UADHead(nn.Module):
    """Uncertainty-Aware Diagonal Head (UADH)."""

    def __init__(self,
                 in_channels=256,
                 feat_channels=128,
                 mid_channels=64,
                 loss_weight_nll=1.0,
                 loss_weight_consistency=0.1,
                 gaussian_sigma_ratio=0.25,
                 mask_threshold=0.1,
                 min_log_var=-6.0,
                 max_log_var=4.0):
        super(UADHead, self).__init__()
        self.loss_weight_nll = loss_weight_nll
        self.loss_weight_consistency = loss_weight_consistency
        self.gaussian_sigma_ratio = gaussian_sigma_ratio
        self.mask_threshold = mask_threshold
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        self.conv1 = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(32, feat_channels)
        self.conv2 = nn.Conv2d(feat_channels, mid_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(32, mid_channels)
        self.conv_out = nn.Conv2d(mid_channels, 2, 3, padding=1)

    def init_weights(self):
        for m in [self.conv1, self.conv2, self.conv_out]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.conv_out.bias, 0)

    def forward(self, feat):
        x = torch.relu(self.gn1(self.conv1(feat)))
        x = torch.relu(self.gn2(self.conv2(x)))
        return self.conv_out(x)

    def _render_gaussian_and_target(self, gt_bboxes, H, W, img_h, img_w, device):
        weight_map = torch.zeros(H, W, device=device)
        target_map = torch.zeros(H, W, device=device)

        if gt_bboxes.shape[0] == 0:
            return weight_map, target_map

        scale_x = float(W) / float(img_w)
        scale_y = float(H) / float(img_h)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij')

        for box in gt_bboxes:
            cx = box[0] * scale_x
            cy = box[1] * scale_y
            gt_w = box[2].clamp(min=1.0)
            gt_h = box[3].clamp(min=1.0)
            sigma = torch.clamp(
                torch.minimum(gt_w * scale_x, gt_h * scale_y)
                * self.gaussian_sigma_ratio,
                min=2.0)

            gaussian = torch.exp(
                -((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
                / (2.0 * sigma ** 2))
            log_l_diag = torch.log(torch.sqrt(gt_w ** 2 + gt_h ** 2))

            replace = gaussian > weight_map
            weight_map = torch.maximum(weight_map, gaussian)
            target_map = torch.where(replace, log_l_diag, target_map)

        return weight_map, target_map

    def _main_log_diag_map(self, main_outs, img_metas, bbox_head, H, W):
        if main_outs is None or bbox_head is None:
            return None

        cls_scores, bbox_preds = main_outs[:2]
        cls_score = cls_scores[0]
        bbox_pred = bbox_preds[0]
        if cls_score.size(-2) != H or cls_score.size(-1) != W:
            return None

        device = bbox_pred.device
        featmap_size = bbox_pred.shape[-2:]
        anchors = bbox_head.anchor_generator.single_level_grid_priors(
            featmap_size, level_idx=0, dtype=bbox_pred.dtype, device=device)
        num_anchors = bbox_head.anchor_generator.num_base_anchors[0]
        num_imgs = bbox_pred.size(0)

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
            num_imgs, H, W, num_anchors, 5)
        anchors = anchors.reshape(H, W, num_anchors, 5)

        decoded = []
        for i in range(num_imgs):
            decoded_i = bbox_head.bbox_coder.decode(
                anchors.reshape(-1, 5), bbox_pred[i].reshape(-1, 5))
            decoded.append(decoded_i.reshape(H, W, num_anchors, 5))
        decoded = torch.stack(decoded, dim=0)

        if getattr(bbox_head, 'use_sigmoid_cls', True):
            scores = cls_score.sigmoid()
        else:
            scores = cls_score.softmax(dim=1)
        scores = scores.permute(0, 2, 3, 1).reshape(num_imgs, H, W, num_anchors, -1)
        scores = scores.max(dim=-1)[0]
        best_anchor = scores.argmax(dim=-1, keepdim=True).unsqueeze(-1).expand(
            num_imgs, H, W, 1, 5)
        best_boxes = decoded.gather(3, best_anchor).squeeze(3)

        max_img_size = max(
            max(int(max(meta.get('pad_shape', meta['img_shape'])[:2])), 1)
            for meta in img_metas)
        wh = best_boxes[..., 2:4].clamp(min=1.0, max=float(max_img_size))
        return torch.log(torch.sqrt((wh ** 2).sum(dim=-1)).clamp(min=1e-6)).detach()

    def loss(self, pred, gt_bboxes_list, img_metas, main_outs=None, bbox_head=None):
        N, _, H, W = pred.shape
        mu = pred[:, 0]
        log_var = pred[:, 1].clamp(self.min_log_var, self.max_log_var)
        main_log_diag = self._main_log_diag_map(main_outs, img_metas, bbox_head, H, W)

        total_nll = pred.new_zeros(())
        total_consistency = pred.new_zeros(())
        total_weight = pred.new_zeros(())

        for i in range(N):
            img_h, img_w = img_metas[i]['img_shape'][:2]
            weight_map, target_map = self._render_gaussian_and_target(
                gt_bboxes_list[i], H, W, img_h, img_w, pred.device)
            mask = weight_map > self.mask_threshold
            if not mask.any():
                continue

            weights = weight_map[mask]
            diff = target_map[mask] - mu[i][mask]
            nll = 0.5 * (log_var[i][mask] + diff.pow(2) / log_var[i][mask].exp().clamp(min=1e-8))
            total_nll = total_nll + (nll * weights).sum()

            if main_log_diag is not None and self.loss_weight_consistency > 0:
                consistency = (mu[i][mask] - main_log_diag[i][mask]).pow(2)
                total_consistency = total_consistency + (consistency * weights).sum()

            total_weight = total_weight + weights.sum()

        normalizer = total_weight.clamp(min=1.0)
        losses = dict(loss_uadh_nll=self.loss_weight_nll * total_nll / normalizer)
        if self.loss_weight_consistency > 0:
            losses['loss_uadh_consistency'] = (
                self.loss_weight_consistency * total_consistency / normalizer)
        return losses
