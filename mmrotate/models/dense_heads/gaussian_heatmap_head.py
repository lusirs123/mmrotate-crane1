# mmrotate/models/dense_heads/gaussian_heatmap_head.py
import torch
import torch.nn as nn
from mmrotate.models.builder import ROTATED_HEADS


@ROTATED_HEADS.register_module(force=True)
class GaussianHeatmapHead(nn.Module):
    """
    轻量 Anchor-free 高斯热图辅助头（MMRotate 0.x 兼容版）
    仅参与训练阶段，推理时不调用。
    """

    def __init__(self, in_channels=256, loss_weight=0.5):
        super(GaussianHeatmapHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.loss_weight = loss_weight

    def init_weights(self):
        """0.x 标准权重初始化接口"""
        nn.init.normal_(self.conv.weight, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, feat):
        """
        Args:
            feat (Tensor): FPN P3 特征图，shape [N, C, H, W]
        Returns:
            Tensor: 热图，shape [N, 1, H, W]，值域 [0,1]
        """
        return self.conv(feat).sigmoid()

    def _render_gaussian(self, gt_bboxes, H, W, img_h, img_w, device):
        """将 GT OBB 中心点渲染为二维高斯热图"""
        heatmap = torch.zeros(1, H, W, device=device)
        if gt_bboxes.shape[0] == 0:
            return heatmap

        scale_x = float(W) / float(img_w)
        scale_y = float(H) / float(img_h)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32)
        )

        for box in gt_bboxes:
            cx = float(box[0]) * scale_x
            cy = float(box[1]) * scale_y
            r = max(2.0, float(min(box[2], box[3])) * min(scale_x, scale_y) / 4.0)
            gaussian = torch.exp(
                -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2.0 * r ** 2)
            )
            heatmap = torch.maximum(heatmap, gaussian.unsqueeze(0))

        return heatmap

    def loss(self, pred_heatmap, gt_bboxes_list, img_metas):
        """
        Gaussian Focal Loss（CornerNet 风格）
        Args:
            pred_heatmap (Tensor): [N, 1, H, W]
            gt_bboxes_list (list[Tensor]): 每张图的 GT OBB，格式 (cx,cy,w,h,theta)
            img_metas (list[dict]): 图像元信息
        Returns:
            Tensor: 标量损失
        """
        total_loss = pred_heatmap.new_zeros(1)
        N = pred_heatmap.shape[0]

        for i in range(N):
            img_h, img_w = img_metas[i]['img_shape'][:2]
            _, H, W = pred_heatmap[i].shape

            gt_hm = self._render_gaussian(
                gt_bboxes_list[i], H, W, img_h, img_w,
                device=pred_heatmap.device
            )

            p = pred_heatmap[i].clamp(min=1e-6, max=1.0 - 1e-6)

            # 正样本（高斯峰值处）
            pos_mask = gt_hm.eq(1.0).float()
            pos_loss = -((1.0 - p) ** 2) * torch.log(p) * pos_mask

            # 负样本（其余位置，按高斯值降权）
            neg_mask = gt_hm.lt(1.0).float()
            neg_loss = -(p ** 2) * torch.log(1.0 - p) \
                       * ((1.0 - gt_hm) ** 4) * neg_mask

            total_loss += (pos_loss.sum() + neg_loss.sum())

        return self.loss_weight * total_loss / max(1, N)