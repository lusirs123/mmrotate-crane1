# mmrotate/models/losses/angle_equi.py
"""
角度圆嵌入工具函数 + 翻转等变一致性损失 L_equi.

设计原则:
  - 在 embedding 空间 (cos2θ, sin2θ) 计算损失，处处可导，无 norm_angle 边界问题。
  - 翻转变换是整数双射: 特征图格点映射用整数索引，严禁浮点插值。
  - GT 桥接: SymPOLA 分配的正样本 anchor 在原图和翻转图上配对，GT 只做索引桥。
  - L_equi 关闭时 = 零开销 (use_equi_loss=False).

Pad 约定 (已验证):
  mmcv.impad(shape=(H_target,W_target)) 使用
  padding=(0, 0, ΔW, ΔH) = (left, top, right, bottom).
  即: 图像 left/top 锚定 (0,0) 处，padding 只加在 right 和 bottom 侧。
  镜像映射: c → W-1-c 在有效像素区域是精确双射；
  padding 区域不在原图有效区域中，等变损失通过 valid_mask 剔除。

角度约定 (le90):
  θ ∈ [-π/2, +π/2), 180° 周期, norm_angle('le90') = (θ+π/2) % π - π/2
  embedding: (cos2θ, sin2θ) → 2θ 在单位圆上，天然 180° 周期无歧义.

翻转角度变换 (见 transforms.py:RRandomFlip.bbox_flip, 已验证):
  horizontal:  θ → -θ  (等价于 (cos2θ, -sin2θ))
  vertical:    θ → -θ  (同上)
  diagonal:    θ →  θ  (emb 不变!)
"""

from crane_project.utils.angle_equi_core import (  # noqa: F401
    angle_to_emb, equi_flip_loss, extract_emb_at_positions, flip_emb,
    mirror_grid_indices, valid_feat_mask)

__all__ = [
    'angle_to_emb', 'flip_emb', 'equi_flip_loss',
    'mirror_grid_indices', 'valid_feat_mask', 'extract_emb_at_positions'
]
