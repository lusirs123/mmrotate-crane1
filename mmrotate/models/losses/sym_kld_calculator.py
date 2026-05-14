

#底层度量算子

import torch
__all__ = ['sym_kld']

def _xywha_to_gaussian(boxes):
    cx, cy, w, h, a = boxes.unbind(-1)
    
    # [物理防线] 在生成协方差矩阵前强制钳制宽高，避免奇异矩阵和除零爆炸
    # 1024 尺度图像中，1 个像素是更稳健的数值下限
    min_size = 1.0
    w = w.clamp(min=min_size)
    h = h.clamp(min=min_size)

    cos_a = torch.cos(a)
    sin_a = torch.sin(a)

    w2 = (w * 0.5) ** 2
    h2 = (h * 0.5) ** 2

    sxx = cos_a * cos_a * w2 + sin_a * sin_a * h2
    sxy = cos_a * sin_a * (w2 - h2)
    syy = sin_a * sin_a * w2 + cos_a * cos_a * h2

    mu = torch.stack([cx, cy], dim=-1)
    sigma = torch.stack([
        torch.stack([sxx, sxy], dim=-1),
        torch.stack([sxy, syy], dim=-1)
    ], dim=-2)
    return mu, sigma

def _inv2x2_safe(sigma, eps=1e-6):
    a = sigma[..., 0, 0]
    b = sigma[..., 0, 1]
    c = sigma[..., 1, 0]
    d = sigma[..., 1, 1]
    det = (a * d - b * c).clamp(min=eps)
    inv = torch.stack([
        torch.stack([d, -b], dim=-1),
        torch.stack([-c, a], dim=-1)
    ], dim=-2) / det[..., None, None]
    # [数值安全] 钳制逆矩阵元素，防止近奇异矩阵产生极大值
    inv = inv.clamp(-1e4, 1e4)
    return inv

def sym_kld(boxes_p, boxes_q, eps=1e-6):
    mu_p, sigma_p = _xywha_to_gaussian(boxes_p)
    mu_q, sigma_q = _xywha_to_gaussian(boxes_q)

    inv_p = _inv2x2_safe(sigma_p, eps=eps)
    inv_q = _inv2x2_safe(sigma_q, eps=eps)

    trace_qp = torch.einsum('...ij,...ji->...', inv_q, sigma_p)
    trace_pq = torch.einsum('...ij,...ji->...', inv_p, sigma_q)

    delta = (mu_p - mu_q).unsqueeze(-1)
    maha = (delta.transpose(-1, -2) @ (inv_p + inv_q) @ delta).squeeze(-1).squeeze(-1)

    # 抑制因计算精度导致的微小负值，同时设置上界防止数值爆炸
    raw = 0.5 * (trace_qp + trace_pq - 4.0 + maha)
    return raw.clamp(min=0.0, max=1e4)


