"""Training-only port augmentations; no crop or geometric translation."""
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class PortIsotropicShrink:
    """Run after RResize, before flip/Normalize/Pad.

    A single affine scale is used for pixels and OBBs, avoiding independent
    width/height rounding factors. The output canvas is rounded up, not stretched.
    Coordinates are invertible; downsampled image detail is not.
    """
    def __init__(self, prob=0.5, scale_range=(0.5, 1.0)):
        if not 0 <= prob <= 1 or not 0 < scale_range[0] <= scale_range[1] <= 1:
            raise ValueError('Invalid shrink probability/range')
        self.prob, self.scale_range = prob, scale_range

    def __call__(self, results):
        if results.get('mask_fields') or results.get('seg_fields'):
            raise ValueError('PortIsotropicShrink supports OBBs only')
        if np.random.random() >= self.prob:
            return results
        scale = float(np.random.uniform(*self.scale_range))
        img = results['img']
        h, w = img.shape[:2]
        matrix = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float32)
        results['img'] = cv2.warpAffine(
            img, matrix, (int(np.ceil(w * scale)), int(np.ceil(h * scale))),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114))
        for field in results.get('bbox_fields', []):
            boxes = results[field].copy()
            if boxes.ndim != 2 or boxes.shape[1] != 5:
                raise ValueError('Expected Nx5 OBBs')
            boxes[:, :4] *= scale
            results[field] = boxes
        results['img_shape'] = results['img'].shape
        results['pad_shape'] = results['img'].shape
        results['scale_factor'] = np.asarray(results['scale_factor'], dtype=np.float32) * scale
        results['port_shrink_scale'] = scale
        return results


@PIPELINES.register_module()
class PortPhotometricAug:
    """Apply to uint8 real images before Normalize; geometry is untouched.

    Gamma > 1 darkens. This simulates exposure variation, not all night effects.
    """
    def __init__(self, prob=0.5, gamma_range=(0.7, 2.0),
                 gain_range=(0.6, 1.2), contrast_range=(0.8, 1.2)):
        if not 0 <= prob <= 1:
            raise ValueError('Invalid probability')
        for bounds in (gamma_range, gain_range, contrast_range):
            if not 0 < bounds[0] <= bounds[1]:
                raise ValueError('Invalid photometric range')
        self.prob = prob
        self.gamma_range, self.gain_range = gamma_range, gain_range
        self.contrast_range = contrast_range

    def __call__(self, results):
        if np.random.random() >= self.prob:
            return results
        img = results['img']
        if img.dtype != np.uint8:
            raise ValueError('Photometric augmentation must precede Normalize')
        gamma = float(np.random.uniform(*self.gamma_range))
        gain = float(np.random.uniform(*self.gain_range))
        contrast = float(np.random.uniform(*self.contrast_range))
        value = np.power(img.astype(np.float32) / 255., gamma)
        mean = value.mean(axis=(0, 1), keepdims=True)
        value = (value - mean) * contrast + mean
        results['img'] = np.rint(np.clip(value * gain, 0, 1) * 255).astype(np.uint8)
        return results
