# tests/test_photo_invar.py
"""
Unit tests for L_invar (photometric invariance) and T_photo.

Test groups:
  (1) T_photo identity / zero-perturbation
  (2) T_photo denormalize/renormalize roundtrip
  (3) T_photo directional effects (gamma darkens, gradient shifts, etc.)
  (4) L_invar loss properties (stop-gradient, symmetry, empty input)
  (5) on_prob behaviour
  (6) T_photo output range sanity

Run:
    cd /Users/mac/Documents/paper/symEOOD
    /opt/anaconda3/envs/mmrot/bin/python3 -m pytest tests/test_photo_invar.py -v
"""

import torch
import pytest

from crane_project.utils.angle_equi_core import (
    angle_to_emb, invar_photo_loss, build_photo_params, apply_t_photo,
    _denormalize_rgb, _renormalize_rgb, _PHOTO_MEAN_RGB, _PHOTO_STD_RGB)


# =====================================================================
# (1) T_photo: identity / zero-perturbation
# =====================================================================

class TestTPhotoIdentity:
    """T_photo with identity params must return the original normalized tensor."""

    def test_identity_params(self):
        """When all params are identity (=1.0), output ≈ input in normalized space."""
        B, C, H, W = 2, 3, 64, 64
        # mid-gray ≈ 114/255 → normalized ≈ 0 (stays in valid pixel range after denorm)
        img = torch.zeros(B, C, H, W)

        params = dict(
            gamma=torch.ones(B),
            rg_gain=torch.ones(B),
            bg_gain=torch.ones(B),
            contrast=torch.ones(B),
            grad_lr=torch.ones(B),
            grad_ud=torch.ones(B),
        )
        out = apply_t_photo(img, params)
        assert torch.allclose(out, img, atol=1e-4), \
            f"Identity T_photo: max diff = {(out - img).abs().max():.6f}"

    def test_denorm_renorm_roundtrip(self):
        """denormalize → renormalize is identity for valid [0,1] values."""
        # mid-gray normalized value ≈ 0 for each channel
        img = torch.zeros(1, 3, 4, 4)
        roundtrip = _renormalize_rgb(_denormalize_rgb(img).clamp(0, 1))
        assert torch.allclose(roundtrip, img, atol=1e-4), \
            f"Roundtrip max diff = {(roundtrip - img).abs().max():.6f}"


# =====================================================================
# (2) T_photo: directional effects (tested in normalized space)
# =====================================================================

class TestTPhotoEffects:
    """Verify T_photo perturbations have the expected direction."""

    def _make_img(self, B=10, pixel_val=128.0):
        """Create a normalized-space tensor corresponding to a uniform pixel value."""
        # pixel_val in [0,255] → normalize: (pixel_val - mean) / std
        # Use mean of all channels for simplicity
        mean = sum(_PHOTO_MEAN_RGB) / 3  # ≈ 114.5
        std = sum(_PHOTO_STD_RGB) / 3    # ≈ 57.7
        norm_val = (pixel_val - mean) / std
        return torch.full((B, 3, 32, 32), norm_val)

    def test_gamma_gt1_darkens(self):
        """gamma > 1 should reduce brightness (in linear space, x^gamma < x for x<1)."""
        img = self._make_img(B=50, pixel_val=128.0)
        params = dict(
            gamma=torch.full((50,), 2.0),
            rg_gain=torch.ones(50),
            bg_gain=torch.ones(50),
            contrast=torch.ones(50),
            grad_lr=torch.ones(50),
            grad_ud=torch.ones(50),
        )
        out = apply_t_photo(img, params)
        assert out.mean() < img.mean(), \
            f"gamma=2 should darken: out_mean={out.mean():.4f} vs in_mean={img.mean():.4f}"

    def test_gamma_lt1_brightens(self):
        """gamma < 1 should increase brightness."""
        img = self._make_img(B=50, pixel_val=128.0)
        params = dict(
            gamma=torch.full((50,), 0.5),
            rg_gain=torch.ones(50),
            bg_gain=torch.ones(50),
            contrast=torch.ones(50),
            grad_lr=torch.ones(50),
            grad_ud=torch.ones(50),
        )
        out = apply_t_photo(img, params)
        assert out.mean() > img.mean(), \
            f"gamma=0.5 should brighten: out_mean={out.mean():.4f} vs in_mean={img.mean():.4f}"

    def test_spatial_gradient_lr_shifts(self):
        """grad_lr > 1 should make right side brighter than left."""
        img = self._make_img(B=1, pixel_val=128.0)
        params = dict(
            gamma=torch.ones(1),
            rg_gain=torch.ones(1),
            bg_gain=torch.ones(1),
            contrast=torch.ones(1),
            grad_lr=torch.tensor([2.0]),
            grad_ud=torch.ones(1),
        )
        out = apply_t_photo(img, params)
        # In normalized space, brighter = more positive
        left_mean = out[:, :, :, :16].mean()
        right_mean = out[:, :, :, 16:].mean()
        assert right_mean > left_mean, \
            f"grad_lr=2: right ({right_mean:.4f}) should > left ({left_mean:.4f})"

    def test_rg_gain_affects_channels(self):
        """rg_gain > 1 should make channel 0 (R) brighter relative to channel 1 (G)."""
        img = self._make_img(B=1, pixel_val=128.0)
        params = dict(
            gamma=torch.ones(1),
            rg_gain=torch.tensor([1.5]),
            bg_gain=torch.ones(1),
            contrast=torch.ones(1),
            grad_lr=torch.ones(1),
            grad_ud=torch.ones(1),
        )
        out = apply_t_photo(img, params)
        r_mean = out[:, 0].mean()
        g_mean = out[:, 1].mean()
        assert r_mean > g_mean, \
            f"rg_gain=1.5: R ({r_mean:.4f}) should > G ({g_mean:.4f})"


# =====================================================================
# (3) L_invar: loss properties
# =====================================================================

class TestInvarLoss:
    """L_invar correctness tests."""

    def test_zero_on_identical(self):
        """L_invar(emb, emb) = 0."""
        theta = torch.tensor([0.3, -0.5, 0.0, 0.8])
        emb = angle_to_emb(theta)
        loss = invar_photo_loss(emb, emb)
        assert loss.item() < 1e-7

    def test_positive_on_different(self):
        """L_invar > 0 when embeddings differ."""
        emb1 = angle_to_emb(torch.tensor([0.3]))
        emb2 = angle_to_emb(torch.tensor([-0.3]))
        loss = invar_photo_loss(emb1, emb2)
        assert loss.item() > 0

    def test_stop_gradient_on_orig(self):
        """pred_emb_orig should NOT receive gradient (one-directional constraint)."""
        emb_orig = angle_to_emb(torch.tensor([0.5]))
        emb_orig_leaf = emb_orig.clone().requires_grad_(True)
        emb_photo = angle_to_emb(torch.tensor([0.3])).detach().requires_grad_(True)

        loss = invar_photo_loss(emb_orig_leaf, emb_photo)
        loss.backward()

        # emb_orig_leaf should have ZERO gradient (detach in invar_photo_loss)
        assert emb_orig_leaf.grad is None or torch.all(emb_orig_leaf.grad == 0), \
            f"emb_orig should have no gradient, got {emb_orig_leaf.grad}"
        # emb_photo should have non-zero gradient
        assert emb_photo.grad is not None and not torch.all(emb_photo.grad == 0), \
            "emb_photo should receive gradient"

    def test_gradient_flows_to_photo(self):
        """Gradient flows to emb_photo but not emb_orig."""
        emb_orig = angle_to_emb(torch.tensor([0.5]))
        emb_photo = torch.tensor([0.8, 0.6], requires_grad=True)
        loss = invar_photo_loss(emb_orig, emb_photo)
        loss.backward()
        assert emb_photo.grad is not None
        assert not torch.all(emb_photo.grad == 0)

    def test_empty_input(self):
        """Empty tensor returns zero loss."""
        emb = torch.zeros((0, 2))
        loss = invar_photo_loss(emb, emb)
        assert loss.item() == 0.0

    def test_symmetry_broken_by_detach(self):
        """With detach on orig, L_invar(a,b) ≠ L_invar(b,a) in general."""
        a = angle_to_emb(torch.tensor([0.3])).detach().requires_grad_(True)
        b = angle_to_emb(torch.tensor([0.5])).detach().requires_grad_(True)
        loss_ab = invar_photo_loss(a, b)
        loss_ba = invar_photo_loss(b, a)
        # Both should be positive (different embeddings)
        assert loss_ab.item() > 0
        assert loss_ba.item() > 0
        # Values should be equal (same L2, detach doesn't change value)
        assert abs(loss_ab.item() - loss_ba.item()) < 1e-6


# =====================================================================
# (4) on_prob behaviour
# =====================================================================

class TestOnProb:
    """Verify Bernoulli on_prob sampling."""

    def test_on_prob_zero_returns_identity(self):
        """With on_prob=0, all components are identity → output ≈ input."""
        B = 10
        # mid-gray (stays in valid pixel range after denorm)
        img = torch.zeros(B, 3, 16, 16)
        params = build_photo_params(B, torch.device('cpu'), on_prob=0.0)

        # All param values should be 1.0 (identity)
        for key in ['gamma', 'rg_gain', 'bg_gain', 'contrast', 'grad_lr', 'grad_ud']:
            assert torch.allclose(params[key], torch.ones(B)), \
                f"on_prob=0: {key} should be all 1.0, got {params[key]}"

        out = apply_t_photo(img, params)
        assert torch.allclose(out, img, atol=1e-4), \
            f"on_prob=0: output should equal input"

    def test_on_prob_one_samples_all(self):
        """With on_prob=1, all components are sampled (not identity)."""
        B = 100
        params = build_photo_params(B, torch.device('cpu'), on_prob=1.0,
                                    gamma_range=(0.8, 1.2))

        # At least some gamma values should differ from 1.0
        non_identity = (params['gamma'] - 1.0).abs() > 0.01
        assert non_identity.sum() > 50, \
            f"on_prob=1: expected >50 non-identity gamma, got {non_identity.sum()}"


# =====================================================================
# (5) T_photo output range sanity
# =====================================================================

class TestTPhotoOutputRange:
    """Verify T_photo output stays in reasonable normalized-space range."""

    def test_output_not_extreme(self):
        """For a realistic normalized image, output should not explode."""
        B = 50
        # Typical normalized values: mean≈0, std≈1
        img = torch.randn(B, 3, 32, 32) * 0.8

        params = build_photo_params(
            B, device=torch.device('cpu'),
            gamma_range=(0.7, 1.5),
            rg_range=(0.95, 1.40),
            bg_range=(0.75, 1.05),
            grad_lr_range=(0.5, 2.0),
            grad_ud_range=(0.7, 1.5),
            contrast_range=(0.5, 2.0),
            on_prob=1.0,  # all perturbations on
        )
        out = apply_t_photo(img, params)

        # Output should be finite and within a reasonable range
        assert torch.isfinite(out).all(), "Output contains NaN/Inf"
        # In normalized space, extreme values would be < -5 or > 5
        assert (out > -5).all() and (out < 5).all(), \
            f"Output out of range: [{out.min():.2f}, {out.max():.2f}]"

    def test_build_photo_params_shapes(self):
        """build_photo_params returns correct shapes."""
        B = 4
        params = build_photo_params(B, torch.device('cpu'))
        for key in ['gamma', 'rg_gain', 'bg_gain', 'contrast', 'grad_lr', 'grad_ud']:
            assert params[key].shape == (B,), f"{key} shape mismatch"
            assert params[key].device == torch.device('cpu')

    def test_apply_t_photo_preserves_shape(self):
        """apply_t_photo output has same shape as input."""
        B, C, H, W = 2, 3, 16, 16
        img = torch.randn(B, C, H, W)
        params = build_photo_params(B, torch.device('cpu'))
        out = apply_t_photo(img, params)
        assert out.shape == img.shape