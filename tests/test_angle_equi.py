# tests/test_angle_equi.py
"""
Unit tests for L_equi (flip equivariance) — pure torch, no mmcv dependency.

Three test groups:
  (1) diagonal陷阱: flip_emb direction correctness
  (2) padding对称性: valid_feat_mask + mirror_grid_indices integration
  (3) 边界角度: embedding continuity + equivariance at θ boundaries

Run:
    pytest tests/test_angle_equi.py -v
"""

import math
import torch
import pytest

# Import core functions directly (no mmcv needed)
from crane_project.utils.angle_equi_core import (
    angle_to_emb, flip_emb, equi_flip_loss,
    mirror_grid_indices, valid_feat_mask)


# =====================================================================
# (1) DIAGONAL 陷阱 + 翻转符号变换
# =====================================================================

class TestFlipEmb:
    """Test flip_emb: the most error-prone part (diagonal MUST be identity)."""

    def test_diagonal_identity(self):
        """flip_emb(emb, 'diagonal') MUST return the same embedding unchanged."""
        for theta_deg in [-90, -89, -45, 0, 45, 89, 90]:
            theta = torch.tensor(math.radians(theta_deg))
            emb = angle_to_emb(theta)
            result = flip_emb(emb, 'diagonal')
            assert torch.allclose(result, emb, atol=1e-7), \
                f"diagonal θ={theta_deg}°: expected identity, got diff={torch.abs(result - emb).max()}"

    def test_horizontal_sign_flip(self):
        """flip_emb(emb, 'horizontal') MUST return (cos2θ, -sin2θ)."""
        for theta_deg in [-90, -45, 0, 45, 89]:
            theta = torch.tensor(math.radians(theta_deg))
            emb = angle_to_emb(theta)
            expected = torch.tensor([math.cos(2 * theta.item()),
                                     -math.sin(2 * theta.item())])
            result = flip_emb(emb, 'horizontal')
            assert torch.allclose(result, expected, atol=1e-6), \
                f"horizontal θ={theta_deg}°: expected {expected}, got {result}"

    def test_vertical_sign_flip(self):
        """flip_emb(emb, 'vertical') MUST return (cos2θ, -sin2θ) — same as horizontal."""
        for theta_deg in [-90, -45, 0, 45, 89]:
            theta = torch.tensor(math.radians(theta_deg))
            emb = angle_to_emb(theta)
            h_result = flip_emb(emb, 'horizontal')
            v_result = flip_emb(emb, 'vertical')
            assert torch.allclose(h_result, v_result, atol=1e-7), \
                f"vertical vs horizontal θ={theta_deg}°: mismatch"

    def test_flip_flip_identity(self):
        """Two flips (horizontal + horizontal) = identity for all directions."""
        theta = torch.tensor(math.radians(37.3))
        emb = angle_to_emb(theta)
        for direction in ['horizontal', 'vertical', 'diagonal']:
            once = flip_emb(emb, direction)
            twice = flip_emb(once, direction)
            assert torch.allclose(twice, emb, atol=1e-6), \
                f"double flip {direction} not identity for θ=37.3°"

    def test_horizontal_vertical_product_is_diagonal(self):
        """flip_h ∘ flip_v = flip_diag for all θ (mathematical identity in emb space)."""
        for theta_deg in [-89, -45, 0, 45, 89]:
            theta = torch.tensor(math.radians(theta_deg))
            emb = angle_to_emb(theta)
            composed = flip_emb(flip_emb(emb, 'horizontal'), 'vertical')
            diag = flip_emb(emb, 'diagonal')
            assert torch.allclose(composed, diag, atol=1e-6), \
                f"h∘v ≠ diag at θ={theta_deg}°"


# =====================================================================
# (2) PADDING 对称性 + MIRROR GRID INDICES + VALID MASK
# =====================================================================

class TestPaddingAndMirror:
    """Test valid_feat_mask + mirror_grid_indices with left/top anchored padding."""

    def test_mirror_horizontal_roundtrip(self):
        """mirror ∘ mirror = identity for horizontal on full grid."""
        H_feat, W_feat = 8, 16
        row = torch.arange(H_feat).unsqueeze(1).expand(H_feat, W_feat).reshape(-1)
        col = torch.arange(W_feat).unsqueeze(0).expand(H_feat, W_feat).reshape(-1)
        mr, mc = mirror_grid_indices(row, col, H_feat, W_feat, 'horizontal')
        assert torch.equal(mr, row), "row should be unchanged for horizontal"
        # Two mirrors = identity
        mr2, mc2 = mirror_grid_indices(mr, mc, H_feat, W_feat, 'horizontal')
        assert torch.equal(mr2, row)
        assert torch.equal(mc2, col)

    def test_mirror_diagonal_roundtrip(self):
        """mirror ∘ mirror = identity for diagonal on full grid."""
        H_feat, W_feat = 8, 16
        row = torch.arange(H_feat).unsqueeze(1).expand(H_feat, W_feat).reshape(-1)
        col = torch.arange(W_feat).unsqueeze(0).expand(H_feat, W_feat).reshape(-1)
        mr, mc = mirror_grid_indices(row, col, H_feat, W_feat, 'diagonal')
        mr2, mc2 = mirror_grid_indices(mr, mc, H_feat, W_feat, 'diagonal')
        assert torch.equal(mr2, row)
        assert torch.equal(mc2, col)

    def test_valid_feat_mask_no_padding(self):
        """When img_H == pad_H (no padding), all cells are valid."""
        H_feat, W_feat = 8, 16
        row = torch.arange(H_feat).unsqueeze(1).expand(H_feat, W_feat).reshape(-1)
        col = torch.arange(W_feat).unsqueeze(0).expand(H_feat, W_feat).reshape(-1)
        mask = valid_feat_mask(row, col, H_feat, W_feat,
                               img_H=1024, img_W=1024,
                               pad_H=1024, pad_W=1024)
        assert mask.all(), "All cells valid when no padding"

    def test_valid_feat_mask_with_right_bottom_padding(self):
        """When img_W < pad_W (right padding), right columns are invalid.

        Pad convention: left/top anchored, padding on right/bottom.
        Example: H_feat=8, W_feat=16, img_H=1024, img_W=768, pad_H=1024, pad_W=1024
        valid_w = ceil(img_W * W_feat / pad_W) = ceil(768*16/1024) = ceil(12) = 12
        So columns [0..11] valid, [12..15] invalid.
        """
        H_feat, W_feat = 8, 16
        img_H, img_W = 1024, 768
        pad_H, pad_W = 1024, 1024

        # Expected valid_w = ceil(768 * 16 / 1024) = ceil(12.0) = 12
        expected_valid_w = math.ceil(img_W * W_feat / pad_W)

        row = torch.arange(H_feat).unsqueeze(1).expand(H_feat, W_feat).reshape(-1)
        col = torch.arange(W_feat).unsqueeze(0).expand(H_feat, W_feat).reshape(-1)
        mask = valid_feat_mask(row, col, H_feat, W_feat,
                               img_H=img_H, img_W=img_W,
                               pad_H=pad_H, pad_W=pad_W)

        # Check specific cells
        for c in range(W_feat):
            is_valid_c = mask[col == c].all().item()
            expected_valid_c = c < expected_valid_w
            assert is_valid_c == expected_valid_c, \
                f"col={c}: expected {'valid' if expected_valid_c else 'invalid'}, " \
                f"got {'valid' if is_valid_c else 'invalid'} (valid_w={expected_valid_w})"

    def test_mirror_maps_valid_to_valid_horizontal(self):
        """For horizontal flip with right padding:
        If cell (r, c) is valid (c < valid_w), its mirror W-1-c should be valid
        in the FLIP image's valid region.

        In the flip image, valid columns = [0..valid_w-1] (left/top anchored).
        Mirror of valid c is W-1-c. For c < valid_w:
            W-1-c > W-1-valid_w (in right region of flip image).
            But the flip image's padding is also on the right!
            Actually: flip image valid region = [W-valid_w..W-1] when measured in original coords,
            but in flip image coords it's [0..valid_w-1].
            mirror_col = W-1-c, and if c < valid_w, then W-1-c > W-1-valid_w.

        The key test: for a valid cell (c_orig < valid_w),
        its mirror in the flip image should be at position c_orig (not W-1-c_orig)
        because the flip image's coordinate system is reversed.

        But we use ORIGINAL coordinate indexing for both images, so mirror(c_orig) = W-1-c_orig.
        We need W-1-c_orig to also be in valid region of flip image.

        Valid in flip image: W-1-c_orig >= W-valid_w (in original coords of flip image).
        Since c_orig < valid_w, we have W-1-c_orig > W-1-valid_w >= W-valid_w. ✓

        This test verifies the numeric result.
        """
        H_feat, W_feat = 8, 16
        img_H, img_W = 1024, 768
        pad_H, pad_W = 1024, 1024

        valid_w = math.ceil(img_W * W_feat / pad_W)

        # Pick a valid cell and check its mirror is also valid (in flip image sense)
        r_orig = torch.tensor([0, 3, 7])
        c_orig = torch.tensor([0, 5, valid_w - 1])  # All valid

        r_mir, c_mir = mirror_grid_indices(r_orig, c_orig, H_feat, W_feat, 'horizontal')
        # r_mir should be same as r_orig for horizontal
        assert torch.equal(r_mir, r_orig)
        # c_mir = W_feat - 1 - c_orig
        assert torch.equal(c_mir, torch.tensor([W_feat - 1, W_feat - 1 - 5, W_feat - valid_w]))

        # c_mir should be in [W_feat-valid_w, W_feat-1] (flip image valid range in original coords)
        flip_valid_start = W_feat - valid_w
        assert (c_mir >= flip_valid_start).all(), \
            f"Mirror {c_mir} should be >= {flip_valid_start}"

    def test_valid_mask_excludes_padding_cells(self):
        """Specific test: 1024x768 image padded to 1024x1024.
        Grid 128x128 (stride=8).
        valid_w = ceil(768*128/1024) = 96.
        Column 96 should be invalid.
        Column 95 should be valid.
        """
        H_feat, W_feat = 128, 128
        img_H, img_W = 1024, 768
        pad_H, pad_W = 1024, 1024

        # Test specific cells
        row_test = torch.tensor([0, 0, 0, 0])
        col_test = torch.tensor([0, 95, 96, 127])
        mask = valid_feat_mask(row_test, col_test, H_feat, W_feat,
                               img_H=img_H, img_W=img_W,
                               pad_H=pad_H, pad_W=pad_W)
        expected = torch.tensor([True, True, False, False])
        assert torch.equal(mask, expected), \
            f"Expected {expected}, got {mask}"


# =====================================================================
# (3) 边界角度: emb 连续性 + 翻转等变关系
# =====================================================================

class TestBoundaryAngles:
    """Test embedding continuity at ±90° and equivariance at boundaries."""

    def test_emb_continuity_at_boundaries(self):
        """angle_to_emb should be continuous at ±90° (no jump in emb space).
        lim θ→±90° emb should approach boundary value smoothly.
        """
        # At θ = ±90° (in radians: ±π/2), 2θ = ±π
        # cos(±π) = -1, sin(±π) = 0 → emb = (-1, 0)
        for theta_deg in [-90.0, 90.0]:
            theta = torch.tensor(math.radians(theta_deg))
            emb = angle_to_emb(theta)
            expected = torch.tensor([-1.0, 0.0])
            assert torch.allclose(emb, expected, atol=1e-6), \
                f"θ={theta_deg}°: expected (-1, 0), got {emb}"

    def test_emb_continuity_near_plus_minus_90(self):
        """Small angle change near ±90° should produce small emb change."""
        for theta_deg in [-89.9, -89.0, -90.0, 89.0, 89.9, 90.0]:
            theta = torch.tensor(math.radians(theta_deg))
            theta_plus = torch.tensor(math.radians(min(theta_deg + 0.1, 89.999)))
            emb = angle_to_emb(theta)
            emb_plus = angle_to_emb(theta_plus)
            diff_norm = torch.norm(emb_plus - emb).item()
            assert diff_norm < 0.05, \
                f"θ={theta_deg}°: emb change {diff_norm:.4f} too large for Δθ=0.1°"

    @pytest.mark.parametrize("theta_deg", [-90, -89, 0, 89, 90])
    def test_horizontal_equivariance(self, theta_deg):
        """For horizontal flip: flip_emb(emb(θ), 'horizontal') ≈ emb(-θ).

        This is the core equivariance relation:
            emb(flip(θ)) = flip_emb(emb(θ))
        where flip(θ) = -θ for horizontal.
        """
        theta = torch.tensor(math.radians(theta_deg))
        theta_flipped = torch.tensor(math.radians(-theta_deg))

        emb_orig = angle_to_emb(theta)
        emb_flipped = angle_to_emb(theta_flipped)
        emb_transformed = flip_emb(emb_orig, 'horizontal')

        assert torch.allclose(emb_flipped, emb_transformed, atol=1e-6), \
            f"θ={theta_deg}°: emb(-θ)={emb_flipped} ≠ flip_emb(emb(θ))={emb_transformed}"

    @pytest.mark.parametrize("theta_deg", [-90, -89, 0, 89, 90])
    def test_vertical_equivariance(self, theta_deg):
        """Same as horizontal: vertical flip also θ→-θ in le90."""
        theta = torch.tensor(math.radians(theta_deg))
        theta_flipped = torch.tensor(math.radians(-theta_deg))

        emb_orig = angle_to_emb(theta)
        emb_flipped = angle_to_emb(theta_flipped)
        emb_transformed = flip_emb(emb_orig, 'vertical')

        assert torch.allclose(emb_flipped, emb_transformed, atol=1e-6), \
            f"θ={theta_deg}°: vertical equivariance failed"

    @pytest.mark.parametrize("theta_deg", [-90, -89, 0, 89, 90])
    def test_diagonal_equivariance(self, theta_deg):
        """Diagonal: θ→θ, so flip_emb = identity."""
        theta = torch.tensor(math.radians(theta_deg))

        emb_orig = angle_to_emb(theta)
        emb_flipped = angle_to_emb(theta)  # Same angle
        emb_transformed = flip_emb(emb_orig, 'diagonal')

        assert torch.allclose(emb_flipped, emb_transformed, atol=1e-6), \
            f"θ={theta_deg}°: diagonal equivariance failed"

    def test_equi_loss_zero_on_perfect_equivariance(self):
        """When predictions perfectly satisfy equivariance, L_equi = 0."""
        for direction in ['horizontal', 'vertical', 'diagonal']:
            theta_orig = torch.tensor(math.radians(45.0))
            emb_orig = angle_to_emb(theta_orig)
            # Perfect flip prediction
            emb_flip = flip_emb(emb_orig, direction)
            loss = equi_flip_loss(emb_orig, emb_flip, direction)
            assert loss.item() < 1e-6, \
                f"{direction}: perfect equivariance should give loss=0, got {loss.item()}"

    def test_equi_loss_positive_on_violation(self):
        """When predictions violate equivariance, L_equi > 0."""
        theta_orig = torch.tensor(math.radians(30.0))
        emb_orig = angle_to_emb(theta_orig)
        # Wrong flip prediction (use wrong direction)
        wrong_direction = 'horizontal'
        emb_wrong = emb_orig + torch.tensor([0.1, 0.1])  # Perturb
        loss = equi_flip_loss(emb_orig, emb_wrong, wrong_direction)
        assert loss.item() > 0, f"Loss should be positive for violation, got {loss.item()}"

    def test_emb_unit_circle(self):
        """angle_to_emb output should lie on unit circle for all θ."""
        thetas = torch.linspace(-math.pi / 2, math.pi / 2, 100)
        emb = angle_to_emb(thetas)
        norms = torch.norm(emb, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"emb norms should all be 1.0, got range [{norms.min()}, {norms.max()}]"

    def test_equi_loss_gradient_flows(self):
        """Verify gradient flows through equi_flip_loss."""
        theta_orig = torch.tensor(math.radians(45.0))
        emb_orig = angle_to_emb(theta_orig)

        # Create a leaf tensor with requires_grad=True
        pred_emb = torch.tensor([math.cos(math.radians(80.0)),
                                 math.sin(math.radians(80.0))],
                                requires_grad=True)

        loss = equi_flip_loss(emb_orig, pred_emb, 'horizontal')
        loss.backward()
        assert pred_emb.grad is not None, "Gradient should flow through equi_flip_loss"
        assert not torch.all(pred_emb.grad == 0), "Gradient should be non-zero"


# =====================================================================
# (4) INTEGRATION: mirror + valid_mask + equi_flip_loss on synthetic data
# =====================================================================

class TestIntegration:
    """End-to-end: mirror grid + extract embeddings + compute loss.

    Uses synthetic feature maps with known angles at grid positions.
    """

    def test_horizontal_flip_with_padding(self):
        """Simulate: 8x8 feature map, 6x6 valid (right padding = 2 cols).
        Place grab at (2, 3) with θ=30°. Mirror at (2, 4) with θ=-30°.
        L_equi should be 0 (perfect equivariance).
        """
        H_feat, W_feat = 8, 8
        img_H, img_W = 6, 6  # valid region
        pad_H, pad_W = 8, 8  # padded to full grid

        r_orig = torch.tensor([2])
        c_orig = torch.tensor([3])

        # Mirror position
        r_mir, c_mir = mirror_grid_indices(r_orig, c_orig, H_feat, W_feat, 'horizontal')
        assert r_mir.item() == 2
        assert c_mir.item() == 4

        # Both should be valid (within 6x6 valid region)
        mask_orig = valid_feat_mask(r_orig, c_orig, H_feat, W_feat,
                                    img_H, img_W, pad_H, pad_W)
        mask_mir = valid_feat_mask(r_mir, c_mir, H_feat, W_feat,
                                   img_H, img_W, pad_H, pad_W)
        assert mask_orig.item(), f"({r_orig.item()}, {c_orig.item()}) should be valid"
        assert mask_mir.item(), f"({r_mir.item()}, {c_mir.item()}) should be valid"

        # Compute equivariance loss with perfect predictions
        theta_orig = torch.tensor(math.radians(30.0))
        theta_flip = torch.tensor(math.radians(-30.0))  # Perfect flip

        emb_orig = angle_to_emb(theta_orig)
        emb_flip = angle_to_emb(theta_flip)
        loss = equi_flip_loss(emb_orig, emb_flip, 'horizontal')
        assert loss.item() < 1e-6, f"Perfect flip should give loss≈0, got {loss.item()}"

    def test_horizontal_flip_with_padding_excludes_invalid(self):
        """Place grab at (2, 6) — in padding zone (valid_w=6).
        valid_feat_mask should reject it.
        """
        H_feat, W_feat = 8, 8
        img_H, img_W = 6, 6
        pad_H, pad_W = 8, 8

        r_orig = torch.tensor([2])
        c_orig = torch.tensor([6])  # col=6 >= valid_w=6, INVALID

        mask = valid_feat_mask(r_orig, c_orig, H_feat, W_feat,
                               img_H, img_W, pad_H, pad_W)
        assert not mask.item(), "(2, 6) should be invalid (in padding zone)"

    def test_vertical_flip_mirror(self):
        """Verify vertical mirror: (r, c) → (H-1-r, c)."""
        H_feat, W_feat = 16, 8
        row = torch.tensor([0, 1, 7, 15])
        col = torch.tensor([3, 4, 2, 1])
        mr, mc = mirror_grid_indices(row, col, H_feat, W_feat, 'vertical')
        assert torch.equal(mr, torch.tensor([15, 14, 8, 0]))
        assert torch.equal(mc, col)

    def test_diagonal_flip_mirror(self):
        """Verify diagonal mirror: (r, c) → (H-1-r, W-1-c)."""
        H_feat, W_feat = 16, 8
        row = torch.tensor([0, 1, 7, 15])
        col = torch.tensor([3, 4, 2, 1])
        mr, mc = mirror_grid_indices(row, col, H_feat, W_feat, 'diagonal')
        assert torch.equal(mr, torch.tensor([15, 14, 8, 0]))
        assert torch.equal(mc, torch.tensor([4, 3, 5, 6]))