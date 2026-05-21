"""Tests for src/processing/gradients.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.processing.gradients import compute_gradients, compute_gradients_stack


def test_gradients_constant_field_is_zero():
    T = np.full((6, 8), 100.0, dtype=np.float32)
    Gx, Gy, G = compute_gradients(T, dx_mm_per_pixel=0.05, dy_mm_per_pixel=0.05)
    assert Gx.shape == Gy.shape == G.shape == T.shape
    assert np.all(Gx == 0)
    assert np.all(Gy == 0)
    assert np.all(G == 0)


def test_gradients_linear_x_ramp():
    """T(y,x) = 10 * x_px, dx=0.5 mm/px  => ∂T/∂x = 10/0.5 = 20 °C/mm in the interior."""
    H, W = 6, 8
    T = np.tile(np.arange(W, dtype=np.float32) * 10.0, (H, 1))
    Gx, Gy, G = compute_gradients(T, dx_mm_per_pixel=0.5, dy_mm_per_pixel=0.5)
    # interior (excluding edges) should have Gx == 20.0 exactly
    np.testing.assert_allclose(Gx[:, 1:-1], 20.0, rtol=0, atol=1e-5)
    # Gy 应处处为 0
    np.testing.assert_allclose(Gy, 0.0, atol=1e-5)
    # G 在内部 == |Gx| == 20.0
    np.testing.assert_allclose(G[:, 1:-1], 20.0, rtol=0, atol=1e-5)


def test_gradients_linear_y_ramp():
    H, W = 6, 8
    T = np.tile(np.arange(H, dtype=np.float32)[:, None] * 5.0, (1, W))
    Gx, Gy, G = compute_gradients(T, dx_mm_per_pixel=0.2, dy_mm_per_pixel=0.1)
    # ∂T/∂y_pix = 5; dy=0.1 -> Gy = 5/0.1 = 50 °C/mm
    np.testing.assert_allclose(Gy[1:-1, :], 50.0, atol=1e-4)
    np.testing.assert_allclose(Gx, 0.0, atol=1e-5)
    np.testing.assert_allclose(G[1:-1, :], 50.0, atol=1e-4)


def test_gradients_missing_spacing_raises_dx():
    T = np.zeros((6, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="dx_mm_per_pixel"):
        compute_gradients(T, dx_mm_per_pixel=None, dy_mm_per_pixel=0.05)


def test_gradients_missing_spacing_raises_dy():
    T = np.zeros((6, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="dx_mm_per_pixel and dy_mm_per_pixel"):
        compute_gradients(T, dx_mm_per_pixel=0.05, dy_mm_per_pixel=None)


def test_gradients_zero_spacing_raises():
    T = np.zeros((6, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="must be positive"):
        compute_gradients(T, dx_mm_per_pixel=0.0, dy_mm_per_pixel=0.05)


def test_gradients_negative_spacing_raises():
    T = np.zeros((6, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="must be positive"):
        compute_gradients(T, dx_mm_per_pixel=0.05, dy_mm_per_pixel=-0.05)


def test_gradients_non_2d_raises():
    T = np.zeros((3, 6, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 2D"):
        compute_gradients(T, dx_mm_per_pixel=0.05, dy_mm_per_pixel=0.05)


def test_gradients_dtype_is_float32():
    T = np.arange(48, dtype=np.float64).reshape(6, 8)
    Gx, Gy, G = compute_gradients(T, dx_mm_per_pixel=0.05, dy_mm_per_pixel=0.05)
    assert Gx.dtype == np.float32
    assert Gy.dtype == np.float32
    assert G.dtype == np.float32


def test_gradients_with_smoothing_does_not_change_shape():
    T = np.random.default_rng(0).normal(100, 20, size=(20, 30)).astype(np.float32)
    Gx, Gy, G = compute_gradients(
        T, dx_mm_per_pixel=0.1, dy_mm_per_pixel=0.1, gaussian_sigma_px=1.0
    )
    assert Gx.shape == Gy.shape == G.shape == T.shape


def test_gradient_stack_consistency_with_single_frame():
    rng = np.random.default_rng(42)
    T_stack = rng.normal(500, 50, size=(3, 6, 8)).astype(np.float32)
    Gx_s, Gy_s, G_s = compute_gradients_stack(
        T_stack, dx_mm_per_pixel=0.05, dy_mm_per_pixel=0.05
    )
    for t in range(T_stack.shape[0]):
        Gx, Gy, G = compute_gradients(T_stack[t], dx_mm_per_pixel=0.05, dy_mm_per_pixel=0.05)
        np.testing.assert_allclose(Gx, Gx_s[t], atol=1e-5)
        np.testing.assert_allclose(Gy, Gy_s[t], atol=1e-5)
        np.testing.assert_allclose(G, G_s[t], atol=1e-5)


def test_gradient_stack_non_3d_raises():
    T = np.zeros((6, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 3D"):
        compute_gradients_stack(T, dx_mm_per_pixel=0.05, dy_mm_per_pixel=0.05)
