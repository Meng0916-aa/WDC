"""表面温度梯度计算。

约定:
- 温度矩阵 T 的 axis=0 是 y (height), axis=1 是 x (width); 与 xtherm_reader 输出
  shape [T, H, W] 一致;
- Gx = ∂T/∂x (单位 °C/mm), Gy = ∂T/∂y (单位 °C/mm), G = sqrt(Gx^2 + Gy^2);
- 缺少 dx_mm_per_pixel 或 dy_mm_per_pixel 时立刻抛错; 不假设默认值, 因为像素到
  毫米的换算来自相机标定, 错一个常数全篇都错;
- 输出 dtype 与输入一致 (内部强制 float32)。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def _validate_spacing(dx_mm_per_pixel: Optional[float], dy_mm_per_pixel: Optional[float]) -> Tuple[float, float]:
    if dx_mm_per_pixel is None or dy_mm_per_pixel is None:
        raise ValueError(
            "dx_mm_per_pixel and dy_mm_per_pixel are required to compute "
            "gradients in physical units (°C/mm). They come from camera "
            "calibration and must be measured before processing. "
            "Got dx={!r}, dy={!r}.".format(dx_mm_per_pixel, dy_mm_per_pixel)
        )
    if not (np.isfinite(dx_mm_per_pixel) and np.isfinite(dy_mm_per_pixel)):
        raise ValueError(
            f"dx/dy spacing must be finite numbers, got dx={dx_mm_per_pixel}, dy={dy_mm_per_pixel}"
        )
    if dx_mm_per_pixel <= 0 or dy_mm_per_pixel <= 0:
        raise ValueError(
            f"dx/dy spacing must be positive, got dx={dx_mm_per_pixel}, dy={dy_mm_per_pixel}"
        )
    return float(dx_mm_per_pixel), float(dy_mm_per_pixel)


def compute_gradients(
    T_frame: np.ndarray,
    *,
    dx_mm_per_pixel: Optional[float],
    dy_mm_per_pixel: Optional[float],
    gaussian_sigma_px: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单帧温度场的梯度。

    Parameters
    ----------
    T_frame : np.ndarray
        shape (H, W), 温度 (°C), dtype 任意数值类型 (内部转 float32)。
    dx_mm_per_pixel, dy_mm_per_pixel : float
        像素到毫米的换算 (来自相机标定)。
    gaussian_sigma_px : float
        梯度前对 T_frame 做高斯平滑的 sigma (像素); 0 表示不平滑。
        红外噪声较大时建议 0.5 ~ 1.5 px。

    Returns
    -------
    (Gx, Gy, G) : tuple of np.ndarray
        三者 shape 均为 (H, W), dtype float32, 单位 °C/mm。
    """
    if T_frame.ndim != 2:
        raise ValueError(f"T_frame must be 2D (H, W), got shape={T_frame.shape}")
    dx, dy = _validate_spacing(dx_mm_per_pixel, dy_mm_per_pixel)

    T = np.asarray(T_frame, dtype=np.float32)
    if gaussian_sigma_px and gaussian_sigma_px > 0:
        T = gaussian_filter(T, sigma=float(gaussian_sigma_px))

    # np.gradient 返回 (axis0_grad, axis1_grad) = (∂T/∂y_pix, ∂T/∂x_pix)
    Gy_pix, Gx_pix = np.gradient(T)
    Gx = (Gx_pix / np.float32(dx)).astype(np.float32, copy=False)
    Gy = (Gy_pix / np.float32(dy)).astype(np.float32, copy=False)
    G = np.sqrt(Gx * Gx + Gy * Gy, dtype=np.float32)
    return Gx, Gy, G


def compute_gradients_stack(
    T_stack: np.ndarray,
    *,
    dx_mm_per_pixel: Optional[float],
    dy_mm_per_pixel: Optional[float],
    gaussian_sigma_px: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对 shape (T, H, W) 的温度序列逐帧计算梯度, 返回 (Gx, Gy, G) 同形数组。

    内存占用约为输入的 3x; 大文件优先用 pipeline 的逐帧路径。
    """
    if T_stack.ndim != 3:
        raise ValueError(f"T_stack must be 3D (T, H, W), got shape={T_stack.shape}")
    dx, dy = _validate_spacing(dx_mm_per_pixel, dy_mm_per_pixel)

    T = np.asarray(T_stack, dtype=np.float32)
    if gaussian_sigma_px and gaussian_sigma_px > 0:
        # 只在空间维度做平滑; 时间维 sigma=0
        T = gaussian_filter(T, sigma=(0.0, float(gaussian_sigma_px), float(gaussian_sigma_px)))

    Gy_pix, Gx_pix = np.gradient(T, axis=(1, 2))
    Gx = (Gx_pix / np.float32(dx)).astype(np.float32, copy=False)
    Gy = (Gy_pix / np.float32(dy)).astype(np.float32, copy=False)
    G = np.sqrt(Gx * Gx + Gy * Gy, dtype=np.float32)
    return Gx, Gy, G
