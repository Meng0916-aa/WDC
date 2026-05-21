"""逐帧特征提取。

每帧返回一个 dict, 字段顺序固定为 FRAME_FEATURE_COLUMNS, 方便:
- 批量写入数据库 (frame_features 表);
- 直接转 pandas.DataFrame 导出 CSV。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# 写入 frame_features 表 / 导出 CSV 时遵循的列顺序
FRAME_FEATURE_COLUMNS = (
    "frame_index",
    "time_s",
    "Tmax",
    "Tmean",
    "Tstd",
    "Gmax",
    "Gmean",
    "Gstd",
    "high_temp_area",
)


def extract_frame_features(
    T_frame: np.ndarray,
    G_frame: np.ndarray,
    *,
    frame_index: int,
    time_s: Optional[float] = None,
    high_temp_threshold_C: float = 1000.0,
) -> Dict[str, float]:
    """计算单帧特征。

    Parameters
    ----------
    T_frame : np.ndarray
        shape (H, W), 温度 (°C)。
    G_frame : np.ndarray
        shape (H, W), 梯度幅值 (°C/mm), 由 gradients.compute_gradients 返回。
    frame_index : int
        当前帧索引 (从 0 开始)。
    time_s : float | None
        当前帧的时间戳 (秒); 由调用方根据 fps 推算, 这里只是透传。
    high_temp_threshold_C : float
        高温像素阈值; 高于该值的像素数量记入 high_temp_area。

    Returns
    -------
    dict
        键值为 FRAME_FEATURE_COLUMNS。Tmax/Tmean/Tstd 单位 °C,
        Gmax/Gmean/Gstd 单位 °C/mm, high_temp_area 单位 像素。
    """
    if T_frame.ndim != 2 or G_frame.ndim != 2:
        raise ValueError(
            f"T_frame and G_frame must be 2D, got T={T_frame.shape}, G={G_frame.shape}"
        )
    if T_frame.shape != G_frame.shape:
        raise ValueError(
            f"T_frame and G_frame shape mismatch: T={T_frame.shape}, G={G_frame.shape}"
        )

    T = np.asarray(T_frame, dtype=np.float32)
    G = np.asarray(G_frame, dtype=np.float32)

    high_mask = T > np.float32(high_temp_threshold_C)
    return {
        "frame_index": int(frame_index),
        "time_s": None if time_s is None else float(time_s),
        "Tmax": float(T.max()),
        "Tmean": float(T.mean()),
        "Tstd": float(T.std()),
        "Gmax": float(G.max()),
        "Gmean": float(G.mean()),
        "Gstd": float(G.std()),
        "high_temp_area": int(high_mask.sum()),
    }
