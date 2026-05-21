"""pytest 全局 fixture。

提供三种 fixture:
1. tiny_xtherm_params: 8x6 / 3 frames / uint16 / scale=0.1 的最小配置;
2. tiny_xtherm_file:  一个真正的临时 .xtherm 二进制文件 (含可预测内容);
3. tmp_db_path:       一个临时数据库路径, 以及打开+初始化好的连接。

这些 fixture 不构成 "synthetic experiment pipeline" —— 它们的存在只是为了
让 reader / probe / gradient / features / db 五个模块的单元测试可以在没有
真实 Xiris 数据时仍然可执行。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

# 让 `import src...` 在 pytest 直接运行 (无 pip install -e .) 时也能工作。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class TinyXthermParams:
    width: int = 8
    height: int = 6
    frames: int = 3
    dtype: str = "uint16"
    endian: str = "little"
    header_offset: int = 0
    temperature_scale: float = 0.1


@pytest.fixture
def tiny_xtherm_params() -> TinyXthermParams:
    return TinyXthermParams()


def _make_tiny_payload(p: TinyXthermParams) -> Tuple[np.ndarray, np.ndarray]:
    """生成 (raw_uint16, expected_temps_float32 in °C) 一对张量。

    每帧 = 帧内常数 * 100 + 一个 ramp; 三帧之间帧间温度有差异, 方便
    在 features 测试里检查 Tmax / Tmean / Tstd 是否随帧变化。
    """
    rng = np.random.default_rng(seed=20260521)
    base_per_frame = np.array([100, 250, 400], dtype=np.uint16)  # raw 100,250,400 -> 10,25,40 °C
    raw = np.empty((p.frames, p.height, p.width), dtype=np.uint16)
    for t in range(p.frames):
        ramp = rng.integers(0, 50, size=(p.height, p.width), dtype=np.uint16)
        raw[t] = base_per_frame[t] + ramp
    expected = raw.astype(np.float32) * np.float32(p.temperature_scale)
    return raw, expected


@pytest.fixture
def tiny_xtherm_file(
    tmp_path: Path, tiny_xtherm_params: TinyXthermParams
) -> Tuple[Path, np.ndarray, np.ndarray, TinyXthermParams]:
    """写一个最小的临时 .xtherm 文件, 返回 (path, raw_uint16, expected_temps_C, params)。"""
    raw, expected = _make_tiny_payload(tiny_xtherm_params)
    file_path = tmp_path / "tiny.xtherm"
    # endian: 测试默认 little, 这里直接落 raw uint16 -> 在小端机器上等同
    raw.astype(f"<u2", copy=False).tofile(file_path)
    return file_path, raw, expected, tiny_xtherm_params


@pytest.fixture
def tiny_xtherm_file_with_header(
    tmp_path: Path, tiny_xtherm_params: TinyXthermParams
) -> Tuple[Path, np.ndarray, np.ndarray, int, TinyXthermParams]:
    """带 128 字节 header 的版本, 用于测试 format_probe 能识别非零 offset。"""
    raw, expected = _make_tiny_payload(tiny_xtherm_params)
    header_offset = 128
    file_path = tmp_path / "tiny_with_header.xtherm"
    with open(file_path, "wb") as f:
        f.write(b"\xAB" * header_offset)
        raw.astype("<u2", copy=False).tofile(f)
    return file_path, raw, expected, header_offset, tiny_xtherm_params


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "thermal_test.db"
