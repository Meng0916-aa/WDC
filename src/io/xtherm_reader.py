"""Xiris VXIR-3000 .xtherm 文件读取器。

文件布局假设 (与项目当前测试 / 默认配置一致):
    [header_offset 字节] + N * (height * width * bytes_per_pixel) 字节
    每帧按 row-major (C-order) 排列, axis=0 是 y / height, axis=1 是 x / width。
真实 Xiris 设备的 header 长度需要通过 format_probe 配合厂商导出说明确认;
本模块不会去 "猜" header_offset, 也不会做 silent reshape。

输入: 路径 + 显式形状参数 (width / height / dtype / endian /
header_offset / temperature_scale)。
输出: numpy.float32 数组, shape = [T, H, W], 单位 °C。

不变量 (任何一项不满足都立刻抛错):
1. 文件存在;
2. (file_size - header_offset) 可被 frame_bytes 整除;
3. dtype 在白名单内;
4. 若指定了 expected_frames, 与计算出的实际帧数完全相等。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


# 受支持的原始数据类型 (与 Xiris 实际可能导出的 raw dtype 对应)
_DTYPE_TABLE = {
    "uint8": np.uint8, "int8": np.int8,
    "uint16": np.uint16, "int16": np.int16,
    "uint32": np.uint32, "int32": np.int32,
    "float32": np.float32, "float64": np.float64,
}


@dataclass(frozen=True)
class XthermReadParams:
    width: int
    height: int
    dtype: str = "uint16"
    endian: str = "little"
    header_offset: int = 0
    temperature_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"width and height must be positive, got width={self.width}, "
                f"height={self.height}"
            )
        if self.dtype not in _DTYPE_TABLE:
            raise ValueError(
                f"Unsupported dtype '{self.dtype}'. "
                f"Allowed: {sorted(_DTYPE_TABLE.keys())}"
            )
        if self.endian not in ("little", "big"):
            raise ValueError(
                f"endian must be 'little' or 'big', got '{self.endian}'"
            )
        if self.header_offset < 0:
            raise ValueError(f"header_offset must be >= 0, got {self.header_offset}")
        if not np.isfinite(self.temperature_scale):
            raise ValueError(f"temperature_scale must be finite, got {self.temperature_scale}")


@dataclass(frozen=True)
class XthermFileLayout:
    file_size: int
    header_offset: int
    bytes_per_pixel: int
    frame_bytes: int
    payload_bytes: int
    n_frames: int


def _numpy_dtype(params: XthermReadParams) -> np.dtype:
    base = np.dtype(_DTYPE_TABLE[params.dtype])
    # 1 字节类型没有字节序; 多字节类型按 endian 显式标注。
    if base.itemsize == 1:
        return base
    prefix = "<" if params.endian == "little" else ">"
    return np.dtype(prefix + base.char)


def compute_expected_layout(
    file_path: Union[str, Path],
    params: XthermReadParams,
) -> XthermFileLayout:
    """根据文件大小与参数计算理论布局; 不能整除时抛错。

    这里就是 "拒绝 silent reshape" 的核心: 必须能整除才返回 layout。
    """
    abs_path = Path(file_path)
    if not abs_path.exists():
        raise FileNotFoundError(f"xtherm file not found: {abs_path}")
    if not abs_path.is_file():
        raise ValueError(f"Path is not a regular file: {abs_path}")

    size = abs_path.stat().st_size
    bpp = _numpy_dtype(params).itemsize
    frame_bytes = bpp * params.width * params.height
    payload = size - params.header_offset

    if payload < 0:
        raise ValueError(
            f"header_offset ({params.header_offset}) larger than file size "
            f"({size}) for {abs_path}"
        )
    if frame_bytes <= 0:
        raise ValueError(
            f"Invalid frame size: width={params.width}, height={params.height}, "
            f"dtype={params.dtype}"
        )
    if payload % frame_bytes != 0:
        # 给出排查所需的全部数字, 而不是一句"shape mismatch"。
        n_full = payload // frame_bytes
        leftover = payload % frame_bytes
        raise ValueError(
            "xtherm file size does not fit an integer number of frames "
            "(silent reshape disabled).\n"
            f"  file        : {abs_path}\n"
            f"  file_size   : {size} bytes\n"
            f"  header_off  : {params.header_offset} bytes\n"
            f"  width x H   : {params.width} x {params.height}\n"
            f"  dtype       : {params.dtype} ({bpp} byte/pixel)\n"
            f"  frame_bytes : {frame_bytes}\n"
            f"  payload     : {payload}\n"
            f"  n_full_frame: {n_full}, leftover_bytes: {leftover}\n"
            "Hint: run format_probe / scripts/probe_registered_file.py to "
            "search the correct header_offset and (W, H)."
        )

    return XthermFileLayout(
        file_size=size,
        header_offset=params.header_offset,
        bytes_per_pixel=bpp,
        frame_bytes=frame_bytes,
        payload_bytes=payload,
        n_frames=int(payload // frame_bytes),
    )


def read_xtherm(
    file_path: Union[str, Path],
    *,
    width: int,
    height: int,
    dtype: str = "uint16",
    endian: str = "little",
    header_offset: int = 0,
    temperature_scale: float = 0.1,
    expected_frames: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """读取 .xtherm, 返回 float32 摄氏度 cube, shape = [T, H, W]。

    Parameters
    ----------
    file_path : Path-like
        .xtherm 文件路径 (绝对或相对项目根目录都行, 但本函数不做项目根解析,
        由调用方 / pipeline 负责)。
    width, height : int
        单帧尺寸 (像素)。
    dtype : str
        原始像素 dtype, 默认 'uint16'。
    endian : str
        多字节 dtype 的字节序, 'little' 或 'big'。
    header_offset : int
        文件起始处需要跳过的字节数。
    temperature_scale : float
        raw_value * scale = temperature in °C。
    expected_frames : int | None
        若提供, 实际帧数必须与之相等, 否则抛错。
    max_frames : int | None
        只读取前 max_frames 帧 (帧数仍按完整文件校验; 这里只截断返回数组)。

    Returns
    -------
    np.ndarray
        dtype float32, shape [T, H, W], 单位 °C, C-contiguous。
    """
    params = XthermReadParams(
        width=int(width),
        height=int(height),
        dtype=str(dtype),
        endian=str(endian),
        header_offset=int(header_offset),
        temperature_scale=float(temperature_scale),
    )
    layout = compute_expected_layout(file_path, params)

    if expected_frames is not None and expected_frames != layout.n_frames:
        raise ValueError(
            f"Frame count mismatch for {file_path}: "
            f"expected={expected_frames}, actual={layout.n_frames}"
        )

    np_dt = _numpy_dtype(params)
    n_frames = layout.n_frames
    if max_frames is not None:
        if max_frames < 0:
            raise ValueError(f"max_frames must be >= 0, got {max_frames}")
        n_frames = min(n_frames, int(max_frames))

    count = n_frames * params.width * params.height
    with open(file_path, "rb") as f:
        if params.header_offset:
            f.seek(params.header_offset)
        flat = np.fromfile(f, dtype=np_dt, count=count)

    if flat.size != count:
        # 兜底: 在已经做过 layout 检查的情况下不应发生, 但仍然显式校验。
        raise ValueError(
            f"Short read on {file_path}: got {flat.size} elements, expected {count}"
        )

    # 显式 reshape, 顺序为 [T, H, W]。frames 在最外层, 行优先。
    raw = flat.reshape(n_frames, params.height, params.width)
    temps = raw.astype(np.float32, copy=True) * np.float32(params.temperature_scale)
    return np.ascontiguousarray(temps)
