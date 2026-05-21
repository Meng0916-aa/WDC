"""未知 .xtherm 文件的格式探测。

输入: 路径 + 候选 header_offset 列表 + (W, H, dtype, endian, temperature_scale)。
做法: 对每个候选 offset, 计算 "(file_size - offset) 是否可被 frame_bytes 整除"; 整除的视为候选,
并对该候选采样 first/middle/last 帧, 估算温度范围。

设计约束:
- 只以只读方式访问文件, 绝不修改;
- 不输出 ndarray 整体, 避免大文件吃掉内存;
- 输出结构化报告 (dataclass + dict), 由调用方决定是否写回数据库 / 打印。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .xtherm_reader import (
    XthermReadParams,
    _numpy_dtype,
    compute_expected_layout,
)


@dataclass
class ProbeCandidate:
    header_offset: int
    n_frames: int
    frame_bytes: int
    payload_bytes: int
    sample_T_min_C: Optional[float] = None
    sample_T_max_C: Optional[float] = None
    sample_T_mean_C: Optional[float] = None
    sampled_frame_indices: List[int] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeReport:
    file_path: str
    file_size: int
    width: int
    height: int
    dtype: str
    endian: str
    temperature_scale: float
    candidates: List[ProbeCandidate]
    best: Optional[ProbeCandidate] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_size": self.file_size,
            "width": self.width,
            "height": self.height,
            "dtype": self.dtype,
            "endian": self.endian,
            "temperature_scale": self.temperature_scale,
            "candidates": [c.to_dict() for c in self.candidates],
            "best": None if self.best is None else self.best.to_dict(),
        }


def _resolve_sample_frames(
    spec: Sequence[Any], n_frames: int
) -> List[int]:
    """把 config 中的 [0, 'middle', -1] 解析为合法帧索引列表。"""
    out: List[int] = []
    if n_frames <= 0:
        return out
    for token in spec:
        if isinstance(token, str):
            if token == "middle":
                idx = n_frames // 2
            elif token in ("first", "start"):
                idx = 0
            elif token in ("last", "end"):
                idx = n_frames - 1
            else:
                continue
        else:
            idx = int(token)
            if idx < 0:
                idx += n_frames
        if 0 <= idx < n_frames and idx not in out:
            out.append(idx)
    return out


def _read_one_frame(
    file_path: Union[str, Path],
    *,
    header_offset: int,
    frame_index: int,
    width: int,
    height: int,
    np_dt: np.dtype,
    temperature_scale: float,
) -> np.ndarray:
    frame_bytes = np_dt.itemsize * width * height
    offset = header_offset + frame_index * frame_bytes
    with open(file_path, "rb") as f:
        f.seek(offset)
        flat = np.fromfile(f, dtype=np_dt, count=width * height)
    if flat.size != width * height:
        raise ValueError(
            f"Short read while sampling frame {frame_index} at offset={offset}"
        )
    return flat.reshape(height, width).astype(np.float32) * np.float32(temperature_scale)


def probe_xtherm(
    file_path: Union[str, Path],
    *,
    width: int,
    height: int,
    dtype: str = "uint16",
    endian: str = "little",
    temperature_scale: float = 0.1,
    header_offset_candidates: Sequence[int] = (0, 128, 256, 512, 1024, 2048, 4096),
    sample_frames_spec: Sequence[Any] = (0, "middle", -1),
    plausible_temp_range_C: Tuple[float, float] = (-50.0, 3500.0),
) -> ProbeReport:
    """逐个尝试候选 header_offset, 报告每个能整除的方案。

    "best" 候选选择规则: 先要求采样温度全部落在 plausible_temp_range_C 内,
    再按 n_frames 较大者优先 (更多帧通常意味着是真实数据而不是边角碎块)。
    """
    abs_path = Path(file_path)
    if not abs_path.exists():
        raise FileNotFoundError(f"xtherm file not found: {abs_path}")
    file_size = abs_path.stat().st_size

    # 解析 dtype 一次, 后续重用
    base_params = XthermReadParams(
        width=int(width), height=int(height),
        dtype=str(dtype), endian=str(endian),
        header_offset=0, temperature_scale=float(temperature_scale),
    )
    np_dt = _numpy_dtype(base_params)

    candidates: List[ProbeCandidate] = []
    for offset in header_offset_candidates:
        try:
            layout = compute_expected_layout(
                abs_path,
                XthermReadParams(
                    width=int(width), height=int(height),
                    dtype=str(dtype), endian=str(endian),
                    header_offset=int(offset),
                    temperature_scale=float(temperature_scale),
                ),
            )
        except ValueError as e:
            candidates.append(
                ProbeCandidate(
                    header_offset=int(offset),
                    n_frames=0,
                    frame_bytes=np_dt.itemsize * int(width) * int(height),
                    payload_bytes=max(0, file_size - int(offset)),
                    note=f"not divisible ({type(e).__name__})",
                )
            )
            continue
        except FileNotFoundError:
            raise
        cand = ProbeCandidate(
            header_offset=int(offset),
            n_frames=layout.n_frames,
            frame_bytes=layout.frame_bytes,
            payload_bytes=layout.payload_bytes,
        )
        # 采样几帧估算温度范围
        sample_idx = _resolve_sample_frames(sample_frames_spec, layout.n_frames)
        cand.sampled_frame_indices = sample_idx
        if sample_idx:
            mins, maxs, sums, counts = [], [], 0.0, 0
            for idx in sample_idx:
                frame = _read_one_frame(
                    abs_path,
                    header_offset=int(offset),
                    frame_index=idx,
                    width=int(width),
                    height=int(height),
                    np_dt=np_dt,
                    temperature_scale=float(temperature_scale),
                )
                mins.append(float(frame.min()))
                maxs.append(float(frame.max()))
                sums += float(frame.sum())
                counts += frame.size
            cand.sample_T_min_C = min(mins)
            cand.sample_T_max_C = max(maxs)
            cand.sample_T_mean_C = sums / counts if counts > 0 else None
        candidates.append(cand)

    # 选 best
    lo, hi = plausible_temp_range_C
    plausible = [
        c for c in candidates
        if c.n_frames > 0
        and c.sample_T_min_C is not None
        and lo <= c.sample_T_min_C
        and c.sample_T_max_C is not None
        and c.sample_T_max_C <= hi
    ]
    best: Optional[ProbeCandidate] = None
    if plausible:
        best = max(plausible, key=lambda c: c.n_frames)
    elif candidates:
        # 退化情况: 没有完全 "合理" 的, 退而选择 n_frames 最多的整除方案
        divisible = [c for c in candidates if c.n_frames > 0]
        if divisible:
            best = max(divisible, key=lambda c: c.n_frames)

    return ProbeReport(
        file_path=str(abs_path),
        file_size=file_size,
        width=int(width),
        height=int(height),
        dtype=str(dtype),
        endian=str(endian),
        temperature_scale=float(temperature_scale),
        candidates=candidates,
        best=best,
    )
