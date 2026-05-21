"""实验 / 试样 / 文件登记 + 处理结果写回。

所有函数都接收一个 sqlite3.Connection 作为第一个参数, 不在内部打开数据库,
这样脚本和 pipeline 可以在同一个事务里调用多个函数。

文件路径统一以 "相对项目根目录" 形式存入数据库, 跨机器迁移更友好。
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from ..utils.paths import PROJECT_ROOT, resolve_under_root


# --------------------------------------------------------------------------- #
# experiments / samples
# --------------------------------------------------------------------------- #
def ensure_default_experiment(
    conn: sqlite3.Connection,
    *,
    name: str,
    powder_material: str,
    substrate_material: str,
    laser_power_W: float,
    scan_speed_mm_per_min: float,
    powder_feed_rate_g_per_min: float,
    hatch_spacing_mm: float,
    notes: Optional[str] = None,
) -> int:
    """按 name 幂等插入一条 experiment 行, 返回其 id。"""
    row = conn.execute(
        "SELECT id FROM experiments WHERE name = ?", (name,)
    ).fetchone()
    if row is not None:
        return int(row["id"])
    cur = conn.execute(
        """
        INSERT INTO experiments (
            name, powder_material, substrate_material,
            laser_power_W, scan_speed_mm_per_min,
            powder_feed_rate_g_per_min, hatch_spacing_mm, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            powder_material,
            substrate_material,
            float(laser_power_W),
            float(scan_speed_mm_per_min),
            float(powder_feed_rate_g_per_min),
            float(hatch_spacing_mm),
            notes,
        ),
    )
    return int(cur.lastrowid)


def ensure_sample(
    conn: sqlite3.Connection,
    *,
    sample_id: str,
    experiment_id: int,
    B_mT: float,
    notes: Optional[str] = None,
) -> int:
    """按 sample_id 幂等插入一条 sample 行, 返回其 id。

    若 sample_id 已存在但 B_mT 不一致, 抛错而不是悄悄更新。
    """
    row = conn.execute(
        "SELECT id, B_mT FROM samples WHERE sample_id = ?", (sample_id,)
    ).fetchone()
    if row is not None:
        if abs(float(row["B_mT"]) - float(B_mT)) > 1e-9:
            raise ValueError(
                f"Sample '{sample_id}' already exists with B_mT={row['B_mT']}, "
                f"refusing to overwrite with B_mT={B_mT}"
            )
        return int(row["id"])
    cur = conn.execute(
        """
        INSERT INTO samples (experiment_id, sample_id, B_mT, notes)
        VALUES (?, ?, ?, ?)
        """,
        (int(experiment_id), sample_id, float(B_mT), notes),
    )
    return int(cur.lastrowid)


# --------------------------------------------------------------------------- #
# xtherm files
# --------------------------------------------------------------------------- #
def _to_relpath_under_root(p: Union[str, Path]) -> str:
    """规范化为相对项目根的 posix 路径; 不在根下则保留绝对路径。"""
    path = Path(p).resolve()
    try:
        rel = path.relative_to(PROJECT_ROOT)
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@dataclass
class RegisteredFile:
    id: int
    file_path: str
    file_size_bytes: int
    status: str


def register_xtherm_file(
    conn: sqlite3.Connection,
    *,
    sample_pk: int,
    file_path: Union[str, Path],
    width: int,
    height: int,
    dtype: str,
    endian: str,
    header_offset: int,
    temperature_scale: float,
    fps: Optional[float] = None,
    dx_mm_per_pixel: Optional[float] = None,
    dy_mm_per_pixel: Optional[float] = None,
    compute_sha256: bool = False,
    notes: Optional[str] = None,
) -> int:
    """登记单个 .xtherm 文件; 同路径已存在时返回原 id 而不报错。"""
    abs_path = resolve_under_root(file_path)
    if not abs_path.exists():
        raise FileNotFoundError(f"xtherm file not found: {abs_path}")
    if not abs_path.is_file():
        raise ValueError(f"Path is not a regular file: {abs_path}")

    rel_path = _to_relpath_under_root(abs_path)
    size = abs_path.stat().st_size

    existing = conn.execute(
        "SELECT id FROM xtherm_files WHERE file_path = ?", (rel_path,)
    ).fetchone()
    if existing is not None:
        return int(existing["id"])

    file_sha = _sha256_of_file(abs_path) if compute_sha256 else None

    # 估算总帧数 (header 之外的内容能整除单帧字节数才有意义, 失败则记 NULL)
    bytes_per_pixel = _bytes_per_pixel(dtype)
    bytes_per_frame = bytes_per_pixel * width * height
    payload = size - int(header_offset)
    estimated_frames = (
        int(payload // bytes_per_frame)
        if (bytes_per_frame > 0 and payload >= 0 and payload % bytes_per_frame == 0)
        else None
    )

    cur = conn.execute(
        """
        INSERT INTO xtherm_files (
            sample_id, file_path, file_size_bytes, file_sha256,
            width, height, dtype, endian, header_offset, temperature_scale,
            estimated_frames, fps, dx_mm_per_pixel, dy_mm_per_pixel,
            status, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'registered', ?)
        """,
        (
            int(sample_pk),
            rel_path,
            int(size),
            file_sha,
            int(width),
            int(height),
            str(dtype),
            str(endian),
            int(header_offset),
            float(temperature_scale),
            estimated_frames,
            None if fps is None else float(fps),
            None if dx_mm_per_pixel is None else float(dx_mm_per_pixel),
            None if dy_mm_per_pixel is None else float(dy_mm_per_pixel),
            notes,
        ),
    )
    return int(cur.lastrowid)


def _bytes_per_pixel(dtype: str) -> int:
    table = {
        "uint8": 1, "int8": 1,
        "uint16": 2, "int16": 2,
        "uint32": 4, "int32": 4, "float32": 4,
        "uint64": 8, "int64": 8, "float64": 8,
    }
    if dtype not in table:
        raise ValueError(f"Unsupported dtype '{dtype}'")
    return table[dtype]


# --------------------------------------------------------------------------- #
# 查询 / 状态管理
# --------------------------------------------------------------------------- #
def fetch_xtherm_file(conn: sqlite3.Connection, file_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM xtherm_files WHERE id = ?", (int(file_id),)
    ).fetchone()
    if row is None:
        raise LookupError(f"xtherm_files.id = {file_id} not found")
    return row


def list_files_by_status(
    conn: sqlite3.Connection,
    status: Optional[str] = None,
) -> List[sqlite3.Row]:
    if status is None:
        rows = conn.execute("SELECT * FROM xtherm_files ORDER BY id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM xtherm_files WHERE status = ? ORDER BY id", (status,)
        ).fetchall()
    return list(rows)


def update_file_status(
    conn: sqlite3.Connection,
    file_id: int,
    status: str,
    *,
    estimated_frames: Optional[int] = None,
    header_offset: Optional[int] = None,
    notes: Optional[str] = None,
) -> None:
    fields = ["status = ?", "last_status_at = datetime('now')"]
    params: list = [status]
    if estimated_frames is not None:
        fields.append("estimated_frames = ?")
        params.append(int(estimated_frames))
    if header_offset is not None:
        fields.append("header_offset = ?")
        params.append(int(header_offset))
    if notes is not None:
        fields.append("notes = ?")
        params.append(notes)
    params.append(int(file_id))
    conn.execute(
        f"UPDATE xtherm_files SET {', '.join(fields)} WHERE id = ?", tuple(params)
    )


# --------------------------------------------------------------------------- #
# 处理结果写回
# --------------------------------------------------------------------------- #
def upsert_processing_result(
    conn: sqlite3.Connection,
    *,
    xtherm_file_id: int,
    status: str,
    error_message: Optional[str] = None,
    n_frames: Optional[int] = None,
    Tmax_global: Optional[float] = None,
    Tmean_global: Optional[float] = None,
    Gmax_global: Optional[float] = None,
    Gmean_global: Optional[float] = None,
    feature_csv_path: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO processing_results (
            xtherm_file_id, status, error_message, n_frames,
            Tmax_global, Tmean_global, Gmax_global, Gmean_global,
            feature_csv_path, processed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(xtherm_file_id) DO UPDATE SET
            status = excluded.status,
            error_message = excluded.error_message,
            n_frames = excluded.n_frames,
            Tmax_global = excluded.Tmax_global,
            Tmean_global = excluded.Tmean_global,
            Gmax_global = excluded.Gmax_global,
            Gmean_global = excluded.Gmean_global,
            feature_csv_path = excluded.feature_csv_path,
            processed_at = datetime('now')
        """,
        (
            int(xtherm_file_id),
            status,
            error_message,
            None if n_frames is None else int(n_frames),
            Tmax_global,
            Tmean_global,
            Gmax_global,
            Gmean_global,
            feature_csv_path,
        ),
    )


def insert_frame_features(
    conn: sqlite3.Connection,
    *,
    xtherm_file_id: int,
    rows: Sequence[dict],
    fps: Optional[float] = None,
) -> int:
    """批量插入逐帧特征; 已存在的 (file_id, frame_index) 会被 REPLACE。

    rows: 每行包含 frame_index, Tmax, Tmean, Tstd, Gmax, Gmean, Gstd,
    high_temp_area。若提供 fps, 自动填充 time_s = frame_index / fps。
    """
    if not rows:
        return 0
    payload: List[tuple] = []
    for r in rows:
        time_s = (
            float(r["frame_index"]) / float(fps)
            if (fps is not None and fps > 0)
            else r.get("time_s")
        )
        payload.append(
            (
                int(xtherm_file_id),
                int(r["frame_index"]),
                time_s,
                float(r["Tmax"]),
                float(r["Tmean"]),
                float(r["Tstd"]),
                float(r["Gmax"]),
                float(r["Gmean"]),
                float(r["Gstd"]),
                int(r["high_temp_area"]),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO frame_features (
            xtherm_file_id, frame_index, time_s,
            Tmax, Tmean, Tstd, Gmax, Gmean, Gstd, high_temp_area
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)
