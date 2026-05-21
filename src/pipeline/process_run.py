"""数据库驱动的单文件处理流程。

输入: 数据库行 (xtherm_files 表的一行) + AppConfig。
步骤:
1. 读取 .xtherm -> 温度 cube (float32, °C);
2. 逐帧计算梯度 (Gx, Gy, G), 并提取特征;
3. 把逐帧特征写入 frame_features 表;
4. 在 processing_results 表里写入文件级 summary (Tmax_global / Tmean_global / ...);
5. 把特征 CSV 落到 data/features/<file_id>__<sample_id>__<basename>.csv;
6. 更新 xtherm_files.status -> 'processed' | 'error'。

错误处理: 任何失败都会被捕获, 写回 processing_results.status='error', 同时
xtherm_files.status='error', 并把异常信息保留在数据库里供事后排查; 函数不再
向上抛出 (只在主程序看到 ProcessOutcome.ok=False)。
"""
from __future__ import annotations

import logging
import sqlite3
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..db.registry import (
    fetch_xtherm_file,
    insert_frame_features,
    update_file_status,
    upsert_processing_result,
)
from ..io.xtherm_reader import read_xtherm
from ..processing.features import (
    FRAME_FEATURE_COLUMNS,
    extract_frame_features,
)
from ..processing.gradients import compute_gradients
from ..utils.config import AppConfig
from ..utils.paths import resolve_under_root


logger = logging.getLogger(__name__)


@dataclass
class ProcessOutcome:
    file_id: int
    ok: bool
    n_frames: int = 0
    feature_csv: Optional[str] = None
    error: Optional[str] = None


def _resolve_dx_dy(row: sqlite3.Row, cfg: AppConfig) -> tuple[Optional[float], Optional[float]]:
    """xtherm_files 行里的标定优先, 缺失则回落到 configs/default.yaml。"""
    dx = row["dx_mm_per_pixel"] if row["dx_mm_per_pixel"] is not None else cfg.camera.dx_mm_per_pixel
    dy = row["dy_mm_per_pixel"] if row["dy_mm_per_pixel"] is not None else cfg.camera.dy_mm_per_pixel
    return dx, dy


def _feature_csv_path(cfg: AppConfig, file_id: int, sample_pk: int, source_path: Path) -> Path:
    out_dir = cfg.paths.data_features_abs()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = source_path.stem
    return out_dir / f"file{file_id:06d}__sample{sample_pk:04d}__{safe}.csv"


def process_registered_file(
    conn: sqlite3.Connection,
    file_id: int,
    cfg: AppConfig,
) -> ProcessOutcome:
    """处理 xtherm_files.id = file_id 的一条记录。"""
    row = fetch_xtherm_file(conn, file_id)
    abs_path = resolve_under_root(row["file_path"])
    logger.info("Processing file_id=%d path=%s", file_id, abs_path)

    try:
        # 1. 读取温度 cube
        T_cube = read_xtherm(
            abs_path,
            width=row["width"],
            height=row["height"],
            dtype=row["dtype"],
            endian=row["endian"],
            header_offset=row["header_offset"],
            temperature_scale=row["temperature_scale"],
            max_frames=cfg.processing.max_frames,
        )
        n_frames = int(T_cube.shape[0])
        if n_frames == 0:
            raise ValueError(f"No frames decoded for file_id={file_id}")

        # 2. 标定
        dx, dy = _resolve_dx_dy(row, cfg)
        fps = row["fps"] if row["fps"] is not None else cfg.camera.fps

        # 3. 逐帧特征
        rows: List[dict] = []
        Tmax_g = -np.inf
        Tmean_sum = 0.0
        Gmax_g = -np.inf
        Gmean_sum = 0.0

        for idx in range(n_frames):
            T = T_cube[idx]
            _, _, G = compute_gradients(
                T,
                dx_mm_per_pixel=dx,
                dy_mm_per_pixel=dy,
                gaussian_sigma_px=cfg.processing.gaussian_sigma_px,
            )
            feat = extract_frame_features(
                T,
                G,
                frame_index=idx,
                time_s=(idx / fps) if (fps and fps > 0) else None,
                high_temp_threshold_C=cfg.processing.high_temp_threshold_C,
            )
            rows.append(feat)
            if feat["Tmax"] > Tmax_g:
                Tmax_g = feat["Tmax"]
            if feat["Gmax"] > Gmax_g:
                Gmax_g = feat["Gmax"]
            Tmean_sum += feat["Tmean"]
            Gmean_sum += feat["Gmean"]
            if cfg.processing.log_every_frames and (idx + 1) % cfg.processing.log_every_frames == 0:
                logger.info("  frame %d / %d", idx + 1, n_frames)

        Tmean_g = Tmean_sum / n_frames
        Gmean_g = Gmean_sum / n_frames

        # 4. 写 frame_features (事务内, 一次性 executemany)
        conn.execute("BEGIN")
        try:
            insert_frame_features(
                conn,
                xtherm_file_id=file_id,
                rows=rows,
                fps=fps,
            )

            # 5. 落特征 CSV
            csv_path = _feature_csv_path(cfg, file_id, int(row["sample_id"]), abs_path)
            df = pd.DataFrame(rows, columns=list(FRAME_FEATURE_COLUMNS))
            df.to_csv(csv_path, index=False)

            # 6. processing_results + xtherm_files 状态
            upsert_processing_result(
                conn,
                xtherm_file_id=file_id,
                status="success",
                error_message=None,
                n_frames=n_frames,
                Tmax_global=float(Tmax_g),
                Tmean_global=float(Tmean_g),
                Gmax_global=float(Gmax_g),
                Gmean_global=float(Gmean_g),
                feature_csv_path=str(csv_path.relative_to(resolve_under_root(".")).as_posix()),
            )
            update_file_status(conn, file_id, "processed", estimated_frames=n_frames)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        logger.info(
            "Done file_id=%d frames=%d Tmax=%.2f Tmean=%.2f Gmax=%.3f Gmean=%.3f",
            file_id, n_frames, Tmax_g, Tmean_g, Gmax_g, Gmean_g,
        )
        return ProcessOutcome(
            file_id=file_id,
            ok=True,
            n_frames=n_frames,
            feature_csv=str(csv_path),
        )

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(limit=4)
        logger.error("Processing failed for file_id=%d: %s", file_id, e)
        try:
            conn.execute("BEGIN")
            upsert_processing_result(
                conn,
                xtherm_file_id=file_id,
                status="error",
                error_message=f"{type(e).__name__}: {e}\n{tb}",
            )
            update_file_status(conn, file_id, "error", notes=f"{type(e).__name__}: {e}")
            conn.execute("COMMIT")
        except Exception:  # noqa: BLE001
            conn.execute("ROLLBACK")
            logger.exception("Failed to record error state for file_id=%d", file_id)
        return ProcessOutcome(file_id=file_id, ok=False, error=str(e))
