"""Export per-frame features from the SQLite database to a single tidy CSV.

Each row joins frame_features with xtherm_files / samples / experiments so
downstream scripts (论文图表 / PyTorch dataset) can read one wide CSV
without having to touch SQL.

Usage:
    python scripts/export_features_from_database.py --config configs/default.yaml
    python scripts/export_features_from_database.py --config configs/default.yaml \
        --output data/features/all_features.csv --B-mT 80
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import open_db  # noqa: E402
from src.utils import load_config, resolve_under_root, setup_logging  # noqa: E402


SELECT_SQL = """
SELECT
    ff.xtherm_file_id        AS xtherm_file_id,
    ff.frame_index           AS frame_index,
    ff.time_s                AS time_s,
    ff.Tmax                  AS Tmax,
    ff.Tmean                 AS Tmean,
    ff.Tstd                  AS Tstd,
    ff.Gmax                  AS Gmax,
    ff.Gmean                 AS Gmean,
    ff.Gstd                  AS Gstd,
    ff.high_temp_area        AS high_temp_area,
    xf.file_path             AS file_path,
    xf.fps                   AS fps,
    s.sample_id              AS sample_id,
    s.B_mT                   AS B_mT,
    e.name                   AS experiment_name,
    e.laser_power_W          AS laser_power_W,
    e.scan_speed_mm_per_min  AS scan_speed_mm_per_min,
    e.powder_feed_rate_g_per_min AS powder_feed_rate_g_per_min,
    e.hatch_spacing_mm       AS hatch_spacing_mm
FROM frame_features ff
JOIN xtherm_files  xf ON ff.xtherm_file_id = xf.id
JOIN samples       s  ON xf.sample_id = s.id
JOIN experiments   e  ON s.experiment_id = e.id
{where_clause}
ORDER BY xf.id, ff.frame_index
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Export frame_features to a single CSV.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: data/features/all_features.csv).")
    parser.add_argument("--B-mT", dest="B_mT", type=float, default=None,
                        help="Optional filter: only export rows for this B (mT).")
    parser.add_argument("--sample-id", default=None,
                        help="Optional filter: only export rows for this sample_id.")
    parser.add_argument("--file-ids", default=None,
                        help="Optional filter: comma-separated xtherm_files.id list.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.format)
    log = logging.getLogger("export_features")

    where_parts: List[str] = []
    params: List = []
    if args.B_mT is not None:
        where_parts.append("s.B_mT = ?")
        params.append(float(args.B_mT))
    if args.sample_id is not None:
        where_parts.append("s.sample_id = ?")
        params.append(args.sample_id)
    if args.file_ids:
        ids = [int(x) for x in args.file_ids.split(",") if x.strip()]
        if ids:
            placeholders = ",".join("?" * len(ids))
            where_parts.append(f"ff.xtherm_file_id IN ({placeholders})")
            params.extend(ids)
    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    sql = SELECT_SQL.format(where_clause=where_clause)

    output_path = (
        resolve_under_root(args.output)
        if args.output
        else cfg.paths.data_features_abs() / "all_features.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = open_db(cfg.paths.database_abs())
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()

    log.info("Loaded %d rows from frame_features.", len(df))
    if df.empty:
        log.warning("No rows matched the filter; writing empty CSV with header only.")
    df.to_csv(output_path, index=False)
    log.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
