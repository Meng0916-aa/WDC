"""Initialize the local SQLite database for the project.

Usage:
    python scripts/init_database.py --config configs/default.yaml

Idempotent: re-running on an existing DB will not drop data; only creates
missing tables / indices and ensures the default experiment row exists.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 让脚本不依赖 `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import open_db, init_schema, ensure_default_experiment  # noqa: E402
from src.utils import load_config, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize SQLite database.")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="YAML config path (relative to project root or absolute).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.format)
    log = logging.getLogger("init_database")

    db_path = cfg.paths.database_abs()
    log.info("DB path: %s", db_path)
    conn = open_db(db_path)
    try:
        init_schema(conn)
        log.info("Schema OK.")

        exp_id = ensure_default_experiment(
            conn,
            name=cfg.experiment.name,
            powder_material=cfg.experiment.powder_material,
            substrate_material=cfg.experiment.substrate_material,
            laser_power_W=cfg.experiment.laser_power_W,
            scan_speed_mm_per_min=cfg.experiment.scan_speed_mm_per_min,
            powder_feed_rate_g_per_min=cfg.experiment.powder_feed_rate_g_per_min,
            hatch_spacing_mm=cfg.experiment.hatch_spacing_mm,
            notes=f"B_levels_mT={cfg.experiment.B_levels_mT}",
        )
        log.info("Default experiment id=%d (name=%s).", exp_id, cfg.experiment.name)

        n_files = conn.execute("SELECT COUNT(*) FROM xtherm_files").fetchone()[0]
        n_samples = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        log.info("Current state: experiments=%d samples=%d xtherm_files=%d",
                 conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
                 n_samples, n_files)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
