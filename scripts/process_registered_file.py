"""Process a single registered .xtherm file by its file_id.

Usage:
    python scripts/process_registered_file.py --config configs/default.yaml --file-id 1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import open_db  # noqa: E402
from src.pipeline import process_registered_file  # noqa: E402
from src.utils import load_config, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Process a single registered .xtherm file.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--file-id", required=True, type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.format)
    log = logging.getLogger("process_registered_file")

    conn = open_db(cfg.paths.database_abs())
    try:
        outcome = process_registered_file(conn, args.file_id, cfg)
    finally:
        conn.close()

    if outcome.ok:
        log.info("OK file_id=%d frames=%d csv=%s",
                 outcome.file_id, outcome.n_frames, outcome.feature_csv)
        return 0
    log.error("FAIL file_id=%d error=%s", outcome.file_id, outcome.error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
