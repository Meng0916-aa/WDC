"""Batch-process registered .xtherm files filtered by status (or explicit ids).

Usage:
    python scripts/batch_process_database.py --config configs/default.yaml --status registered
    python scripts/batch_process_database.py --config configs/default.yaml --status probed
    python scripts/batch_process_database.py --config configs/default.yaml --file-ids 1,3,5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import open_db  # noqa: E402
from src.pipeline import batch_process  # noqa: E402
from src.utils import load_config, setup_logging  # noqa: E402


def _parse_ids(s: Optional[str]) -> Optional[List[int]]:
    if s is None or not s.strip():
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch process registered .xtherm files.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--status", default="registered",
                        help="xtherm_files.status filter (default: registered). "
                             "Use 'all' to process every row.")
    parser.add_argument("--file-ids", default=None,
                        help="Comma-separated list of xtherm_files.id; overrides --status.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.format)
    log = logging.getLogger("batch_process_database")

    ids = _parse_ids(args.file_ids)
    status = None if args.status == "all" else args.status

    conn = open_db(cfg.paths.database_abs())
    try:
        report = batch_process(
            conn, cfg,
            status=status, file_ids=ids,
            stop_on_error=args.stop_on_error,
        )
    finally:
        conn.close()

    log.info("Batch done: total=%d succeeded=%d failed=%d",
             report.total, report.succeeded, report.failed)
    return 0 if report.failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
