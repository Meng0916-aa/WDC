"""Probe a registered .xtherm file to verify its header_offset and frame count.

Reads the file path / shape from xtherm_files.id = <file_id>, tries each
candidate header_offset from configs.format_probe.header_offset_candidates,
prints a structured report and (optionally) updates the DB row in place.

Usage:
    python scripts/probe_registered_file.py --config configs/default.yaml --file-id 1
    python scripts/probe_registered_file.py --config configs/default.yaml --file-id 1 --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import (  # noqa: E402
    fetch_xtherm_file,
    open_db,
    update_file_status,
)
from src.io import probe_xtherm  # noqa: E402
from src.utils import load_config, resolve_under_root, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe header_offset / n_frames for a registered file.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--file-id", required=True, type=int)
    parser.add_argument("--apply", action="store_true",
                        help="If a single best candidate is found, update xtherm_files in place "
                             "and set status to 'probed'.")
    parser.add_argument("--json", action="store_true",
                        help="Print the full probe report as JSON (machine readable).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.format)
    log = logging.getLogger("probe_registered_file")

    conn = open_db(cfg.paths.database_abs())
    try:
        row = fetch_xtherm_file(conn, args.file_id)
        abs_path = resolve_under_root(row["file_path"])
        log.info("Probing file_id=%d path=%s size=%d bytes",
                 args.file_id, abs_path, row["file_size_bytes"])

        report = probe_xtherm(
            abs_path,
            width=row["width"],
            height=row["height"],
            dtype=row["dtype"],
            endian=row["endian"],
            temperature_scale=row["temperature_scale"],
            header_offset_candidates=cfg.format_probe.header_offset_candidates,
            sample_frames_spec=cfg.format_probe.sample_frames,
        )

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"\nFile: {report.file_path}")
            print(f"  size = {report.file_size} bytes; W*H = {report.width}*{report.height}; "
                  f"dtype={report.dtype} endian={report.endian} scale={report.temperature_scale}")
            print(f"  {'offset':>8}  {'n_frames':>9}  {'Tmin':>8}  {'Tmax':>8}  {'Tmean':>8}  note")
            for c in report.candidates:
                Tmin = "-" if c.sample_T_min_C is None else f"{c.sample_T_min_C:8.2f}"
                Tmax = "-" if c.sample_T_max_C is None else f"{c.sample_T_max_C:8.2f}"
                Tmean = "-" if c.sample_T_mean_C is None else f"{c.sample_T_mean_C:8.2f}"
                print(f"  {c.header_offset:8d}  {c.n_frames:9d}  {Tmin}  {Tmax}  {Tmean}  {c.note}")
            if report.best is not None:
                print(f"  Best  : offset={report.best.header_offset} n_frames={report.best.n_frames}")
            else:
                print("  Best  : <none>")

        if args.apply and report.best is not None:
            update_file_status(
                conn,
                args.file_id,
                "probed",
                estimated_frames=report.best.n_frames,
                header_offset=report.best.header_offset,
                notes=f"probed: offset={report.best.header_offset}, n_frames={report.best.n_frames}",
            )
            log.info("Applied best candidate to xtherm_files.id=%d", args.file_id)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
