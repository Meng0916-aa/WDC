"""Register real .xtherm files into the SQLite database.

Stores only the file path and metadata; the raw binary content is NEVER
copied into the database.

Usage examples:
    # one file
    python scripts/register_xtherm_files.py --config configs/default.yaml \
        --input data/raw/B000/run01.xtherm --sample-id B000_sample01 --B-mT 0

    # recursive scan of a directory
    python scripts/register_xtherm_files.py --config configs/default.yaml \
        --input data/raw/B080/sample02 --sample-id B080_sample02 --B-mT 80 --recursive
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import (  # noqa: E402
    ensure_default_experiment,
    ensure_sample,
    init_schema,
    open_db,
    register_xtherm_file,
)
from src.utils import load_config, resolve_under_root, setup_logging  # noqa: E402


def _gather_files(input_path: Path, recursive: bool, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"--input not found: {input_path}")
    if recursive:
        files = sorted(input_path.rglob(pattern))
    else:
        files = sorted(input_path.glob(pattern))
    return [p for p in files if p.is_file()]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register .xtherm files into the DB.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True,
                        help="A .xtherm file OR a directory (use --recursive to walk).")
    parser.add_argument("--sample-id", required=True, help="Logical sample identifier.")
    parser.add_argument("--B-mT", dest="B_mT", required=True, type=float,
                        help="Magnetic flux density for this sample, in mT.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--pattern", default="*.xtherm",
                        help="Glob pattern when --input is a directory (default: *.xtherm).")
    parser.add_argument("--sha256", action="store_true",
                        help="Compute and store SHA-256 of each file (slower on large files).")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.format)
    log = logging.getLogger("register_xtherm_files")

    input_path = resolve_under_root(args.input)
    files = _gather_files(input_path, args.recursive, args.pattern)
    if not files:
        log.warning("No .xtherm files found under %s (pattern=%s, recursive=%s)",
                    input_path, args.pattern, args.recursive)
        return 0

    db_path = cfg.paths.database_abs()
    conn = open_db(db_path)
    try:
        init_schema(conn)
        exp_id = ensure_default_experiment(
            conn,
            name=cfg.experiment.name,
            powder_material=cfg.experiment.powder_material,
            substrate_material=cfg.experiment.substrate_material,
            laser_power_W=cfg.experiment.laser_power_W,
            scan_speed_mm_per_min=cfg.experiment.scan_speed_mm_per_min,
            powder_feed_rate_g_per_min=cfg.experiment.powder_feed_rate_g_per_min,
            hatch_spacing_mm=cfg.experiment.hatch_spacing_mm,
        )
        sample_pk = ensure_sample(
            conn,
            sample_id=args.sample_id,
            experiment_id=exp_id,
            B_mT=args.B_mT,
            notes=args.notes,
        )
        log.info("experiment_id=%d sample_pk=%d sample_id=%s B=%.1f mT",
                 exp_id, sample_pk, args.sample_id, args.B_mT)

        n_inserted = 0
        for f in files:
            try:
                file_id = register_xtherm_file(
                    conn,
                    sample_pk=sample_pk,
                    file_path=f,
                    width=cfg.camera.width,
                    height=cfg.camera.height,
                    dtype=cfg.camera.dtype,
                    endian=cfg.camera.endian,
                    header_offset=cfg.camera.header_offset,
                    temperature_scale=cfg.camera.temperature_scale,
                    fps=cfg.camera.fps,
                    dx_mm_per_pixel=cfg.camera.dx_mm_per_pixel,
                    dy_mm_per_pixel=cfg.camera.dy_mm_per_pixel,
                    compute_sha256=args.sha256,
                    notes=args.notes,
                )
                log.info("  registered id=%d path=%s", file_id, f)
                n_inserted += 1
            except Exception as e:  # noqa: BLE001
                log.error("Failed to register %s: %s", f, e)

        log.info("Registered %d / %d files.", n_inserted, len(files))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
