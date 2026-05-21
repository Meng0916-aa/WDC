"""Detect & remove duplicate frames; emit a deduplicated NPZ + a CSV report.

This is a stand-alone QA tool. It does NOT modify the original .xtherm
file, the xtherm_files.status field, or processing_results rows.

Two input modes (mutually exclusive):
  --file-id <id>   Read a registered .xtherm via the DB metadata.
  --npz <path>     Read temperature from an existing NPZ (key 'temperature').

Examples:
  python scripts/detect_duplicate_frames.py --config configs/default.yaml \\
      --file-id 1

  python scripts/detect_duplicate_frames.py --config configs/default.yaml \\
      --file-id 1 --mae-threshold 0.5 --max-abs-threshold 5.0

  python scripts/detect_duplicate_frames.py --config configs/default.yaml \\
      --npz data/processed/B000_sample01_temperature.npz
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import fetch_xtherm_file, open_db  # noqa: E402
from src.io.xtherm_reader import read_xtherm  # noqa: E402
from src.processing import detect_duplicate_frames, remove_duplicate_frames  # noqa: E402
from src.utils import load_config, resolve_under_root, setup_logging  # noqa: E402
from src.utils.config import AppConfig  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402


logger = logging.getLogger("detect_duplicate_frames")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _load_from_file_id(
    conn: sqlite3.Connection,
    file_id: int,
    cfg: AppConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load temperature cube + provenance metadata for a registered file."""
    row = fetch_xtherm_file(conn, file_id)
    abs_path = resolve_under_root(row["file_path"])
    logger.info("Reading file_id=%d path=%s", file_id, abs_path)

    temperature = read_xtherm(
        abs_path,
        width=row["width"],
        height=row["height"],
        dtype=row["dtype"],
        endian=row["endian"],
        header_offset=row["header_offset"],
        temperature_scale=row["temperature_scale"],
        max_frames=cfg.processing.max_frames,
    )

    # Pull experiment-level metadata via joins for richer NPZ provenance
    sample_row = conn.execute(
        """
        SELECT s.sample_id AS sample_id, s.B_mT AS B_mT,
               e.name AS run_id,
               e.laser_power_W AS laser_power_w,
               e.scan_speed_mm_per_min AS scan_speed_mm_min,
               e.powder_feed_rate_g_per_min AS powder_feed_g_min,
               e.powder_material AS powder,
               e.substrate_material AS substrate
        FROM samples s
        JOIN experiments e ON s.experiment_id = e.id
        WHERE s.id = ?
        """,
        (int(row["sample_id"]),),
    ).fetchone()
    if sample_row is None:
        raise LookupError(f"orphan xtherm_files.id={file_id}: sample_id={row['sample_id']} missing")

    meta: Dict[str, Any] = {
        "source_file_id": int(file_id),
        "source_file": str(_to_rel_under_root(abs_path)),
        "sample_id": str(sample_row["sample_id"]),
        "B_mT": float(sample_row["B_mT"]),
        "run_id": str(sample_row["run_id"]),
        "laser_power_w": float(sample_row["laser_power_w"]),
        "scan_speed_mm_min": float(sample_row["scan_speed_mm_min"]),
        "powder_feed_g_min": float(sample_row["powder_feed_g_min"]),
        "powder": str(sample_row["powder"]),
        "substrate": str(sample_row["substrate"]),
        "fps": float(row["fps"]) if row["fps"] is not None else float("nan"),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "dtype": str(row["dtype"]),
        "endian": str(row["endian"]),
        "header_offset": int(row["header_offset"]),
        "temperature_scale": float(row["temperature_scale"]),
    }
    return temperature, meta


def _load_from_npz(npz_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    logger.info("Loading NPZ: %s", npz_path)
    data = np.load(npz_path, allow_pickle=False)
    if "temperature" not in data.files:
        raise KeyError(f"NPZ {npz_path} missing required key 'temperature'")
    temperature = np.asarray(data["temperature"])
    if temperature.ndim != 3:
        raise ValueError(
            f"temperature in {npz_path} must be 3D, got shape={temperature.shape}"
        )
    meta: Dict[str, Any] = {"source_npz": str(_to_rel_under_root(npz_path))}
    # Carry through any common provenance keys that may exist in the input
    for k in (
        "fps", "width", "height", "sample_id", "B_mT", "run_id",
        "laser_power_w", "scan_speed_mm_min", "powder_feed_g_min",
        "powder", "substrate", "source_file",
    ):
        if k in data.files:
            meta[k] = data[k].item() if data[k].ndim == 0 else data[k]
    return temperature, meta


# --------------------------------------------------------------------------- #
# Path / naming helpers
# --------------------------------------------------------------------------- #
def _to_rel_under_root(p: Path) -> str:
    """Best-effort: return posix path relative to PROJECT_ROOT, else absolute posix."""
    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(p).resolve().as_posix()


def _default_outputs(
    cfg: AppConfig,
    meta: Dict[str, Any],
    file_id: Optional[int],
    input_npz: Optional[Path],
) -> Tuple[Path, Path]:
    """Return (output_npz, output_report) absolute paths."""
    if file_id is not None:
        sample_id = meta.get("sample_id", f"sample{file_id}")
        track_id = f"file{file_id}"
        stem = f"{sample_id}_{track_id}"
    elif input_npz is not None:
        stem = input_npz.stem
    else:
        stem = "unknown_source"

    out_npz = cfg.paths.data_processed_abs() / f"{stem}_temperature_dedup.npz"
    out_report = cfg.paths.data_features_abs() / f"{stem}_duplicate_frames.csv"
    return out_npz, out_report


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect & remove duplicate frames.")
    parser.add_argument("--config", default="configs/default.yaml")
    src_grp = parser.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--file-id", type=int, default=None,
                         help="Read a registered .xtherm by xtherm_files.id.")
    src_grp.add_argument("--npz", type=str, default=None,
                         help="Read temperature from an existing NPZ.")
    parser.add_argument("--mae-threshold", type=float, default=None,
                        help="Frames with MAE-to-prev <= this are flagged near-duplicate. "
                             "Unit matches the input array (degC or raw counts).")
    parser.add_argument("--max-abs-threshold", type=float, default=None,
                        help="Frames with max-abs-diff-to-prev <= this are flagged near-duplicate.")
    parser.add_argument("--remove-near-duplicates", action="store_true",
                        help="Also drop frames flagged as near-duplicate "
                             "(default: only exact duplicates are removed).")
    parser.add_argument("--output-npz", default=None,
                        help="Override output NPZ path "
                             "(default: data/processed/{sample}_{track}_temperature_dedup.npz).")
    parser.add_argument("--output-report", default=None,
                        help="Override output CSV path "
                             "(default: data/features/{sample}_{track}_duplicate_frames.csv).")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.format)

    # 1. Load temperature + provenance metadata
    input_npz_path: Optional[Path] = None
    if args.file_id is not None:
        conn = open_db(cfg.paths.database_abs())
        try:
            temperature, meta = _load_from_file_id(conn, args.file_id, cfg)
        finally:
            conn.close()
    else:
        input_npz_path = resolve_under_root(args.npz)
        temperature, meta = _load_from_npz(input_npz_path)

    n_orig = int(temperature.shape[0])
    logger.info(
        "Loaded temperature cube shape=%s dtype=%s",
        temperature.shape, temperature.dtype,
    )

    # 2. Detect
    report = detect_duplicate_frames(
        temperature,
        exact=True,
        mae_threshold=args.mae_threshold,
        max_abs_threshold=args.max_abs_threshold,
    )
    n_exact = int(report["is_exact_duplicate"].sum())
    n_near = int(report["is_near_duplicate"].sum())
    logger.info("Detection: exact=%d near=%d (of %d frames)", n_exact, n_near, n_orig)

    # 3. Remove
    temperature_dedup, keep_indices, removed_indices = remove_duplicate_frames(
        temperature,
        report,
        remove_near_duplicates=args.remove_near_duplicates,
    )
    logger.info(
        "After removal: kept=%d removed=%d (remove_near_duplicates=%s)",
        len(keep_indices), len(removed_indices), args.remove_near_duplicates,
    )

    # 4. Resolve output paths
    if args.output_npz:
        out_npz = resolve_under_root(args.output_npz)
    else:
        out_npz, _ = _default_outputs(cfg, meta, args.file_id, input_npz_path)
    if args.output_report:
        out_report = resolve_under_root(args.output_report)
    else:
        _, out_report = _default_outputs(cfg, meta, args.file_id, input_npz_path)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    # 5. Write CSV report
    report.to_csv(out_report, index=False)
    logger.info("Wrote duplicate report: %s", out_report)

    # 6. Write dedup NPZ (with provenance + the report path)
    save_kwargs: Dict[str, Any] = {
        "temperature": temperature_dedup,
        "original_frame_indices": keep_indices,
        "removed_frame_indices": removed_indices,
        "duplicate_report_path": _to_rel_under_root(out_report),
        "fps": float(meta.get("fps", float("nan"))),
        "width": int(meta.get("width", temperature_dedup.shape[2])),
        "height": int(meta.get("height", temperature_dedup.shape[1])),
        "n_frames_original": int(n_orig),
        "n_frames_kept": int(len(keep_indices)),
        "n_frames_removed": int(len(removed_indices)),
        "n_exact_duplicates": int(n_exact),
        "n_near_duplicates": int(n_near),
        "remove_near_duplicates": bool(args.remove_near_duplicates),
        "mae_threshold": (
            float(args.mae_threshold) if args.mae_threshold is not None else float("nan")
        ),
        "max_abs_threshold": (
            float(args.max_abs_threshold) if args.max_abs_threshold is not None else float("nan")
        ),
    }
    # Mutually-exclusive provenance pointer
    if args.file_id is not None:
        save_kwargs["source_file_id"] = int(args.file_id)
    if input_npz_path is not None:
        save_kwargs["source_npz"] = _to_rel_under_root(input_npz_path)
    # Carry rich metadata when available
    for k in (
        "source_file", "sample_id", "B_mT", "run_id",
        "laser_power_w", "scan_speed_mm_min", "powder_feed_g_min",
        "powder", "substrate", "temperature_scale",
        "dtype", "endian", "header_offset",
    ):
        if k in meta and meta[k] is not None:
            save_kwargs[k] = meta[k]

    np.savez(out_npz, **save_kwargs)
    logger.info("Wrote dedup NPZ: %s", out_npz)

    # 7. Brief summary to stdout (so callers without a logger setup still see it)
    print(
        f"Duplicate-frame detection done.\n"
        f"  input  frames: {n_orig}\n"
        f"  exact dups   : {n_exact}\n"
        f"  near dups    : {n_near}\n"
        f"  kept         : {len(keep_indices)}\n"
        f"  removed      : {len(removed_indices)}\n"
        f"  report CSV   : {out_report}\n"
        f"  dedup NPZ    : {out_npz}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
