"""Merge a folder of single-frame .xtherm files into one [T,H,W] NPZ.

Use this when Xiris / WeldStudio exported one .xtherm per frame
(e.g. ``Image_00000.xtherm``, ``Image_00001.xtherm``, ...). The output
NPZ is ready to feed into ``scripts/detect_duplicate_frames.py --npz``
and downstream processing.

Example
-------
python scripts/merge_xtherm_folder.py --config configs/default.yaml \\
    --input-dir data/raw/B000/sample01 \\
    --sample-id B000_sample01 --B-mT 0 \\
    --output data/processed/B000_sample01_temperature_sequence.npz

The original .xtherm files are read-only; nothing is modified in place.
The database is NOT touched.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.merge_xtherm_folder import merge_xtherm_folder  # noqa: E402
from src.utils import load_config, resolve_under_root, setup_logging  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402


logger = logging.getLogger("merge_xtherm_folder")


def _to_relpath_under_root(p: Path) -> str:
    """Best-effort: posix path relative to PROJECT_ROOT, else absolute posix."""
    p = Path(p).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a folder of single-frame .xtherm files into one NPZ."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing single-frame .xtherm files.")
    parser.add_argument("--sample-id", required=True,
                        help="Logical sample identifier (used in default output filename).")
    parser.add_argument("--B-mT", dest="B_mT", required=True, type=float,
                        help="Magnetic flux density for this sample (mT).")
    parser.add_argument("--output", default=None,
                        help="Output NPZ path. Default: "
                             "data/processed/{sample_id}_temperature_sequence.npz")
    parser.add_argument("--pattern", default="*.xtherm",
                        help="Glob pattern for input files (default: *.xtherm).")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively scan subdirectories of --input-dir.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.format)

    input_dir = resolve_under_root(args.input_dir)
    logger.info(
        "Merging %s/%s (recursive=%s)  W=%d H=%d header_offset=%d scale=%g",
        input_dir, args.pattern, args.recursive,
        cfg.camera.width, cfg.camera.height,
        cfg.camera.header_offset, cfg.camera.temperature_scale,
    )

    result = merge_xtherm_folder(
        input_dir,
        width=cfg.camera.width,
        height=cfg.camera.height,
        dtype=cfg.camera.dtype,
        endian=cfg.camera.endian,
        header_offset=cfg.camera.header_offset,
        temperature_scale=cfg.camera.temperature_scale,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    if args.output:
        out_path = resolve_under_root(args.output)
    else:
        out_path = cfg.paths.data_processed_abs() / f"{args.sample_id}_temperature_sequence.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Store source paths relative to project root when possible (portable)
    rel_sources = [_to_relpath_under_root(Path(s)) for s in result.source_files]

    np.savez(
        out_path,
        # Required by the spec
        temperature=result.temperature,
        source_files=np.array(rel_sources, dtype=object),
        fps=float(cfg.camera.fps),
        width=int(cfg.camera.width),
        height=int(cfg.camera.height),
        header_offset=int(cfg.camera.header_offset),
        temperature_scale=float(cfg.camera.temperature_scale),
        sample_id=str(args.sample_id),
        B_mT=float(args.B_mT),
        # Extra provenance per project conventions (xtherm-reader skill).
        # These are not required, but make the NPZ self-describing for
        # downstream consumers (dedup / processing / export) without
        # forcing them to re-read configs/default.yaml.
        dtype=str(cfg.camera.dtype),
        endian=str(cfg.camera.endian),
        run_id=str(cfg.experiment.name),
        laser_power_w=float(cfg.experiment.laser_power_W),
        scan_speed_mm_min=float(cfg.experiment.scan_speed_mm_per_min),
        powder_feed_g_min=float(cfg.experiment.powder_feed_rate_g_per_min),
        powder=str(cfg.experiment.powder_material),
        substrate=str(cfg.experiment.substrate_material),
        n_frames=int(result.n_files),
        input_dir=_to_relpath_under_root(input_dir),
    )

    logger.info("Merged %d frames -> %s", result.n_files, out_path)
    print(
        f"Merge done.\n"
        f"  input dir  : {input_dir}\n"
        f"  pattern    : {args.pattern}\n"
        f"  n_frames   : {result.n_files}\n"
        f"  shape      : {result.temperature.shape}\n"
        f"  dtype      : {result.temperature.dtype}\n"
        f"  output NPZ : {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
