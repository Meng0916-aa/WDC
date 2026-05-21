"""Compute Gx / Gy / G and per-frame features from a temperature NPZ.

Companion to ``scripts/merge_xtherm_folder.py`` and
``scripts/detect_duplicate_frames.py``. This script is the NPZ-driven
sibling of ``src.pipeline.process_run`` (which is DB-driven):

- Inputs the same kind of NPZ produced upstream (``temperature`` key
  shape ``[T, H, W]``, degC).
- Pulls ``dx / dy / gaussian_sigma / high_temp_threshold`` from
  ``configs/default.yaml``.
- Reuses the existing core functions:
  ``src.processing.compute_gradients`` and
  ``src.processing.extract_frame_features``.

Side-effect policy
------------------
- Reads from disk only.
- Writes one CSV (the per-frame feature table).
- Does NOT modify the input NPZ, the database, or any ``.xtherm`` file.
- The full Gx / Gy / G cubes are NEVER saved (too large). Only frame-
  level scalars go to CSV. An optional ``--save-gradient-stats-npz``
  flag emits a small NPZ with the same scalars in numpy-array form, for
  convenience of plotting scripts.

time_s priority
---------------
1. ``original_frame_indices`` from the NPZ (e.g. produced by dedup) divided by fps;
2. ``arange(T) / fps`` if no ``original_frame_indices``;
3. NaN if no fps is available either.

The fps itself prefers the NPZ field over ``config.camera.fps``, so a
merged / dedup NPZ that was recorded at non-standard fps will still get
the right time axis.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.processing import compute_gradients, extract_frame_features  # noqa: E402
from src.utils import load_config, resolve_under_root, setup_logging  # noqa: E402
from src.utils.config import AppConfig  # noqa: E402


CSV_COLUMNS = (
    "frame", "time_s",
    "Tmax", "Tmean", "Tstd",
    "Gmax", "Gmean", "Gstd",
    "high_temp_area",
)


logger = logging.getLogger("extract_features_from_npz")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _scalar(arr) -> float:
    """Extract a finite scalar from a possibly 0-d or 1-element numpy array."""
    a = np.asarray(arr)
    if a.size == 0:
        return float("nan")
    return float(a.reshape(-1)[0])


def _resolve_fps(data, cfg: AppConfig) -> Optional[float]:
    """fps priority: NPZ ``fps`` field > ``config.camera.fps``. Returns None if neither is usable."""
    if "fps" in data.files:
        v = _scalar(data["fps"])
        if np.isfinite(v) and v > 0:
            return v
    fps = cfg.camera.fps
    if fps and fps > 0:
        return float(fps)
    return None


def _resolve_time_s(data, n_frames: int, fps: Optional[float]) -> np.ndarray:
    """time_s array per the spec's priority list."""
    if "original_frame_indices" in data.files:
        ofi = np.asarray(data["original_frame_indices"]).astype(np.float64).reshape(-1)
        if ofi.size != n_frames:
            raise ValueError(
                f"original_frame_indices length {ofi.size} does not match "
                f"temperature.shape[0] = {n_frames}"
            )
        if fps is not None and fps > 0:
            return ofi / fps
        logger.warning(
            "NPZ has original_frame_indices but no usable fps; time_s set to NaN."
        )
        return np.full(n_frames, np.nan, dtype=np.float64)
    if fps is not None and fps > 0:
        return np.arange(n_frames, dtype=np.float64) / fps
    logger.warning("No fps in NPZ or config; time_s set to NaN.")
    return np.full(n_frames, np.nan, dtype=np.float64)


def _default_output_csv(input_npz: Path, cfg: AppConfig) -> Path:
    """Strip well-known suffixes from input stem, append _features.csv."""
    stem = input_npz.stem
    for suffix in ("_temperature_dedup", "_dedup", "_temperature_sequence"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return cfg.paths.data_features_abs() / f"{stem}_features.csv"


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def extract_features_from_npz(
    input_npz: Path,
    cfg: AppConfig,
    output_csv: Path,
    save_gradient_stats_npz: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the full per-frame extraction on a single NPZ. Returns the resulting DataFrame.

    Raises
    ------
    FileNotFoundError
        If the input NPZ does not exist.
    KeyError
        If the NPZ has no ``temperature`` key.
    ValueError
        If ``temperature`` is not 3D, dx/dy are not set in cfg, or
        ``original_frame_indices`` length does not match the frame count.
    """
    input_npz = Path(input_npz)
    if not input_npz.exists():
        raise FileNotFoundError(f"input NPZ not found: {input_npz}")

    # Fail fast on missing calibration so a 410-frame file doesn't go halfway.
    dx = cfg.camera.dx_mm_per_pixel
    dy = cfg.camera.dy_mm_per_pixel
    if dx is None or dy is None:
        raise ValueError(
            "dx_mm_per_pixel and dy_mm_per_pixel must be set in configs/default.yaml "
            f"(got dx={dx!r}, dy={dy!r}). Run scripts/check_ready_for_real_data.py "
            "to confirm before re-running."
        )

    data = np.load(input_npz, allow_pickle=False)
    if "temperature" not in data.files:
        raise KeyError(f"NPZ {input_npz} missing required key 'temperature'")
    T_cube = np.asarray(data["temperature"])
    if T_cube.ndim != 3:
        raise ValueError(
            f"temperature must be 3D (T, H, W), got shape={T_cube.shape}"
        )

    n_frames = int(T_cube.shape[0])
    fps = _resolve_fps(data, cfg)
    time_s = _resolve_time_s(data, n_frames, fps)

    logger.info(
        "NPZ=%s frames=%d (H=%d,W=%d) fps=%s dx=%g dy=%g sigma=%g high_thr=%g",
        input_npz, n_frames, T_cube.shape[1], T_cube.shape[2],
        fps, dx, dy,
        cfg.processing.gaussian_sigma_px, cfg.processing.high_temp_threshold_C,
    )

    rows: List[dict] = []
    Gmax_arr = np.empty(n_frames, dtype=np.float64)
    Gmean_arr = np.empty(n_frames, dtype=np.float64)
    Gstd_arr = np.empty(n_frames, dtype=np.float64)

    for i in range(n_frames):
        _, _, G = compute_gradients(
            T_cube[i],
            dx_mm_per_pixel=dx,
            dy_mm_per_pixel=dy,
            gaussian_sigma_px=cfg.processing.gaussian_sigma_px,
        )
        feat = extract_frame_features(
            T_cube[i], G,
            frame_index=i,
            time_s=float(time_s[i]) if np.isfinite(time_s[i]) else None,
            high_temp_threshold_C=cfg.processing.high_temp_threshold_C,
        )
        # Rename to match the user-spec'd CSV column name
        feat["frame"] = feat.pop("frame_index")
        rows.append(feat)
        Gmax_arr[i] = feat["Gmax"]
        Gmean_arr[i] = feat["Gmean"]
        Gstd_arr[i] = feat["Gstd"]
        if cfg.processing.log_every_frames and (i + 1) % cfg.processing.log_every_frames == 0:
            logger.info("  frame %d / %d", i + 1, n_frames)

    df = pd.DataFrame(rows, columns=list(CSV_COLUMNS))

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Wrote %d feature rows to %s", len(df), output_csv)

    if save_gradient_stats_npz is not None:
        save_gradient_stats_npz = Path(save_gradient_stats_npz)
        save_gradient_stats_npz.parent.mkdir(parents=True, exist_ok=True)
        # NOTE: deliberately NOT writing the full Gx / Gy / G cubes — would be huge.
        np.savez(
            save_gradient_stats_npz,
            frame=np.arange(n_frames, dtype=np.int64),
            time_s=time_s,
            Gmax=Gmax_arr,
            Gmean=Gmean_arr,
            Gstd=Gstd_arr,
        )
        logger.info("Wrote gradient stats NPZ: %s", save_gradient_stats_npz)

    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Gx/Gy/G and per-frame features from a temperature NPZ."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True,
                        help="Input NPZ (must contain key 'temperature' of shape [T,H,W]).")
    parser.add_argument("--output", default=None,
                        help="Output features CSV. Default: "
                             "data/features/{input_stem_without_known_suffix}_features.csv "
                             "(strips _temperature_dedup / _temperature_sequence / _dedup).")
    parser.add_argument("--save-gradient-stats-npz", default=None,
                        help="Optional path: also write a small NPZ with frame-level "
                             "Gmax/Gmean/Gstd as arrays. The full 3D Gx/Gy/G cubes are "
                             "never written (file size).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.format)

    input_npz = resolve_under_root(args.input)
    output_csv = resolve_under_root(args.output) if args.output else _default_output_csv(input_npz, cfg)
    save_stats = (
        resolve_under_root(args.save_gradient_stats_npz)
        if args.save_gradient_stats_npz else None
    )

    df = extract_features_from_npz(input_npz, cfg, output_csv, save_stats)

    Tmax_range = (float(df["Tmax"].min()), float(df["Tmax"].max()))
    Gmax_range = (float(df["Gmax"].min()), float(df["Gmax"].max()))
    print(
        f"Feature extraction done.\n"
        f"  input NPZ : {input_npz}\n"
        f"  output CSV: {output_csv}\n"
        f"  rows      : {len(df)}\n"
        f"  Tmax range: [{Tmax_range[0]:.2f}, {Tmax_range[1]:.2f}] degC\n"
        f"  Gmax range: [{Gmax_range[0]:.4f}, {Gmax_range[1]:.4f}] degC/mm"
    )
    if save_stats is not None:
        print(f"  stats NPZ : {save_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
