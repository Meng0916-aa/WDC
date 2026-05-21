"""Preflight: verify the project is ready to receive real .xtherm data.

Usage:
    python scripts/check_ready_for_real_data.py --config configs/default.yaml

Exit codes (so it composes well into a Makefile / CI gate):
    0 : ready for everything, including gradient processing (dx/dy set)
    1 : ready for registration and probing only (gradient blocked on missing dx/dy)
    2 : not ready for registration (DB / directories / camera config incomplete)
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, setup_logging  # noqa: E402
from src.utils.config import AppConfig  # noqa: E402


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class CheckItem:
    name: str
    status: str            # 'OK' | 'WARN' | 'FAIL' | 'INFO'
    detail: str
    fix: str = ""


@dataclass
class ReadinessReport:
    config_path: Path
    checks: List[CheckItem] = field(default_factory=list)
    ready_for_registration: bool = False
    ready_for_probing: bool = False
    ready_for_gradient_processing: bool = False

    def print(self, stream: Optional[TextIO] = None) -> None:
        stream = stream or sys.stdout
        title = f"Project readiness preflight ({self.config_path})"
        print(title, file=stream)
        print("=" * len(title), file=stream)
        max_name = max((len(c.name) for c in self.checks), default=0)
        for c in self.checks:
            tag = f"[{c.status:^4}]"
            print(f"{tag} {c.name:<{max_name}}  : {c.detail}", file=stream)
            if c.status in ("WARN", "FAIL") and c.fix:
                print(f"        fix: {c.fix}", file=stream)
        print(file=stream)
        print("Summary", file=stream)
        print("-------", file=stream)
        print(f"ready_for_registration        : {'YES' if self.ready_for_registration else 'NO'}",
              file=stream)
        print(f"ready_for_probing             : {'YES' if self.ready_for_probing else 'NO'}",
              file=stream)
        print(f"ready_for_gradient_processing : {'YES' if self.ready_for_gradient_processing else 'NO'}",
              file=stream)


def _add(report: ReadinessReport, **kwargs) -> CheckItem:
    item = CheckItem(**kwargs)
    report.checks.append(item)
    return item


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def _check_database(report: ReadinessReport, cfg: AppConfig) -> tuple[bool, bool]:
    """Returns (db_file_ok, schema_ok)."""
    db_path = cfg.paths.database_abs()
    if not db_path.exists():
        _add(report, name="database", status="FAIL",
             detail=f"missing {db_path}",
             fix="python scripts/init_database.py --config configs/default.yaml")
        return False, False

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='xtherm_files'"
            ).fetchone()
            has_schema = row is not None
            n_files = (
                conn.execute("SELECT COUNT(*) FROM xtherm_files").fetchone()[0]
                if has_schema
                else 0
            )
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        _add(report, name="database", status="FAIL",
             detail=f"{db_path} present but unreadable ({e})",
             fix="delete the file and re-run scripts/init_database.py")
        return False, False

    if not has_schema:
        _add(report, name="database", status="FAIL",
             detail=f"{db_path} exists but xtherm_files table is missing",
             fix="re-run scripts/init_database.py --config configs/default.yaml")
        return True, False

    _add(report, name="database", status="OK", detail=str(db_path))
    _add(report, name="xtherm_files registered", status="INFO", detail=str(n_files))
    return True, True


def _check_directories(report: ReadinessReport, cfg: AppConfig) -> bool:
    dirs_ok = True
    for label, abs_path in (
        ("data/raw", cfg.paths.data_raw_abs()),
        ("data/processed", cfg.paths.data_processed_abs()),
        ("data/features", cfg.paths.data_features_abs()),
        ("results/figures", cfg.paths.results_figures_abs()),
    ):
        if abs_path.exists() and abs_path.is_dir():
            _add(report, name=label, status="OK", detail=str(abs_path))
        else:
            dirs_ok = False
            _add(report, name=label, status="FAIL",
                 detail=f"missing {abs_path}",
                 fix=f"mkdir -p {abs_path}")
    return dirs_ok


def _check_camera_basics(report: ReadinessReport, cfg: AppConfig) -> bool:
    cam = cfg.camera
    cam_ok = True

    if cam.width > 0:
        _add(report, name="camera.width", status="OK", detail=str(cam.width))
    else:
        cam_ok = False
        _add(report, name="camera.width", status="FAIL",
             detail=f"{cam.width} (must be > 0)",
             fix="set camera.width in configs/default.yaml")

    if cam.height > 0:
        _add(report, name="camera.height", status="OK", detail=str(cam.height))
    else:
        cam_ok = False
        _add(report, name="camera.height", status="FAIL",
             detail=f"{cam.height} (must be > 0)",
             fix="set camera.height in configs/default.yaml")

    if cam.fps and cam.fps > 0:
        _add(report, name="camera.fps", status="OK", detail=str(cam.fps))
    else:
        cam_ok = False
        _add(report, name="camera.fps", status="FAIL",
             detail=f"{cam.fps} (must be > 0)",
             fix="set camera.fps in configs/default.yaml")

    if cam.temperature_scale and cam.temperature_scale != 0:
        _add(report, name="camera.temperature_scale", status="OK",
             detail=str(cam.temperature_scale))
    else:
        cam_ok = False
        _add(report, name="camera.temperature_scale", status="FAIL",
             detail=f"{cam.temperature_scale} (must be non-zero)",
             fix="set camera.temperature_scale in configs/default.yaml")

    return cam_ok


def _check_calibration(report: ReadinessReport, cfg: AppConfig) -> bool:
    """dx/dy together determine whether °C/mm gradients can be computed."""
    cam = cfg.camera
    dx_dy_ok = True

    def _check_one(field_name: str, value: Optional[float]) -> bool:
        if value is None:
            _add(report, name=f"camera.{field_name}", status="WARN",
                 detail="null (camera calibration not yet measured)",
                 fix=f"measure with a target then set camera.{field_name} in configs/default.yaml")
            return False
        if not (value > 0):
            _add(report, name=f"camera.{field_name}", status="FAIL",
                 detail=f"{value} (must be > 0)",
                 fix=f"re-measure and set camera.{field_name} in configs/default.yaml")
            return False
        _add(report, name=f"camera.{field_name}", status="OK", detail=str(value))
        return True

    dx_dy_ok &= _check_one("dx_mm_per_pixel", cam.dx_mm_per_pixel)
    dx_dy_ok &= _check_one("dy_mm_per_pixel", cam.dy_mm_per_pixel)

    if not dx_dy_ok:
        _add(report, name="gradient feasibility", status="WARN",
             detail="without dx/dy, the project can register and probe files only; "
                    "compute_gradients refuses to run (degC/mm has no defined units)",
             fix="set BOTH camera.dx_mm_per_pixel and camera.dy_mm_per_pixel "
                 "(from camera calibration) in configs/default.yaml")
    return dx_dy_ok


def _check_header_offset(report: ReadinessReport, cfg: AppConfig) -> None:
    if cfg.camera.header_offset == 0:
        _add(report, name="camera.header_offset", status="INFO",
             detail="0 (default; once a real .xtherm is registered, run "
                    "scripts/probe_registered_file.py --file-id <id> to verify, "
                    "then re-run with --apply to write the best candidate back)")
    else:
        _add(report, name="camera.header_offset", status="OK",
             detail=str(cfg.camera.header_offset))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def check_readiness(cfg: AppConfig) -> ReadinessReport:
    """Pure function: assemble the readiness report from cfg + filesystem state."""
    report = ReadinessReport(config_path=cfg.source_path)
    db_ok, schema_ok = _check_database(report, cfg)
    dirs_ok = _check_directories(report, cfg)
    cam_ok = _check_camera_basics(report, cfg)
    dx_dy_ok = _check_calibration(report, cfg)
    _check_header_offset(report, cfg)

    report.ready_for_registration = bool(db_ok and schema_ok and dirs_ok and cam_ok)
    # probing has the same prerequisites as registration; it operates on any
    # already-registered file_id and never needs dx/dy.
    report.ready_for_probing = report.ready_for_registration
    report.ready_for_gradient_processing = bool(report.ready_for_registration and dx_dy_ok)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight readiness check for real .xtherm data.")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="YAML config path (relative to project root or absolute).")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] cannot load config '{args.config}': {e}", file=sys.stderr)
        return 2

    setup_logging(cfg.logging.level, cfg.logging.format)
    logging.getLogger("check_ready_for_real_data").debug(
        "config loaded from %s", cfg.source_path
    )

    report = check_readiness(cfg)
    report.print()

    if report.ready_for_gradient_processing:
        return 0
    if report.ready_for_registration:
        print(
            "\nNote: registration and probing are ready; gradient processing is "
            "blocked until camera.dx_mm_per_pixel and camera.dy_mm_per_pixel "
            "are set in configs/default.yaml.",
            file=sys.stderr,
        )
        return 1
    print(
        "\nNote: the project is not yet ready for registration. Address the "
        "FAIL items above before placing real .xtherm files in data/raw/.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
