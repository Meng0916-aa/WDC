"""Tests for scripts/check_ready_for_real_data.py.

This test file does NOT modify any existing passing test. It creates a fake
project tree under pytest's tmp_path and points PROJECT_ROOT at it via
monkeypatch, so the real D:\\GEJ-WDC layout is never touched.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_ready_for_real_data as cr  # noqa: E402

from src.db import init_schema, open_db  # noqa: E402
from src.utils import load_config  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures (local to this file; does not affect existing conftest.py)
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_project(tmp_path: Path, monkeypatch) -> Path:
    """Build a minimal project tree under tmp_path; patch PROJECT_ROOT to it.

    Always includes:
      configs/default.yaml (copied verbatim from the real repo)
      database/ data/raw/ data/processed/ data/features/ results/figures/  (empty)
    Does NOT create the database file itself; tests decide whether to call _init_db.
    """
    (tmp_path / "configs").mkdir()
    shutil.copyfile(
        REPO_ROOT / "configs" / "default.yaml",
        tmp_path / "configs" / "default.yaml",
    )
    for d in ("database", "data/raw", "data/processed", "data/features", "results/figures"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    import src.utils.paths as paths_mod
    monkeypatch.setattr(paths_mod, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _init_db(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "database" / "thermal_cladding.db")
    init_schema(conn)
    conn.close()


def _find(checks, name: str) -> cr.CheckItem:
    matches = [c for c in checks if c.name == name]
    assert matches, f"no CheckItem named {name!r}; have: {[c.name for c in checks]}"
    return matches[0]


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_not_ready_when_db_and_dirs_missing(fake_project, tmp_path):
    # Wipe the data dirs the fixture created -> not ready for registration
    for d in ("data/raw", "data/processed", "data/features", "results/figures"):
        shutil.rmtree(tmp_path / d)

    cfg = load_config("configs/default.yaml")
    report = cr.check_readiness(cfg)

    assert not report.ready_for_registration
    assert not report.ready_for_probing
    assert not report.ready_for_gradient_processing
    assert _find(report.checks, "database").status == "FAIL"
    for d in ("data/raw", "data/processed", "data/features", "results/figures"):
        assert _find(report.checks, d).status == "FAIL"


def test_db_present_dirs_present_but_no_calibration(fake_project, tmp_path):
    _init_db(tmp_path)
    cfg = load_config("configs/default.yaml")  # ships with dx/dy = null

    report = cr.check_readiness(cfg)

    assert report.ready_for_registration
    assert report.ready_for_probing
    assert not report.ready_for_gradient_processing

    dx = _find(report.checks, "camera.dx_mm_per_pixel")
    dy = _find(report.checks, "camera.dy_mm_per_pixel")
    assert dx.status == "WARN" and "calibration" in dx.detail.lower()
    assert dy.status == "WARN" and "calibration" in dy.detail.lower()

    # gradient feasibility warning explicitly mentions °C/mm + dx/dy
    grad = _find(report.checks, "gradient feasibility")
    assert grad.status == "WARN"
    assert "compute_gradients" in grad.detail or "°C/mm" in grad.detail


def test_fully_ready_when_dx_dy_set(fake_project, tmp_path):
    _init_db(tmp_path)
    cfg = load_config("configs/default.yaml")
    cfg.camera.dx_mm_per_pixel = 0.05
    cfg.camera.dy_mm_per_pixel = 0.05

    report = cr.check_readiness(cfg)

    assert report.ready_for_registration
    assert report.ready_for_probing
    assert report.ready_for_gradient_processing
    assert all(c.status != "FAIL" for c in report.checks)
    assert _find(report.checks, "camera.dx_mm_per_pixel").status == "OK"
    assert _find(report.checks, "camera.dy_mm_per_pixel").status == "OK"


def test_header_offset_zero_emits_probe_hint(fake_project, tmp_path):
    _init_db(tmp_path)
    cfg = load_config("configs/default.yaml")
    assert cfg.camera.header_offset == 0

    report = cr.check_readiness(cfg)
    hdr = _find(report.checks, "camera.header_offset")
    assert hdr.status == "INFO"
    assert "probe_registered_file" in hdr.detail


def test_header_offset_nonzero_is_ok(fake_project, tmp_path):
    _init_db(tmp_path)
    cfg = load_config("configs/default.yaml")
    cfg.camera.header_offset = 512

    report = cr.check_readiness(cfg)
    hdr = _find(report.checks, "camera.header_offset")
    assert hdr.status == "OK"
    assert hdr.detail == "512"


def test_db_file_exists_but_schema_missing(fake_project, tmp_path):
    """An empty file at the DB path looks like a blank SQLite db with no tables."""
    (tmp_path / "database" / "thermal_cladding.db").touch()
    cfg = load_config("configs/default.yaml")

    report = cr.check_readiness(cfg)
    assert not report.ready_for_registration
    db = _find(report.checks, "database")
    assert db.status == "FAIL"
    assert "xtherm_files" in db.detail


def test_negative_dx_is_fail_not_warn(fake_project, tmp_path):
    _init_db(tmp_path)
    cfg = load_config("configs/default.yaml")
    cfg.camera.dx_mm_per_pixel = -0.05
    cfg.camera.dy_mm_per_pixel = 0.05

    report = cr.check_readiness(cfg)
    assert _find(report.checks, "camera.dx_mm_per_pixel").status == "FAIL"
    assert _find(report.checks, "camera.dy_mm_per_pixel").status == "OK"
    assert not report.ready_for_gradient_processing


def test_xtherm_files_counter_reflects_table_state(fake_project, tmp_path):
    """After init, table exists but is empty -> INFO count should be 0."""
    _init_db(tmp_path)
    cfg = load_config("configs/default.yaml")

    report = cr.check_readiness(cfg)
    counter = _find(report.checks, "xtherm_files registered")
    assert counter.status == "INFO"
    assert counter.detail == "0"


def test_main_cli_exit_codes(fake_project, tmp_path, capsys):
    # State 1: no DB, dirs present, default cfg -> exit 2 (not ready for registration)
    rc = cr.main(["--config", "configs/default.yaml"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "ready_for_registration        : NO" in captured.out

    # State 2: DB present, but no calibration -> exit 1 (registration/probing only)
    _init_db(tmp_path)
    rc = cr.main(["--config", "configs/default.yaml"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "ready_for_registration        : YES" in captured.out
    assert "ready_for_gradient_processing : NO" in captured.out

    # State 3: patch YAML to add dx/dy -> exit 0 (fully ready)
    yaml_path = tmp_path / "configs" / "default.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    text = text.replace("dx_mm_per_pixel: null", "dx_mm_per_pixel: 0.05")
    text = text.replace("dy_mm_per_pixel: null", "dy_mm_per_pixel: 0.05")
    yaml_path.write_text(text, encoding="utf-8")

    rc = cr.main(["--config", "configs/default.yaml"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "ready_for_gradient_processing : YES" in captured.out
