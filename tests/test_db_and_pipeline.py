"""Tests for src/db (schema + registry) and pipeline integration on tiny fixture.

This test does NOT touch real .xtherm files; it uses the temporary tiny
binary fixture produced by conftest.py, registers it into a fresh
in-temp-dir database, runs the pipeline end-to-end, and checks the resulting
rows in xtherm_files / processing_results / frame_features.
"""
from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.db import (
    ensure_default_experiment,
    ensure_sample,
    fetch_xtherm_file,
    init_schema,
    insert_frame_features,
    list_files_by_status,
    open_db,
    register_xtherm_file,
    update_file_status,
    upsert_processing_result,
)
from src.pipeline import process_registered_file
from src.utils import load_config


@pytest.fixture
def project_root_in_tmp(tmp_path: Path, monkeypatch) -> Path:
    """Copy configs/default.yaml into a temp project dir and patch PROJECT_ROOT.

    This lets the pipeline write feature CSVs and the DB into tmp_path without
    polluting the real D:\\GEJ-WDC tree.
    """
    src_root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    shutil.copyfile(src_root / "configs" / "default.yaml", tmp_path / "configs" / "default.yaml")
    for sub in ("database", "data/raw", "data/processed", "data/features", "results/figures"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    # Patch the cached PROJECT_ROOT used by resolve_under_root
    import src.utils.paths as paths_mod
    monkeypatch.setattr(paths_mod, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_init_schema_idempotent(tmp_db_path):
    conn = open_db(tmp_db_path)
    init_schema(conn)
    init_schema(conn)  # second call should not error
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"experiments", "samples", "xtherm_files", "processing_results", "frame_features"}.issubset(tables)
    conn.close()


def test_register_xtherm_idempotent(tmp_db_path, tiny_xtherm_file):
    path, _, _, p = tiny_xtherm_file
    conn = open_db(tmp_db_path)
    init_schema(conn)
    exp_id = ensure_default_experiment(
        conn, name="exp", powder_material="CoCrNi", substrate_material="316L",
        laser_power_W=450, scan_speed_mm_per_min=800,
        powder_feed_rate_g_per_min=40, hatch_spacing_mm=0.8,
    )
    sample_pk = ensure_sample(conn, sample_id="S1", experiment_id=exp_id, B_mT=0)
    fid1 = register_xtherm_file(
        conn, sample_pk=sample_pk, file_path=path,
        width=p.width, height=p.height, dtype=p.dtype, endian=p.endian,
        header_offset=p.header_offset, temperature_scale=p.temperature_scale,
    )
    fid2 = register_xtherm_file(
        conn, sample_pk=sample_pk, file_path=path,
        width=p.width, height=p.height, dtype=p.dtype, endian=p.endian,
        header_offset=p.header_offset, temperature_scale=p.temperature_scale,
    )
    assert fid1 == fid2
    row = fetch_xtherm_file(conn, fid1)
    assert row["status"] == "registered"
    assert row["estimated_frames"] == p.frames
    conn.close()


def test_ensure_sample_rejects_inconsistent_B(tmp_db_path):
    conn = open_db(tmp_db_path)
    init_schema(conn)
    exp_id = ensure_default_experiment(
        conn, name="exp", powder_material="CoCrNi", substrate_material="316L",
        laser_power_W=450, scan_speed_mm_per_min=800,
        powder_feed_rate_g_per_min=40, hatch_spacing_mm=0.8,
    )
    ensure_sample(conn, sample_id="S2", experiment_id=exp_id, B_mT=40)
    with pytest.raises(ValueError, match="already exists"):
        ensure_sample(conn, sample_id="S2", experiment_id=exp_id, B_mT=80)
    conn.close()


def test_pipeline_processes_tiny_file_end_to_end(
    project_root_in_tmp, tiny_xtherm_file, monkeypatch
):
    """Register the tiny fixture, run pipeline, verify rows + CSV file."""
    path, _, expected, p = tiny_xtherm_file
    # The fixture file is in pytest tmp_path; PROJECT_ROOT is also tmp_path
    # but path may be elsewhere — pipeline stores file_path as posix; resolution
    # will use absolute paths because path is outside the new PROJECT_ROOT.

    # Patch camera dx/dy in config to non-null values so gradient compute works.
    cfg = load_config("configs/default.yaml")
    cfg.camera.dx_mm_per_pixel = 0.05
    cfg.camera.dy_mm_per_pixel = 0.05

    conn = open_db(cfg.paths.database_abs())
    init_schema(conn)
    exp_id = ensure_default_experiment(
        conn, name=cfg.experiment.name,
        powder_material=cfg.experiment.powder_material,
        substrate_material=cfg.experiment.substrate_material,
        laser_power_W=cfg.experiment.laser_power_W,
        scan_speed_mm_per_min=cfg.experiment.scan_speed_mm_per_min,
        powder_feed_rate_g_per_min=cfg.experiment.powder_feed_rate_g_per_min,
        hatch_spacing_mm=cfg.experiment.hatch_spacing_mm,
    )
    sample_pk = ensure_sample(conn, sample_id="B000_test", experiment_id=exp_id, B_mT=0)
    file_id = register_xtherm_file(
        conn, sample_pk=sample_pk, file_path=path,
        width=p.width, height=p.height, dtype=p.dtype, endian=p.endian,
        header_offset=p.header_offset, temperature_scale=p.temperature_scale,
        fps=180, dx_mm_per_pixel=0.05, dy_mm_per_pixel=0.05,
    )

    outcome = process_registered_file(conn, file_id, cfg)
    assert outcome.ok, f"pipeline failed: {outcome.error}"
    assert outcome.n_frames == p.frames

    # row state
    row = fetch_xtherm_file(conn, file_id)
    assert row["status"] == "processed"

    # processing_results summary
    pr = conn.execute(
        "SELECT * FROM processing_results WHERE xtherm_file_id = ?",
        (file_id,),
    ).fetchone()
    assert pr is not None
    assert pr["status"] == "success"
    assert pr["n_frames"] == p.frames
    np.testing.assert_allclose(pr["Tmax_global"], float(expected.max()), atol=1e-5)

    # frame_features rows
    rows = conn.execute(
        "SELECT * FROM frame_features WHERE xtherm_file_id = ? ORDER BY frame_index",
        (file_id,),
    ).fetchall()
    assert len(rows) == p.frames
    for t, row_t in enumerate(rows):
        np.testing.assert_allclose(row_t["Tmax"], float(expected[t].max()), atol=1e-5)
        np.testing.assert_allclose(row_t["Tmean"], float(expected[t].mean()), atol=1e-5)

    # feature CSV exists
    assert outcome.feature_csv is not None
    assert Path(outcome.feature_csv).exists()
    conn.close()


def test_pipeline_records_error_when_dx_missing(
    project_root_in_tmp, tiny_xtherm_file
):
    """No dx/dy anywhere -> pipeline must catch ValueError and mark error."""
    path, _, _, p = tiny_xtherm_file
    cfg = load_config("configs/default.yaml")
    cfg.camera.dx_mm_per_pixel = None
    cfg.camera.dy_mm_per_pixel = None

    conn = open_db(cfg.paths.database_abs())
    init_schema(conn)
    exp_id = ensure_default_experiment(
        conn, name=cfg.experiment.name,
        powder_material=cfg.experiment.powder_material,
        substrate_material=cfg.experiment.substrate_material,
        laser_power_W=cfg.experiment.laser_power_W,
        scan_speed_mm_per_min=cfg.experiment.scan_speed_mm_per_min,
        powder_feed_rate_g_per_min=cfg.experiment.powder_feed_rate_g_per_min,
        hatch_spacing_mm=cfg.experiment.hatch_spacing_mm,
    )
    sample_pk = ensure_sample(conn, sample_id="B000_err", experiment_id=exp_id, B_mT=0)
    file_id = register_xtherm_file(
        conn, sample_pk=sample_pk, file_path=path,
        width=p.width, height=p.height, dtype=p.dtype, endian=p.endian,
        header_offset=p.header_offset, temperature_scale=p.temperature_scale,
        fps=180,  # no dx/dy on the file row either
    )
    outcome = process_registered_file(conn, file_id, cfg)
    assert not outcome.ok
    row = fetch_xtherm_file(conn, file_id)
    assert row["status"] == "error"
    pr = conn.execute(
        "SELECT * FROM processing_results WHERE xtherm_file_id = ?", (file_id,)
    ).fetchone()
    assert pr["status"] == "error"
    assert pr["error_message"] is not None
    assert "dx_mm_per_pixel" in pr["error_message"]
    conn.close()


def test_list_files_by_status(tmp_db_path, tiny_xtherm_file):
    path, _, _, p = tiny_xtherm_file
    conn = open_db(tmp_db_path)
    init_schema(conn)
    exp_id = ensure_default_experiment(
        conn, name="exp", powder_material="CoCrNi", substrate_material="316L",
        laser_power_W=450, scan_speed_mm_per_min=800,
        powder_feed_rate_g_per_min=40, hatch_spacing_mm=0.8,
    )
    sample_pk = ensure_sample(conn, sample_id="S3", experiment_id=exp_id, B_mT=40)
    fid = register_xtherm_file(
        conn, sample_pk=sample_pk, file_path=path,
        width=p.width, height=p.height, dtype=p.dtype, endian=p.endian,
        header_offset=p.header_offset, temperature_scale=p.temperature_scale,
    )
    update_file_status(conn, fid, "probed", estimated_frames=p.frames)
    assert len(list_files_by_status(conn, status="probed")) == 1
    assert len(list_files_by_status(conn, status="registered")) == 0
    assert len(list_files_by_status(conn, status=None)) == 1
    conn.close()
