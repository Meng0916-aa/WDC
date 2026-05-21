"""Tests for scripts/extract_features_from_npz.py."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import extract_features_from_npz as ex  # noqa: E402

from src.utils import load_config  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures (local; don't touch existing conftest)
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_project(tmp_path, monkeypatch) -> Path:
    """Minimal project tree + monkey-patched PROJECT_ROOT."""
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


def _write_npz(
    path: Path,
    *,
    T_cube: np.ndarray,
    fps: Optional[float] = None,
    original_frame_indices: Optional[np.ndarray] = None,
    **extra,
) -> None:
    kw = {"temperature": T_cube}
    if fps is not None:
        kw["fps"] = float(fps)
    if original_frame_indices is not None:
        kw["original_frame_indices"] = np.asarray(original_frame_indices)
    kw.update(extra)
    np.savez(path, **kw)


def _cfg_with_calibration(dx: float = 0.05, dy: float = 0.05):
    cfg = load_config("configs/default.yaml")
    cfg.camera.dx_mm_per_pixel = dx
    cfg.camera.dy_mm_per_pixel = dy
    cfg.processing.gaussian_sigma_px = 0.0  # exact comparisons in tests
    return cfg


# --------------------------------------------------------------------------- #
# time_s priority
# --------------------------------------------------------------------------- #
def test_time_s_uses_original_frame_indices_over_fps(fake_project, tmp_path):
    """Spec: original_frame_indices / fps takes priority."""
    H, W, T = 6, 8, 3
    T_cube = np.zeros((T, H, W), dtype=np.float32)
    npz = tmp_path / "in.npz"
    _write_npz(npz, T_cube=T_cube, fps=180.0, original_frame_indices=np.array([0, 2, 5]))
    cfg = _cfg_with_calibration()

    df = ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")
    np.testing.assert_allclose(
        df["time_s"].to_numpy(),
        np.array([0, 2, 5], dtype=np.float64) / 180.0,
        atol=1e-9,
    )


def test_time_s_falls_back_to_frame_over_fps_when_no_indices(fake_project, tmp_path):
    """Spec: without original_frame_indices, use frame_index / fps."""
    H, W, T = 6, 8, 4
    T_cube = np.zeros((T, H, W), dtype=np.float32)
    npz = tmp_path / "in.npz"
    _write_npz(npz, T_cube=T_cube, fps=180.0)  # no original_frame_indices
    cfg = _cfg_with_calibration()

    df = ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")
    np.testing.assert_allclose(
        df["time_s"].to_numpy(),
        np.arange(T, dtype=np.float64) / 180.0,
        atol=1e-9,
    )


def test_fps_from_npz_overrides_config(fake_project, tmp_path):
    """fps stored in NPZ wins over config.camera.fps."""
    H, W, T = 4, 4, 3
    T_cube = np.zeros((T, H, W), dtype=np.float32)
    npz = tmp_path / "in.npz"
    _write_npz(npz, T_cube=T_cube, fps=60.0)  # NPZ says 60, config defaults 180
    cfg = _cfg_with_calibration()
    assert cfg.camera.fps == 180.0  # sanity

    df = ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")
    np.testing.assert_allclose(df["time_s"].to_numpy(), np.arange(T) / 60.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #
def test_dx_dy_missing_raises(fake_project, tmp_path):
    H, W, T = 6, 8, 2
    npz = tmp_path / "in.npz"
    _write_npz(npz, T_cube=np.zeros((T, H, W), dtype=np.float32), fps=180.0)
    cfg = load_config("configs/default.yaml")
    cfg.camera.dx_mm_per_pixel = None
    cfg.camera.dy_mm_per_pixel = None

    with pytest.raises(ValueError, match="dx_mm_per_pixel"):
        ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")


def test_missing_temperature_key_raises(fake_project, tmp_path):
    npz = tmp_path / "bad.npz"
    np.savez(npz, foo=np.zeros((1, 4, 4)))
    cfg = _cfg_with_calibration()
    with pytest.raises(KeyError, match="temperature"):
        ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")


def test_non_3d_temperature_raises(fake_project, tmp_path):
    npz = tmp_path / "bad.npz"
    np.savez(npz, temperature=np.zeros((4, 4), dtype=np.float32))  # 2D
    cfg = _cfg_with_calibration()
    with pytest.raises(ValueError, match="3D"):
        ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")


def test_original_frame_indices_length_mismatch_raises(fake_project, tmp_path):
    H, W, T = 4, 4, 3
    npz = tmp_path / "bad.npz"
    np.savez(
        npz,
        temperature=np.zeros((T, H, W), dtype=np.float32),
        fps=180.0,
        original_frame_indices=np.array([0, 1]),  # length 2 != T=3
    )
    cfg = _cfg_with_calibration()
    with pytest.raises(ValueError, match="original_frame_indices"):
        ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")


def test_missing_npz_file_raises(fake_project, tmp_path):
    cfg = _cfg_with_calibration()
    with pytest.raises(FileNotFoundError):
        ex.extract_features_from_npz(tmp_path / "nope.npz", cfg, tmp_path / "out.csv")


# --------------------------------------------------------------------------- #
# CSV contents
# --------------------------------------------------------------------------- #
def test_csv_columns_complete_and_in_order(fake_project, tmp_path):
    H, W, T = 6, 8, 2
    T_cube = np.zeros((T, H, W), dtype=np.float32)
    npz = tmp_path / "in.npz"
    _write_npz(npz, T_cube=T_cube, fps=180.0)
    cfg = _cfg_with_calibration()

    out = tmp_path / "out.csv"
    df = ex.extract_features_from_npz(npz, cfg, out)

    expected = [
        "frame", "time_s",
        "Tmax", "Tmean", "Tstd",
        "Gmax", "Gmean", "Gstd",
        "high_temp_area",
    ]
    assert list(df.columns) == expected
    # Round-trip via CSV must preserve column order
    df_loaded = pd.read_csv(out)
    assert list(df_loaded.columns) == expected
    assert len(df_loaded) == T


def test_constant_field_gives_zero_gradient(fake_project, tmp_path):
    H, W, T = 6, 8, 2
    T_cube = np.full((T, H, W), 500.0, dtype=np.float32)
    npz = tmp_path / "in.npz"
    _write_npz(npz, T_cube=T_cube, fps=180.0)
    cfg = _cfg_with_calibration()

    df = ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")
    np.testing.assert_allclose(df["Gmax"].to_numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(df["Gmean"].to_numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(df["Gstd"].to_numpy(), 0.0, atol=1e-6)
    assert (df["Tmax"] == 500.0).all()
    assert (df["Tmean"] == 500.0).all()
    assert (df["Tstd"] == 0.0).all()


def test_linear_x_ramp_gives_correct_gmax(fake_project, tmp_path):
    """T(y, x) = 10 * x, dx = 0.5 mm/pixel  =>  Gx = 10/0.5 = 20.0 degC/mm everywhere.

    np.gradient on a perfectly linear ramp gives a UNIFORM slope (forward
    diff at the left edge, central diff in the middle, backward diff at
    the right edge -- all 10 for this ramp), so Gmean is exactly Gmax = 20.
    """
    H, W = 6, 8
    ramp = np.tile(np.arange(W, dtype=np.float32) * 10.0, (H, 1))
    T_cube = ramp[None, :, :].copy()
    npz = tmp_path / "in.npz"
    _write_npz(npz, T_cube=T_cube, fps=180.0)

    cfg = _cfg_with_calibration(dx=0.5, dy=0.5)
    df = ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")
    np.testing.assert_allclose(df.iloc[0]["Gmax"], 20.0, atol=1e-4)
    np.testing.assert_allclose(df.iloc[0]["Gmean"], 20.0, atol=1e-4)
    # std of a uniform field is 0 (any tiny numeric noise is OK)
    np.testing.assert_allclose(df.iloc[0]["Gstd"], 0.0, atol=1e-4)


def test_linear_y_ramp_gives_correct_gmax(fake_project, tmp_path):
    """T(y, x) = 5 * y, dy = 0.1 mm/pixel  =>  interior Gy = 5/0.1 = 50.0 degC/mm."""
    H, W = 6, 8
    ramp = np.tile((np.arange(H, dtype=np.float32) * 5.0)[:, None], (1, W))
    T_cube = ramp[None, :, :].copy()
    npz = tmp_path / "in.npz"
    _write_npz(npz, T_cube=T_cube, fps=180.0)

    cfg = _cfg_with_calibration(dx=0.2, dy=0.1)
    df = ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")
    np.testing.assert_allclose(df.iloc[0]["Gmax"], 50.0, atol=1e-4)


def test_high_temp_threshold_applied(fake_project, tmp_path):
    """high_temp_area counts pixels above cfg.processing.high_temp_threshold_C (default 1000)."""
    H, W = 4, 4
    frame = np.full((H, W), 200.0, dtype=np.float32)
    frame[0, 0] = 1500.0
    frame[0, 1] = 1500.0
    T_cube = frame[None, :, :].copy()
    npz = tmp_path / "in.npz"
    _write_npz(npz, T_cube=T_cube, fps=180.0)
    cfg = _cfg_with_calibration()

    df = ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")
    assert df.iloc[0]["high_temp_area"] == 2


# --------------------------------------------------------------------------- #
# side-effect / output policy
# --------------------------------------------------------------------------- #
def test_does_not_modify_input_npz(fake_project, tmp_path):
    H, W, T = 4, 4, 2
    npz = tmp_path / "in.npz"
    _write_npz(
        npz,
        T_cube=np.full((T, H, W), 100.0, dtype=np.float32),
        fps=180.0,
        original_frame_indices=np.array([0, 1]),
    )
    original_bytes = npz.read_bytes()

    cfg = _cfg_with_calibration()
    ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")
    assert npz.read_bytes() == original_bytes


def test_save_gradient_stats_npz_when_requested(fake_project, tmp_path):
    H, W, T = 4, 4, 3
    npz = tmp_path / "in.npz"
    _write_npz(npz, T_cube=np.full((T, H, W), 100.0, dtype=np.float32), fps=180.0)
    cfg = _cfg_with_calibration()

    out_csv = tmp_path / "out.csv"
    out_stats = tmp_path / "out_stats.npz"
    ex.extract_features_from_npz(npz, cfg, out_csv, save_gradient_stats_npz=out_stats)

    assert out_stats.exists()
    data = np.load(out_stats, allow_pickle=False)
    assert set(data.files) == {"frame", "time_s", "Gmax", "Gmean", "Gstd"}
    assert data["frame"].shape == (T,)
    assert data["Gmax"].shape == (T,)


def test_no_gradient_stats_npz_by_default(fake_project, tmp_path):
    """Default behavior: only the CSV is written; no extra NPZ side-effect."""
    H, W, T = 4, 4, 2
    npz = tmp_path / "in.npz"
    _write_npz(npz, T_cube=np.zeros((T, H, W), dtype=np.float32), fps=180.0)
    cfg = _cfg_with_calibration()

    before = set(tmp_path.iterdir())
    ex.extract_features_from_npz(npz, cfg, tmp_path / "out.csv")
    after = set(tmp_path.iterdir())
    new_files = {p for p in after - before if p.suffix == ".npz"}
    assert not new_files, f"unexpected new NPZ files: {new_files}"


# --------------------------------------------------------------------------- #
# default output path
# --------------------------------------------------------------------------- #
def test_default_output_csv_strips_temperature_dedup(fake_project):
    cfg = load_config("configs/default.yaml")
    p = Path("data/processed/B000_sample01_temperature_sequence_temperature_dedup.npz")
    out = ex._default_output_csv(p, cfg)
    assert out.name == "B000_sample01_temperature_sequence_features.csv"
    assert "features" in out.parent.as_posix()  # data/features


def test_default_output_csv_strips_temperature_sequence(fake_project):
    cfg = load_config("configs/default.yaml")
    p = Path("data/processed/B000_sample01_temperature_sequence.npz")
    out = ex._default_output_csv(p, cfg)
    assert out.name == "B000_sample01_features.csv"


def test_default_output_csv_unknown_suffix(fake_project):
    cfg = load_config("configs/default.yaml")
    p = Path("data/processed/some_random.npz")
    out = ex._default_output_csv(p, cfg)
    assert out.name == "some_random_features.csv"
