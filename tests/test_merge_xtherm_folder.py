"""Tests for src/io/merge_xtherm_folder.py.

Each test builds tiny temporary single-frame .xtherm files in pytest's
tmp_path with the production-matching 56-byte header. No real Xiris data
is required.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.io.merge_xtherm_folder import (
    MergeResult,
    list_xtherm_files,
    merge_xtherm_folder,
    natural_sort_key,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
HEADER_BYTES = 56  # matches the user's probed value
W, H = 8, 6        # tiny fixture grid; 8*6*2 + 56 = 152 bytes per file


def _write_single_frame_xtherm(path: Path, frame_uint16: np.ndarray,
                               header_bytes: int = HEADER_BYTES) -> None:
    """Write `header_bytes` of filler + a single (H,W) uint16 frame."""
    assert frame_uint16.ndim == 2
    assert frame_uint16.dtype == np.uint16
    with open(path, "wb") as f:
        f.write(b"\xAA" * header_bytes)
        frame_uint16.astype("<u2", copy=False).tofile(f)


def _names(result: MergeResult) -> List[str]:
    return [Path(p).name for p in result.source_files]


# --------------------------------------------------------------------------- #
# natural sort
# --------------------------------------------------------------------------- #
def test_natural_sort_key_basic():
    assert natural_sort_key("frame2.xtherm") < natural_sort_key("frame10.xtherm")
    assert natural_sort_key("frame10.xtherm") < natural_sort_key("frame100.xtherm")
    # zero-padded names are already in order under both schemes
    seq = ["f001.xtherm", "f002.xtherm", "f010.xtherm", "f100.xtherm"]
    assert sorted(seq, key=natural_sort_key) == seq


def test_merge_uses_natural_sort(tmp_path: Path):
    # Create files in NON-natural alphabetical order deliberately
    raw_frames = {
        "frame1.xtherm":   np.full((H, W), 100, dtype=np.uint16),
        "frame2.xtherm":   np.full((H, W), 200, dtype=np.uint16),
        "frame10.xtherm":  np.full((H, W), 300, dtype=np.uint16),
        "frame20.xtherm":  np.full((H, W), 400, dtype=np.uint16),
        "frame100.xtherm": np.full((H, W), 500, dtype=np.uint16),
    }
    # Write in arbitrary (insertion) order
    for name, frame in raw_frames.items():
        _write_single_frame_xtherm(tmp_path / name, frame)

    result = merge_xtherm_folder(
        tmp_path, width=W, height=H, dtype="uint16",
        endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
    )
    assert _names(result) == [
        "frame1.xtherm", "frame2.xtherm", "frame10.xtherm",
        "frame20.xtherm", "frame100.xtherm",
    ]
    # Each frame is constant -> temperature[i].mean() reflects raw_value * 0.1
    expected_mean = [10.0, 20.0, 30.0, 40.0, 50.0]
    means = [float(result.temperature[i].mean()) for i in range(result.n_files)]
    np.testing.assert_allclose(means, expected_mean, atol=1e-6)


# --------------------------------------------------------------------------- #
# shape, dtype, temperature scale
# --------------------------------------------------------------------------- #
def test_merge_shape_and_dtype(tmp_path: Path):
    N = 4
    for i in range(N):
        _write_single_frame_xtherm(
            tmp_path / f"f{i:03d}.xtherm",
            np.full((H, W), i * 100, dtype=np.uint16),
        )
    result = merge_xtherm_folder(
        tmp_path, width=W, height=H, dtype="uint16",
        endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
    )
    assert result.temperature.shape == (N, H, W)
    assert result.temperature.dtype == np.float32
    assert result.n_files == N


def test_temperature_scale_conversion(tmp_path: Path):
    """raw_value * temperature_scale == degC."""
    raw = np.full((H, W), 5000, dtype=np.uint16)  # 5000 * 0.1 = 500.0 degC
    _write_single_frame_xtherm(tmp_path / "f0.xtherm", raw)

    result = merge_xtherm_folder(
        tmp_path, width=W, height=H, dtype="uint16",
        endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
    )
    np.testing.assert_allclose(result.temperature[0], 500.0, atol=1e-6)
    # Different scale -> linear
    result_5 = merge_xtherm_folder(
        tmp_path, width=W, height=H, dtype="uint16",
        endian="little", header_offset=HEADER_BYTES, temperature_scale=0.05,
    )
    np.testing.assert_allclose(result_5.temperature[0], 250.0, atol=1e-6)


def test_per_pixel_values_preserved(tmp_path: Path):
    """Ramp pattern: ensure pixels arrive in (H, W) order, not transposed."""
    raw = np.arange(H * W, dtype=np.uint16).reshape(H, W)
    _write_single_frame_xtherm(tmp_path / "f0.xtherm", raw)

    result = merge_xtherm_folder(
        tmp_path, width=W, height=H, dtype="uint16",
        endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
    )
    np.testing.assert_allclose(
        result.temperature[0],
        raw.astype(np.float32) * 0.1,
        atol=1e-6,
    )


# --------------------------------------------------------------------------- #
# source_files
# --------------------------------------------------------------------------- #
def test_source_files_recorded_in_merged_order(tmp_path: Path):
    names = ["a.xtherm", "b.xtherm", "c.xtherm"]
    for n in names:
        _write_single_frame_xtherm(tmp_path / n, np.zeros((H, W), dtype=np.uint16))

    result = merge_xtherm_folder(
        tmp_path, width=W, height=H, dtype="uint16",
        endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
    )
    assert _names(result) == names
    # source_files are posix paths
    for s in result.source_files:
        assert "\\" not in s, f"expected posix path, got {s!r}"


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #
def test_file_size_mismatch_raises(tmp_path: Path):
    """A file 1 byte short of (header + W*H*2) must raise (no silent reshape)."""
    bad = tmp_path / "bad.xtherm"
    with open(bad, "wb") as f:
        f.write(b"\xAA" * HEADER_BYTES)
        f.write(b"\x00" * (W * H * 2 - 1))  # one byte short

    with pytest.raises(ValueError, match="silent reshape disabled"):
        merge_xtherm_folder(
            tmp_path, width=W, height=H, dtype="uint16",
            endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
        )


def test_two_frames_in_one_file_raises(tmp_path: Path):
    """A file with payload of 2 frames must raise (expected_frames=1)."""
    bad = tmp_path / "twoframes.xtherm"
    with open(bad, "wb") as f:
        f.write(b"\xAA" * HEADER_BYTES)
        f.write(b"\x00" * (W * H * 2 * 2))  # 2 frames worth

    with pytest.raises(ValueError, match="Frame count mismatch"):
        merge_xtherm_folder(
            tmp_path, width=W, height=H, dtype="uint16",
            endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
        )


def test_empty_folder_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No files matching"):
        merge_xtherm_folder(
            tmp_path, width=W, height=H, dtype="uint16",
            endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
        )


def test_missing_folder_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        merge_xtherm_folder(
            missing, width=W, height=H, dtype="uint16",
            endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
        )


def test_not_a_directory_raises(tmp_path: Path):
    f = tmp_path / "not_a_dir"
    f.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        merge_xtherm_folder(
            f, width=W, height=H, dtype="uint16",
            endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
        )


def test_pattern_filters_non_matching_files(tmp_path: Path):
    _write_single_frame_xtherm(tmp_path / "frame1.xtherm",
                               np.full((H, W), 1, dtype=np.uint16))
    # Distractor file with the same payload but wrong extension
    other = tmp_path / "frame1.bin"
    _write_single_frame_xtherm(other, np.full((H, W), 2, dtype=np.uint16))

    result = merge_xtherm_folder(
        tmp_path, width=W, height=H, dtype="uint16",
        endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
        pattern="*.xtherm",
    )
    assert result.n_files == 1
    assert _names(result) == ["frame1.xtherm"]


def test_recursive_scan(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    _write_single_frame_xtherm(tmp_path / "top.xtherm",
                               np.full((H, W), 1, dtype=np.uint16))
    _write_single_frame_xtherm(tmp_path / "sub" / "deep.xtherm",
                               np.full((H, W), 2, dtype=np.uint16))

    result_flat = merge_xtherm_folder(
        tmp_path, width=W, height=H, dtype="uint16",
        endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
        recursive=False,
    )
    assert result_flat.n_files == 1

    result_rec = merge_xtherm_folder(
        tmp_path, width=W, height=H, dtype="uint16",
        endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
        recursive=True,
    )
    assert result_rec.n_files == 2


def test_source_files_not_modified(tmp_path: Path):
    """Merging is strictly read-only on disk."""
    payloads = {}
    for i in range(3):
        path = tmp_path / f"f{i}.xtherm"
        _write_single_frame_xtherm(path, np.full((H, W), i + 1, dtype=np.uint16))
        payloads[path.name] = path.read_bytes()

    merge_xtherm_folder(
        tmp_path, width=W, height=H, dtype="uint16",
        endian="little", header_offset=HEADER_BYTES, temperature_scale=0.1,
    )

    for path in sorted(tmp_path.glob("*.xtherm")):
        assert path.read_bytes() == payloads[path.name]


def test_list_xtherm_files_returns_paths(tmp_path: Path):
    for n in ["c.xtherm", "a.xtherm", "b.xtherm"]:
        _write_single_frame_xtherm(tmp_path / n, np.zeros((H, W), dtype=np.uint16))
    paths = list_xtherm_files(tmp_path)
    assert [p.name for p in paths] == ["a.xtherm", "b.xtherm", "c.xtherm"]
    assert all(isinstance(p, Path) for p in paths)
