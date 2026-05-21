"""Tests for src/io/xtherm_reader.py.

Covers:
- happy path with tiny binary fixture (W=8, H=6, T=3, uint16, scale=0.1);
- size-mismatch ValueError (silent reshape is forbidden);
- max_frames truncation;
- expected_frames assertion;
- bad dtype / endian rejected at construction time.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.io.xtherm_reader import (
    XthermReadParams,
    compute_expected_layout,
    read_xtherm,
)


def test_read_tiny_xtherm_matches_expected(tiny_xtherm_file):
    path, raw, expected, p = tiny_xtherm_file
    temps = read_xtherm(
        path,
        width=p.width, height=p.height,
        dtype=p.dtype, endian=p.endian,
        header_offset=p.header_offset,
        temperature_scale=p.temperature_scale,
    )
    assert temps.dtype == np.float32
    assert temps.shape == (p.frames, p.height, p.width)
    np.testing.assert_allclose(temps, expected, rtol=0, atol=1e-6)


def test_read_xtherm_max_frames_truncates(tiny_xtherm_file):
    path, _, expected, p = tiny_xtherm_file
    temps = read_xtherm(
        path,
        width=p.width, height=p.height,
        dtype=p.dtype, endian=p.endian,
        header_offset=p.header_offset,
        temperature_scale=p.temperature_scale,
        max_frames=2,
    )
    assert temps.shape == (2, p.height, p.width)
    np.testing.assert_allclose(temps, expected[:2], atol=1e-6)


def test_read_xtherm_expected_frames_match(tiny_xtherm_file):
    path, _, _, p = tiny_xtherm_file
    read_xtherm(
        path, width=p.width, height=p.height,
        dtype=p.dtype, endian=p.endian,
        header_offset=p.header_offset,
        temperature_scale=p.temperature_scale,
        expected_frames=p.frames,
    )  # no raise


def test_read_xtherm_expected_frames_mismatch_raises(tiny_xtherm_file):
    path, _, _, p = tiny_xtherm_file
    with pytest.raises(ValueError, match="Frame count mismatch"):
        read_xtherm(
            path, width=p.width, height=p.height,
            dtype=p.dtype, endian=p.endian,
            header_offset=p.header_offset,
            temperature_scale=p.temperature_scale,
            expected_frames=p.frames + 1,
        )


def test_size_mismatch_raises(tmp_path):
    """字节数与 frame_bytes 不整除时, reader 必须拒绝 (拒绝 silent reshape)。"""
    bad = tmp_path / "bad.xtherm"
    # 写 7 个 uint16 字节流 (14 字节), 不可能整除 8x6 帧 (96 字节)
    np.arange(7, dtype="<u2").tofile(bad)
    with pytest.raises(ValueError, match="silent reshape disabled"):
        read_xtherm(bad, width=8, height=6, dtype="uint16",
                    endian="little", header_offset=0, temperature_scale=0.1)


def test_header_offset_too_large_raises(tmp_path):
    bad = tmp_path / "short.xtherm"
    bad.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="header_offset"):
        read_xtherm(bad, width=8, height=6, dtype="uint16",
                    endian="little", header_offset=4096, temperature_scale=0.1)


def test_unsupported_dtype_raises():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        XthermReadParams(width=8, height=6, dtype="complex64", endian="little",
                         header_offset=0, temperature_scale=0.1)


def test_bad_endian_raises():
    with pytest.raises(ValueError, match="endian must be"):
        XthermReadParams(width=8, height=6, dtype="uint16", endian="middle",
                         header_offset=0, temperature_scale=0.1)


def test_negative_offset_raises():
    with pytest.raises(ValueError, match="header_offset must be >= 0"):
        XthermReadParams(width=8, height=6, dtype="uint16", endian="little",
                         header_offset=-1, temperature_scale=0.1)


def test_compute_expected_layout(tiny_xtherm_file):
    path, _, _, p = tiny_xtherm_file
    params = XthermReadParams(
        width=p.width, height=p.height,
        dtype=p.dtype, endian=p.endian,
        header_offset=p.header_offset,
        temperature_scale=p.temperature_scale,
    )
    layout = compute_expected_layout(path, params)
    assert layout.n_frames == p.frames
    assert layout.bytes_per_pixel == 2
    assert layout.frame_bytes == p.width * p.height * 2
    assert layout.header_offset == p.header_offset
