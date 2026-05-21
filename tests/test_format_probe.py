"""Tests for src/io/format_probe.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.io.format_probe import probe_xtherm


def test_probe_finds_offset_zero(tiny_xtherm_file):
    path, _, _, p = tiny_xtherm_file
    report = probe_xtherm(
        path,
        width=p.width, height=p.height,
        dtype=p.dtype, endian=p.endian,
        temperature_scale=p.temperature_scale,
        header_offset_candidates=[0, 128, 256, 512],
        sample_frames_spec=[0, "middle", -1],
    )
    assert report.best is not None
    assert report.best.header_offset == 0
    assert report.best.n_frames == p.frames
    # T range plausible
    assert report.best.sample_T_min_C >= 0
    assert report.best.sample_T_max_C <= 100  # 我们 fixture 的 raw 最大约 450 -> 45°C


def test_probe_finds_offset_128(tiny_xtherm_file_with_header):
    path, _, _, hoff, p = tiny_xtherm_file_with_header
    report = probe_xtherm(
        path,
        width=p.width, height=p.height,
        dtype=p.dtype, endian=p.endian,
        temperature_scale=p.temperature_scale,
        header_offset_candidates=[0, 64, 128, 256, 512, 1024],
        sample_frames_spec=[0, "middle", -1],
    )
    # offset=0 不会整除 (因 header 是 128 字节, 总大小 128 + 3*96 = 416, 不被 96 整除)
    cand_0 = next(c for c in report.candidates if c.header_offset == 0)
    assert cand_0.n_frames == 0
    # offset=128 应该正好对上
    cand_128 = next(c for c in report.candidates if c.header_offset == 128)
    assert cand_128.n_frames == p.frames
    assert report.best is not None
    assert report.best.header_offset == hoff


def test_probe_rejects_implausible_temp_range(tiny_xtherm_file):
    """若 plausible_temp_range_C 设得离谱, best 应回落到最大整除候选。"""
    path, _, _, p = tiny_xtherm_file
    report = probe_xtherm(
        path,
        width=p.width, height=p.height,
        dtype=p.dtype, endian=p.endian,
        temperature_scale=p.temperature_scale,
        header_offset_candidates=[0, 128],
        sample_frames_spec=[0, -1],
        plausible_temp_range_C=(1e6, 2e6),  # 没有任何候选合理
    )
    # 此时 plausible 列表为空, best 应回落到整除帧数最多的那个
    assert report.best is not None
    assert report.best.header_offset == 0
    assert report.best.n_frames == p.frames


def test_probe_does_not_modify_file(tiny_xtherm_file):
    path, _, _, p = tiny_xtherm_file
    original = path.read_bytes()
    probe_xtherm(
        path,
        width=p.width, height=p.height,
        dtype=p.dtype, endian=p.endian,
        temperature_scale=p.temperature_scale,
        header_offset_candidates=[0, 128],
        sample_frames_spec=[0, "middle", -1],
    )
    assert path.read_bytes() == original


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_xtherm(
            tmp_path / "nonexistent.xtherm",
            width=8, height=6, dtype="uint16", endian="little",
            temperature_scale=0.1,
        )
