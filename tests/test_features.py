"""Tests for src/processing/features.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.processing.features import FRAME_FEATURE_COLUMNS, extract_frame_features


def test_extract_features_keys_match_columns():
    T = np.full((6, 8), 100.0, dtype=np.float32)
    G = np.zeros_like(T)
    feat = extract_frame_features(T, G, frame_index=0, time_s=0.0)
    assert set(feat.keys()) == set(FRAME_FEATURE_COLUMNS)


def test_extract_features_constant_temperature():
    T = np.full((6, 8), 50.0, dtype=np.float32)
    G = np.zeros_like(T)
    feat = extract_frame_features(T, G, frame_index=1, time_s=0.5,
                                  high_temp_threshold_C=1000.0)
    assert feat["frame_index"] == 1
    assert feat["time_s"] == 0.5
    assert feat["Tmax"] == 50.0
    assert feat["Tmean"] == 50.0
    assert feat["Tstd"] == 0.0
    assert feat["Gmax"] == 0.0
    assert feat["Gmean"] == 0.0
    assert feat["Gstd"] == 0.0
    assert feat["high_temp_area"] == 0


def test_extract_features_high_temp_threshold():
    T = np.array([[200, 1200], [1500, 800]], dtype=np.float32)
    G = np.zeros_like(T)
    feat = extract_frame_features(T, G, frame_index=0, high_temp_threshold_C=1000.0)
    assert feat["high_temp_area"] == 2  # 1200 and 1500 exceed 1000
    assert feat["Tmax"] == 1500.0


def test_extract_features_gradient_stats():
    T = np.zeros((4, 4), dtype=np.float32)
    G = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float32,
    )
    feat = extract_frame_features(T, G, frame_index=0)
    assert feat["Gmax"] == 16.0
    np.testing.assert_allclose(feat["Gmean"], G.mean(), rtol=1e-6)
    np.testing.assert_allclose(feat["Gstd"], G.std(), rtol=1e-6)


def test_extract_features_shape_mismatch_raises():
    T = np.zeros((6, 8), dtype=np.float32)
    G = np.zeros((6, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        extract_frame_features(T, G, frame_index=0)


def test_extract_features_non_2d_raises():
    T = np.zeros((3, 6, 8), dtype=np.float32)
    G = np.zeros_like(T)
    with pytest.raises(ValueError, match="must be 2D"):
        extract_frame_features(T, G, frame_index=0)


def test_extract_features_time_s_none():
    T = np.zeros((4, 4), dtype=np.float32)
    G = np.zeros_like(T)
    feat = extract_frame_features(T, G, frame_index=2, time_s=None)
    assert feat["time_s"] is None
    assert feat["frame_index"] == 2
