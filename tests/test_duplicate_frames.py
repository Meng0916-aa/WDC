"""Tests for src/processing/duplicate_frames.py.

All tests use small in-memory arrays — no real .xtherm dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.duplicate_frames import (
    ACTION_FLAG_NEAR,
    ACTION_KEEP,
    ACTION_REMOVE_EXACT,
    REPORT_COLUMNS,
    detect_duplicate_frames,
    remove_duplicate_frames,
)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def test_no_duplicates_all_keep():
    rng = np.random.default_rng(0)
    T = rng.normal(500, 50, size=(5, 4, 4)).astype(np.float32)
    report = detect_duplicate_frames(T)
    assert list(report.columns) == list(REPORT_COLUMNS)
    assert (report["action"] == ACTION_KEEP).all()
    assert (~report["is_exact_duplicate"]).all()
    assert (~report["is_near_duplicate"]).all()


def test_first_frame_always_keep_with_nan_diffs():
    T = np.zeros((3, 4, 4), dtype=np.float32)
    report = detect_duplicate_frames(T)
    row0 = report.iloc[0]
    assert row0["frame_index"] == 0
    assert row0["prev_frame_index"] == -1
    assert pd.isna(row0["mae_to_prev"])
    assert pd.isna(row0["max_abs_diff_to_prev"])
    assert row0["is_exact_duplicate"] == False  # noqa: E712
    assert row0["is_near_duplicate"] == False  # noqa: E712
    assert row0["action"] == ACTION_KEEP


def test_exact_duplicate_consecutive():
    T = np.zeros((3, 4, 4), dtype=np.float32)
    T[0] = 1.0
    T[1] = 1.0  # exact dup of T[0]
    T[2] = 2.0
    report = detect_duplicate_frames(T)
    assert report.iloc[1]["is_exact_duplicate"]
    assert report.iloc[1]["action"] == ACTION_REMOVE_EXACT
    assert report.iloc[2]["action"] == ACTION_KEEP
    # MAE / max_abs for the duplicate pair should both be 0
    assert report.iloc[1]["mae_to_prev"] == 0.0
    assert report.iloc[1]["max_abs_diff_to_prev"] == 0.0


def test_multiple_consecutive_duplicates_flagged_as_remove():
    T = np.zeros((4, 4, 4), dtype=np.float32)
    T[0] = 1.0
    T[1] = 1.0  # dup of 0
    T[2] = 1.0  # dup of 1
    T[3] = 2.0
    report = detect_duplicate_frames(T)
    assert report.iloc[0]["action"] == ACTION_KEEP
    assert report.iloc[1]["action"] == ACTION_REMOVE_EXACT
    assert report.iloc[2]["action"] == ACTION_REMOVE_EXACT
    assert report.iloc[3]["action"] == ACTION_KEEP


def test_near_duplicate_flagged_with_mae_threshold():
    T = np.zeros((3, 4, 4), dtype=np.float32)
    T[0] = 1.0
    T[1] = 1.01  # MAE = 0.01, not exact
    T[2] = 5.0
    report = detect_duplicate_frames(T, mae_threshold=0.05)
    assert not report.iloc[1]["is_exact_duplicate"]
    assert report.iloc[1]["is_near_duplicate"]
    assert report.iloc[1]["action"] == ACTION_FLAG_NEAR
    assert report.iloc[2]["action"] == ACTION_KEEP


def test_near_duplicate_with_max_abs_threshold_only():
    T = np.zeros((2, 4, 4), dtype=np.float32)
    T[0] = 1.0
    T[1] = 1.0          # start identical to T[0]
    T[1, 0, 0] = 1.005  # perturb one pixel by 0.005; max_abs = 0.005
    report = detect_duplicate_frames(T, max_abs_threshold=0.01)
    assert not report.iloc[1]["is_exact_duplicate"]  # 0.005 != 0
    assert report.iloc[1]["is_near_duplicate"]
    assert report.iloc[1]["action"] == ACTION_FLAG_NEAR


def test_both_thresholds_require_conjunction():
    # MAE small but max_abs big -> not near (both must hold)
    T = np.zeros((2, 4, 4), dtype=np.float32)
    T[0] = 1.0
    T[1] = 1.0
    T[1, 0, 0] = 100.0  # one pixel jumps; MAE = 99/16 = 6.1875, max_abs = 99
    report = detect_duplicate_frames(T, mae_threshold=10.0, max_abs_threshold=10.0)
    assert not report.iloc[1]["is_near_duplicate"]
    # Just MAE alone would flag it:
    report_mae_only = detect_duplicate_frames(T, mae_threshold=10.0)
    assert report_mae_only.iloc[1]["is_near_duplicate"]


def test_exact_takes_priority_over_near():
    """When exact AND near both could apply, action == remove_exact_duplicate."""
    T = np.zeros((2, 4, 4), dtype=np.float32)
    T[0] = 1.0
    T[1] = 1.0  # exact dup => MAE=0, max_abs=0, also <= any non-negative threshold
    report = detect_duplicate_frames(T, mae_threshold=0.5, max_abs_threshold=0.5)
    assert report.iloc[1]["is_exact_duplicate"]
    assert not report.iloc[1]["is_near_duplicate"]
    assert report.iloc[1]["action"] == ACTION_REMOVE_EXACT


def test_exact_false_skips_exact_check():
    T = np.zeros((2, 4, 4), dtype=np.float32)
    T[0] = 1.0
    T[1] = 1.0
    report = detect_duplicate_frames(T, exact=False)
    assert not report.iloc[1]["is_exact_duplicate"]
    assert report.iloc[1]["action"] == ACTION_KEEP

    # With a threshold, they become near-duplicates instead
    report2 = detect_duplicate_frames(T, exact=False, mae_threshold=0.5)
    assert report2.iloc[1]["is_near_duplicate"]
    assert report2.iloc[1]["action"] == ACTION_FLAG_NEAR


def test_detect_non_3d_raises():
    with pytest.raises(ValueError, match="3D"):
        detect_duplicate_frames(np.zeros((4, 4), dtype=np.float32))


def test_detect_negative_threshold_raises():
    T = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="mae_threshold"):
        detect_duplicate_frames(T, mae_threshold=-0.1)
    with pytest.raises(ValueError, match="max_abs_threshold"):
        detect_duplicate_frames(T, max_abs_threshold=-0.1)


# --------------------------------------------------------------------------- #
# removal
# --------------------------------------------------------------------------- #
def test_remove_keeps_first_of_consecutive_duplicates():
    T = np.zeros((4, 4, 4), dtype=np.float32)
    T[0] = 1.0
    T[1] = 1.0  # dup of 0
    T[2] = 1.0  # dup of 1
    T[3] = 2.0
    report = detect_duplicate_frames(T)
    dedup, keep, removed = remove_duplicate_frames(T, report)
    assert keep.tolist() == [0, 3]
    assert removed.tolist() == [1, 2]
    assert dedup.shape == (2, 4, 4)
    np.testing.assert_array_equal(dedup[0], T[0])
    np.testing.assert_array_equal(dedup[1], T[3])


def test_remove_near_default_keeps_them():
    T = np.zeros((3, 4, 4), dtype=np.float32)
    T[0] = 1.0
    T[1] = 1.01
    T[2] = 5.0
    report = detect_duplicate_frames(T, mae_threshold=0.05)
    dedup, keep, removed = remove_duplicate_frames(T, report)
    assert keep.tolist() == [0, 1, 2]
    assert removed.tolist() == []
    assert dedup.shape == T.shape


def test_remove_near_when_flag_set():
    T = np.zeros((3, 4, 4), dtype=np.float32)
    T[0] = 1.0
    T[1] = 1.01
    T[2] = 5.0
    report = detect_duplicate_frames(T, mae_threshold=0.05)
    dedup, keep, removed = remove_duplicate_frames(T, report, remove_near_duplicates=True)
    assert keep.tolist() == [0, 2]
    assert removed.tolist() == [1]
    assert dedup.shape == (2, 4, 4)


def test_keep_and_removed_indices_disjoint_and_cover():
    rng = np.random.default_rng(7)
    T = rng.normal(0, 1, size=(6, 3, 3)).astype(np.float32)
    T[3] = T[2]  # plant one duplicate
    report = detect_duplicate_frames(T)
    _, keep, removed = remove_duplicate_frames(T, report)
    union = np.sort(np.concatenate([keep, removed]))
    assert union.tolist() == list(range(T.shape[0]))
    assert set(keep).isdisjoint(set(removed))


def test_detect_does_not_modify_input():
    rng = np.random.default_rng(42)
    T = rng.normal(500, 50, size=(5, 4, 4)).astype(np.float32)
    T_orig = T.copy()
    detect_duplicate_frames(T, mae_threshold=10.0, max_abs_threshold=50.0)
    np.testing.assert_array_equal(T, T_orig)


def test_remove_does_not_modify_input():
    T = np.zeros((3, 4, 4), dtype=np.float32)
    T[1] = T[0]  # exact dup
    T[2] = 5.0
    T_orig = T.copy()
    report = detect_duplicate_frames(T)
    dedup, _, _ = remove_duplicate_frames(T, report)
    # Mutating the dedup output must not propagate back
    dedup[0] = 999.0
    np.testing.assert_array_equal(T, T_orig)


def test_remove_length_mismatch_raises():
    T = np.zeros((3, 4, 4), dtype=np.float32)
    bad = pd.DataFrame({"action": [ACTION_KEEP, ACTION_KEEP]})
    with pytest.raises(ValueError, match="length"):
        remove_duplicate_frames(T, bad)


def test_remove_missing_action_column_raises():
    T = np.zeros((2, 4, 4), dtype=np.float32)
    bad = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(ValueError, match="action"):
        remove_duplicate_frames(T, bad)


def test_remove_non_3d_raises():
    bad = np.zeros((4, 4), dtype=np.float32)
    report = pd.DataFrame({"action": [ACTION_KEEP, ACTION_KEEP, ACTION_KEEP, ACTION_KEEP]})
    with pytest.raises(ValueError, match="3D"):
        remove_duplicate_frames(bad, report)


def test_single_frame_input():
    T = np.full((1, 4, 4), 5.0, dtype=np.float32)
    report = detect_duplicate_frames(T)
    assert len(report) == 1
    assert report.iloc[0]["action"] == ACTION_KEEP
    dedup, keep, removed = remove_duplicate_frames(T, report)
    assert dedup.shape == (1, 4, 4)
    assert keep.tolist() == [0]
    assert removed.tolist() == []
