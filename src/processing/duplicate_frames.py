"""Detect and remove duplicate frames in a temperature stack.

Motivation
----------
Xiris VXIR-3000 + WeldStudio occasionally emits repeated frames (the same
raw payload streamed twice in a row), most often under heavy load or in
high-fps AOI mode. Such duplicates inflate apparent frame count, bias
features like Tmean / time_s, and corrupt time-series gradient stats. This
module is a QA / quality-check stage; it does NOT modify the original
.xtherm. It produces a deduplicated cube + a per-frame audit report.

Comparison logic
----------------
For each frame at index t > 0, the report compares against frame t-1
(the previous frame in the original stream order, not the previous
*kept* frame). This lets multi-frame runs of identical frames cascade
correctly: A, A, A -> frame 1 is dup of 0, frame 2 is dup of 1, removal
keeps only the first.

- ``exact``: ``np.array_equal(frame_t, frame_{t-1})``
- ``mae_to_prev``: ``mean(|frame_t - frame_{t-1}|)``
- ``max_abs_diff_to_prev``: ``max(|frame_t - frame_{t-1}|)``
- ``is_near_duplicate``: if both thresholds are given, BOTH must hold;
  if only one is given, just that one is used. Never True for frame 0.

Actions (priority: exact > near > keep):
- ``remove_exact_duplicate``: ``is_exact_duplicate`` is True (only when exact=True)
- ``flag_near_duplicate``: not exact-dup, but near-dup
- ``keep``: everything else (including frame 0 always)

Unit awareness
--------------
``mae_threshold`` and ``max_abs_threshold`` are in the same unit as the
input ``temperature`` array. If you pass degC, thresholds are degC; if you
pass raw counts, they are counts. Default temperature_scale=0.1 in this
project means ``1 raw count = 0.1 degC``, so a 1-count flicker in
WeldStudio shows up as a 0.1 degC delta in degC arrays.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


REPORT_COLUMNS: Tuple[str, ...] = (
    "frame_index",
    "prev_frame_index",
    "is_exact_duplicate",
    "mae_to_prev",
    "max_abs_diff_to_prev",
    "is_near_duplicate",
    "action",
)

ACTION_KEEP = "keep"
ACTION_REMOVE_EXACT = "remove_exact_duplicate"
ACTION_FLAG_NEAR = "flag_near_duplicate"


def detect_duplicate_frames(
    temperature: np.ndarray,
    *,
    exact: bool = True,
    mae_threshold: Optional[float] = None,
    max_abs_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Build a per-frame duplicate report for a temperature stack.

    Parameters
    ----------
    temperature : np.ndarray
        shape (T, H, W); any numeric dtype. The array is NOT modified.
    exact : bool
        If True, frames identical to the previous frame are flagged as
        ``is_exact_duplicate=True`` and assigned ``action='remove_exact_duplicate'``.
    mae_threshold : float | None
        If provided (and >= 0), frames whose MAE to the previous frame is
        ``<= mae_threshold`` are flagged ``is_near_duplicate=True`` (subject
        to the conjunction rule with ``max_abs_threshold``).
    max_abs_threshold : float | None
        Like ``mae_threshold`` but using max abs diff.

    Returns
    -------
    pd.DataFrame
        One row per input frame, columns in order ``REPORT_COLUMNS``.
        For frame 0: ``prev_frame_index = -1``, MAE / max_abs are NaN,
        ``is_exact_duplicate / is_near_duplicate`` are False, action is ``keep``.
    """
    if temperature.ndim != 3:
        raise ValueError(
            f"temperature must be 3D (T, H, W), got shape={temperature.shape}"
        )
    if mae_threshold is not None and mae_threshold < 0:
        raise ValueError(f"mae_threshold must be >= 0, got {mae_threshold}")
    if max_abs_threshold is not None and max_abs_threshold < 0:
        raise ValueError(f"max_abs_threshold must be >= 0, got {max_abs_threshold}")

    arr = np.asarray(temperature)
    T = int(arr.shape[0])

    frame_index = np.arange(T, dtype=np.int64)
    prev_frame_index = np.where(frame_index > 0, frame_index - 1, -1).astype(np.int64)
    mae = np.full(T, np.nan, dtype=np.float64)
    max_abs = np.full(T, np.nan, dtype=np.float64)
    is_exact = np.zeros(T, dtype=bool)

    if T >= 2:
        # Cast to float64 for stable diff; abs_diffs has shape (T-1, H, W)
        abs_diffs = np.abs(
            arr[1:].astype(np.float64, copy=False) - arr[:-1].astype(np.float64, copy=False)
        )
        flat = abs_diffs.reshape(T - 1, -1)
        mae[1:] = flat.mean(axis=1)
        max_abs[1:] = flat.max(axis=1)
        if exact:
            # Element-wise equality per row; avoids float vs int dtype quirks
            # of using max_abs == 0 as a shortcut.
            for t in range(1, T):
                is_exact[t] = bool(np.array_equal(arr[t], arr[t - 1]))

    near_components = []
    if mae_threshold is not None:
        near_components.append(mae <= float(mae_threshold))
    if max_abs_threshold is not None:
        near_components.append(max_abs <= float(max_abs_threshold))

    if near_components:
        is_near = np.logical_and.reduce(near_components)
        # mae[0] / max_abs[0] are NaN -> NaN <= x is False, so frame 0 stays False
        is_near = is_near & ~np.isnan(mae)
    else:
        is_near = np.zeros(T, dtype=bool)

    # Exact-duplicate strictly takes priority over near-duplicate label.
    if exact:
        is_near = is_near & ~is_exact

    action = np.full(T, ACTION_KEEP, dtype=object)
    if exact:
        action[is_exact] = ACTION_REMOVE_EXACT
    action[is_near] = ACTION_FLAG_NEAR
    action[0] = ACTION_KEEP  # explicit safeguard for the first frame

    return pd.DataFrame(
        {
            "frame_index": frame_index,
            "prev_frame_index": prev_frame_index,
            "is_exact_duplicate": is_exact,
            "mae_to_prev": mae,
            "max_abs_diff_to_prev": max_abs,
            "is_near_duplicate": is_near,
            "action": action,
        },
        columns=list(REPORT_COLUMNS),
    )


def remove_duplicate_frames(
    temperature: np.ndarray,
    duplicate_report: pd.DataFrame,
    *,
    remove_near_duplicates: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the report to drop flagged frames; the input array is not modified.

    Parameters
    ----------
    temperature : np.ndarray
        shape (T, H, W); same array (or a copy) used to build the report.
    duplicate_report : pd.DataFrame
        Output of ``detect_duplicate_frames`` (or any DataFrame with an
        ``action`` column whose length matches ``temperature.shape[0]``).
    remove_near_duplicates : bool
        If True, frames with ``action == 'flag_near_duplicate'`` are also dropped.
        Default False -- near-duplicates are reported, not removed.

    Returns
    -------
    temperature_dedup : np.ndarray
        Copy of ``temperature`` with marked frames removed; shape (T_kept, H, W).
    keep_indices : np.ndarray (int64)
        Original frame indices retained.
    removed_indices : np.ndarray (int64)
        Original frame indices dropped.
    """
    if temperature.ndim != 3:
        raise ValueError(
            f"temperature must be 3D (T, H, W), got shape={temperature.shape}"
        )
    if len(duplicate_report) != temperature.shape[0]:
        raise ValueError(
            f"duplicate_report length {len(duplicate_report)} does not match "
            f"temperature.shape[0]={temperature.shape[0]}"
        )
    if "action" not in duplicate_report.columns:
        raise ValueError("duplicate_report must contain an 'action' column")

    actions = duplicate_report["action"].to_numpy()
    keep_mask = actions != ACTION_REMOVE_EXACT
    if remove_near_duplicates:
        keep_mask = keep_mask & (actions != ACTION_FLAG_NEAR)

    keep_indices = np.where(keep_mask)[0].astype(np.int64)
    removed_indices = np.where(~keep_mask)[0].astype(np.int64)
    temperature_dedup = temperature[keep_indices].copy()
    return temperature_dedup, keep_indices, removed_indices
