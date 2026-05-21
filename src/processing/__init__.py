from .gradients import compute_gradients, compute_gradients_stack
from .features import extract_frame_features, FRAME_FEATURE_COLUMNS
from .duplicate_frames import (
    detect_duplicate_frames,
    remove_duplicate_frames,
    REPORT_COLUMNS,
    ACTION_KEEP,
    ACTION_REMOVE_EXACT,
    ACTION_FLAG_NEAR,
)

__all__ = [
    "compute_gradients",
    "compute_gradients_stack",
    "extract_frame_features",
    "FRAME_FEATURE_COLUMNS",
    "detect_duplicate_frames",
    "remove_duplicate_frames",
    "REPORT_COLUMNS",
    "ACTION_KEEP",
    "ACTION_REMOVE_EXACT",
    "ACTION_FLAG_NEAR",
]
