"""Merge a folder of single-frame .xtherm files into one temperature sequence.

Xiris WeldStudio can be configured to export one .xtherm per frame
(e.g. ``Image_00000.xtherm``, ``Image_00001.xtherm``, ...), each carrying
a small binary header followed by exactly one ``height x width`` raw
frame. For this project the header is 56 bytes and the frame is
``640 x 512 x uint16`` (file size 655416 bytes; payload 655360 bytes).

Responsibilities of this module
-------------------------------
- naturally sort the file list so ``frame10.xtherm`` comes after
  ``frame2.xtherm`` (alphabetic sort gets that wrong);
- per-file: delegate to :func:`src.io.xtherm_reader.read_xtherm` with
  ``expected_frames=1`` so any size mismatch is rejected immediately
  (no silent reshape);
- stack into ``[T, H, W]`` float32 degC;
- record the merged order in ``source_files`` so downstream consumers
  (dedup, processing, export) can trace each frame back to its file;
- never modify the source files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from .xtherm_reader import read_xtherm


_DIGITS_RE = re.compile(r"(\d+)")


def natural_sort_key(name: str) -> Tuple:
    """Split a filename into ``(text, int, text, int, ...)`` for natural sorting.

    Examples
    --------
    >>> natural_sort_key("frame2.xtherm")
    ('frame', 2, '.xtherm')
    >>> natural_sort_key("frame10.xtherm")
    ('frame', 10, '.xtherm')

    Sort order using this key: ``frame2.xtherm`` < ``frame10.xtherm``,
    which is what humans expect but plain string sort gets wrong.
    """
    parts = _DIGITS_RE.split(name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


@dataclass
class MergeResult:
    """Output of :func:`merge_xtherm_folder`."""
    temperature: np.ndarray         # shape (T, H, W), float32, degC
    source_files: List[str]         # posix paths in merged order
    n_files: int


def list_xtherm_files(
    input_dir: Union[str, Path],
    *,
    pattern: str = "*.xtherm",
    recursive: bool = False,
) -> List[Path]:
    """Return a naturally-sorted list of .xtherm files in ``input_dir``."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    if recursive:
        files = list(input_dir.rglob(pattern))
    else:
        files = list(input_dir.glob(pattern))
    files = [p for p in files if p.is_file()]
    files.sort(key=lambda p: natural_sort_key(p.name))
    return files


def merge_xtherm_folder(
    input_dir: Union[str, Path],
    *,
    width: int,
    height: int,
    dtype: str = "uint16",
    endian: str = "little",
    header_offset: int = 0,
    temperature_scale: float = 0.1,
    pattern: str = "*.xtherm",
    recursive: bool = False,
) -> MergeResult:
    """Read every single-frame .xtherm in ``input_dir`` and stack into [T,H,W].

    Parameters
    ----------
    input_dir : str | Path
        Folder containing .xtherm files (one frame each).
    width, height : int
        Single-frame pixel grid; must match every file.
    dtype, endian, header_offset, temperature_scale
        Same semantics as :func:`src.io.xtherm_reader.read_xtherm`.
    pattern : str
        Glob pattern; default ``*.xtherm``.
    recursive : bool
        If True, walks subdirectories with ``rglob``.

    Returns
    -------
    MergeResult
        ``temperature`` shape ``(n_files, height, width)``, dtype float32, degC.
        ``source_files`` is a list of posix paths (same form as the inputs).

    Raises
    ------
    FileNotFoundError
        If ``input_dir`` does not exist or no files match ``pattern``.
    ValueError
        If any single file does not contain exactly one frame at the
        given shape / header_offset (delegated to ``read_xtherm``).
    """
    files = list_xtherm_files(input_dir, pattern=pattern, recursive=recursive)
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} found in {Path(input_dir).resolve()} "
            f"(recursive={recursive})"
        )

    frames: List[np.ndarray] = []
    for f in files:
        cube = read_xtherm(
            f,
            width=int(width),
            height=int(height),
            dtype=str(dtype),
            endian=str(endian),
            header_offset=int(header_offset),
            temperature_scale=float(temperature_scale),
            expected_frames=1,
        )
        # cube is (1, H, W) float32 degC; extract the single frame
        frames.append(cube[0])

    temperature = np.stack(frames, axis=0).astype(np.float32, copy=False)
    source_files = [Path(f).as_posix() for f in files]
    return MergeResult(
        temperature=temperature,
        source_files=source_files,
        n_files=len(files),
    )
