"""项目根目录与相对路径解析。

所有数据库、配置、数据文件的路径都相对于 PROJECT_ROOT 解析,
避免在代码中散落硬编码的 D:\\GEJ-WDC 字面值。
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

# src/utils/paths.py -> src/utils -> src -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def resolve_under_root(p: Union[str, Path]) -> Path:
    """把相对路径解析为相对于项目根目录的绝对路径; 绝对路径原样返回。

    Parameters
    ----------
    p : str | Path
        相对路径 (相对 PROJECT_ROOT) 或绝对路径。

    Returns
    -------
    Path
        绝对路径 (未做存在性检查)。
    """
    path = Path(p)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()
