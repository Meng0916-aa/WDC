"""全局日志初始化。"""
from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: str = "INFO", fmt: Optional[str] = None) -> None:
    """配置根 logger; 已存在的 handler 不会重复添加。"""
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
