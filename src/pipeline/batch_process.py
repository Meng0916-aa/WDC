"""批量处理 xtherm_files 表中的若干记录。"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from ..db.registry import list_files_by_status
from ..utils.config import AppConfig
from .process_run import ProcessOutcome, process_registered_file


logger = logging.getLogger(__name__)


@dataclass
class BatchReport:
    total: int
    succeeded: int
    failed: int
    outcomes: List[ProcessOutcome]


def batch_process(
    conn: sqlite3.Connection,
    cfg: AppConfig,
    *,
    status: Optional[str] = "registered",
    file_ids: Optional[List[int]] = None,
    stop_on_error: bool = False,
) -> BatchReport:
    """根据 status 或显式 file_ids 列表批量处理。

    Parameters
    ----------
    status : str | None
        若 file_ids 为 None, 用 status 过滤 (默认 'registered'); 给 None 表示处理全部。
    file_ids : list[int] | None
        若提供则忽略 status, 精确处理指定 id。
    stop_on_error : bool
        True: 第一次失败即返回; False (默认): 记录失败后继续。
    """
    if file_ids is not None:
        rows = []
        for fid in file_ids:
            row = conn.execute(
                "SELECT * FROM xtherm_files WHERE id = ?", (int(fid),)
            ).fetchone()
            if row is None:
                logger.warning("file_id=%d not found, skipped", fid)
                continue
            rows.append(row)
    else:
        rows = list_files_by_status(conn, status=status)

    outcomes: List[ProcessOutcome] = []
    succ = 0
    fail = 0
    logger.info("Batch processing %d files (status=%s)", len(rows), status)
    for row in rows:
        out = process_registered_file(conn, int(row["id"]), cfg)
        outcomes.append(out)
        if out.ok:
            succ += 1
        else:
            fail += 1
            if stop_on_error:
                logger.error("stop_on_error=True, aborting after file_id=%d", row["id"])
                break

    return BatchReport(total=len(rows), succeeded=succ, failed=fail, outcomes=outcomes)
