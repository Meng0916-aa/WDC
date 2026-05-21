"""SQLite 连接与 schema 初始化。

设计原则:
- 5 张表 (experiments / samples / xtherm_files / processing_results /
  frame_features), 用外键约束保持引用一致性 (PRAGMA foreign_keys = ON);
- 原始 .xtherm 文件内容不入库, xtherm_files 表只保存路径与元数据;
- 数据库文件路径来自 configs/default.yaml -> paths.database, 由 init_database.py
  脚本在项目根目录下首次建立。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union


SCHEMA_SQL: str = """
-- 实验级别元数据 (CLAUDE.md 表 1.1)
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    powder_material TEXT NOT NULL,
    substrate_material TEXT NOT NULL,
    laser_power_W REAL NOT NULL,
    scan_speed_mm_per_min REAL NOT NULL,
    powder_feed_rate_g_per_min REAL NOT NULL,
    hatch_spacing_mm REAL NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 单个试样 (一个磁场强度档位 / 一组扫描参数对应一个 sample)
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    sample_id TEXT NOT NULL UNIQUE,
    B_mT REAL NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

-- .xtherm 文件登记表 (只存路径与元数据, 不存内容)
CREATE TABLE IF NOT EXISTS xtherm_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id INTEGER NOT NULL,
    file_path TEXT NOT NULL UNIQUE,        -- 相对项目根的相对路径
    file_size_bytes INTEGER NOT NULL,
    file_sha256 TEXT,                      -- 可选: 完整性校验
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    dtype TEXT NOT NULL,                   -- 'uint16' 等
    endian TEXT NOT NULL,                  -- 'little' | 'big'
    header_offset INTEGER NOT NULL,
    temperature_scale REAL NOT NULL,
    estimated_frames INTEGER,
    fps REAL,
    dx_mm_per_pixel REAL,
    dy_mm_per_pixel REAL,
    status TEXT NOT NULL DEFAULT 'registered',
    -- 状态机: registered -> probed -> processed | error
    notes TEXT,
    registered_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_status_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (sample_id) REFERENCES samples(id) ON DELETE CASCADE
);

-- 单文件级处理结果汇总
CREATE TABLE IF NOT EXISTS processing_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    xtherm_file_id INTEGER NOT NULL UNIQUE,
    status TEXT NOT NULL,                  -- 'success' | 'error'
    error_message TEXT,
    n_frames INTEGER,
    Tmax_global REAL,
    Tmean_global REAL,
    Gmax_global REAL,
    Gmean_global REAL,
    feature_csv_path TEXT,
    processed_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (xtherm_file_id) REFERENCES xtherm_files(id) ON DELETE CASCADE
);

-- 逐帧特征
CREATE TABLE IF NOT EXISTS frame_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    xtherm_file_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    time_s REAL,
    Tmax REAL NOT NULL,
    Tmean REAL NOT NULL,
    Tstd REAL NOT NULL,
    Gmax REAL NOT NULL,
    Gmean REAL NOT NULL,
    Gstd REAL NOT NULL,
    high_temp_area INTEGER NOT NULL,
    UNIQUE(xtherm_file_id, frame_index),
    FOREIGN KEY (xtherm_file_id) REFERENCES xtherm_files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_xtherm_files_status ON xtherm_files(status);
CREATE INDEX IF NOT EXISTS idx_xtherm_files_sample ON xtherm_files(sample_id);
CREATE INDEX IF NOT EXISTS idx_frame_features_file ON frame_features(xtherm_file_id);
"""


def open_db(db_path: Union[str, Path]) -> sqlite3.Connection:
    """打开 (或创建) SQLite 连接, 打开外键约束。

    使用 detect_types 让 TEXT 时间戳可被识别; sqlite3.Row 让查询结果支持
    按列名取值, 便于在 pipeline / scripts 中处理。
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        detect_types=sqlite3.PARSE_DECLTYPES,
        isolation_level=None,  # 显式 BEGIN/COMMIT 由调用方控制
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """在传入连接上执行 schema DDL (幂等, CREATE IF NOT EXISTS)。"""
    conn.executescript(SCHEMA_SQL)
