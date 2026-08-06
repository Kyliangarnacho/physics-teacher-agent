"""Stage 10 SQLite 数据库连接与初始化入口。

约定：
- 仅使用 Python 标准库 ``sqlite3``，不引入 ORM 或第三方数据库依赖；
- 短生命周期连接：每次 ``connect_database`` 都返回一个新的连接，由调用方负责关闭，
  模块内不保存全局长期连接；
- 不保存原始图片、Base64、Data URL 或 API Key。
"""

from __future__ import annotations

import os
import sqlite3
from os import PathLike
from typing import Union

from src.config import get_database_path
from src.storage.migrations import migrate_database


DatabasePath = Union[str, PathLike[str]]


def connect_database(path: DatabasePath | None = None) -> sqlite3.Connection:
    """打开一个新的 SQLite 连接并应用项目约定。

    自动创建父目录，并依次设置：
    - ``row_factory = sqlite3.Row``
    - ``PRAGMA foreign_keys = ON``
    - ``PRAGMA journal_mode = WAL``
    - ``PRAGMA busy_timeout = 5000``
    """
    db_path = _resolve_path(path)
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def initialize_database(path: DatabasePath | None = None) -> str:
    """将数据库迁移到当前 schema 版本并返回实际使用的路径。

    使用短生命周期连接，迁移完成后立即关闭，不保留长期连接。
    """
    db_path = _resolve_path(path)
    conn = connect_database(db_path)
    try:
        migrate_database(conn)
    finally:
        conn.close()
    return db_path


def _resolve_path(path: DatabasePath | None) -> str:
    if path is None:
        return get_database_path()
    return os.fspath(path)
