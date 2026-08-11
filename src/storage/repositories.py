"""SQLite Repository 数据访问层。

约定：
- SQL 只集中在本文件，业务文件不直接接触 SQLite；
- 每次调用使用短生命周期连接并显式提交/回滚，不保存全局连接；
- 输入输出均为 Python dict/list；JSON 字段（*_json / image_metadata_json）
  在内部自动序列化（``ensure_ascii=False``）与反序列化；
- 新记录默认生成 UUID（``uuid4().hex``），时间使用带时区的 UTC ISO 8601 字符串；
- 数据库异常统一包装为 RepositoryError，不暴露 SQL 语句或数据库绝对路径。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from src.context.schemas import ConversationSummary
from src.storage.schemas import (
    GenerationJob,
    GenerationJobPayload,
    GenerationJobStatus,
)
from src.storage.database import DatabasePath, connect_database


_CONVERSATION_COLUMNS = (
    "id",
    "title",
    "created_at",
    "updated_at",
    "archived",
)
_MESSAGE_COLUMNS = (
    "id",
    "conversation_id",
    "role",
    "display_content",
    "model_content",
    "image_metadata_json",
    "created_at",
)
_AGENT_RUN_COLUMNS = (
    "run_id",
    "conversation_id",
    "user_message_id",
    "assistant_message_id",
    "status",
    "teaching_mode",
    "use_rag",
    "use_tools",
    "total_model_requests",
    "total_duration_ms",
    "sources_json",
    "tool_records_json",
    "trace_json",
    "analysis_json",
    "created_at",
)
_CONVERSATION_STATE_COLUMNS = (
    "conversation_id",
    "active_problem_text",
    "active_image_context",
    "teaching_mode",
    "hint_step",
    "updated_at",
)
_MEMORY_COLUMNS = (
    "id",
    "memory_type",
    "topic",
    "content",
    "normalized_content",
    "evidence_count",
    "confidence",
    "source_conversation_id",
    "source_message_id",
    "confirmed",
    "active",
    "created_at",
    "updated_at",
)
_GENERATION_JOB_COLUMNS = (
    "id",
    "conversation_id",
    "user_message_id",
    "status",
    "payload_json",
    "attempts",
    "error_type",
    "error_message",
    "created_at",
    "updated_at",
    "started_at",
    "finished_at",
)
_GENERATION_JOB_SELECT = ", ".join(_GENERATION_JOB_COLUMNS)
_CONVERSATION_SUMMARY_COLUMNS = (
    "conversation_id",
    "summary_text",
    "covered_until_message_id",
    "covered_turn_count",
    "summary_revision",
    "model_name",
    "created_at",
    "updated_at",
)
_CONVERSATION_SUMMARY_SELECT = ", ".join(_CONVERSATION_SUMMARY_COLUMNS)
_STATE_OPTIONAL_FIELDS = (
    "active_problem_text",
    "active_image_context",
    "teaching_mode",
    "hint_step",
)


class StorageError(Exception):
    """存储层统一异常，不暴露 SQL 语句或数据库绝对路径。"""


class RepositoryError(StorageError):
    """数据访问层操作失败。"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid4().hex


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError("JSON 字段必须是 dict、list 或 JSON 字符串。")


def _json_loads(value: str | None) -> Any:
    if value is None or value == "":
        return None
    return json.loads(value)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _message_from_row(row: sqlite3.Row) -> dict[str, Any]:
    data = _row_to_dict(row)
    data["image_metadata_json"] = _json_loads(data["image_metadata_json"])
    return data


def _agent_run_from_row(row: sqlite3.Row) -> dict[str, Any]:
    data = _row_to_dict(row)
    for name in (
        "sources_json",
        "tool_records_json",
        "trace_json",
        "analysis_json",
    ):
        data[name] = _json_loads(data[name])
    return data


def _generation_job_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = _row_to_dict(row)
    payload = GenerationJobPayload.model_validate(
        _json_loads(data.pop("payload_json"))
    )
    data["status"] = GenerationJobStatus(data["status"])
    data["payload"] = payload
    return GenerationJob.model_validate(data).model_dump(mode="json")


def _conversation_summary_from_row(
    row: sqlite3.Row | None,
) -> dict[str, Any] | None:
    if row is None:
        return None
    return ConversationSummary.model_validate(
        _row_to_dict(row)
    ).model_dump(mode="json")


def _validate_summary_upsert_input(
    conversation_id: object,
    summary_text: object,
    covered_until_message_id: object,
    covered_turn_count: object,
    model_name: object,
) -> ConversationSummary:
    """用公开数据合同校验 upsert 输入；时间与 revision 由 Repository 管理。"""
    return ConversationSummary.model_validate(
        {
            "conversation_id": conversation_id,
            "summary_text": summary_text,
            "covered_until_message_id": covered_until_message_id,
            "covered_turn_count": covered_turn_count,
            "summary_revision": 1,
            "model_name": model_name,
            "created_at": "repository-managed",
            "updated_at": "repository-managed",
        }
    )


def _generation_job_status_value(
    status: GenerationJobStatus | str,
) -> str:
    try:
        return GenerationJobStatus(status).value
    except (TypeError, ValueError) as exc:
        raise RepositoryError("Generation Job 状态无效。") from exc


@contextmanager
def _connection(path: DatabasePath | None) -> Iterator[sqlite3.Connection]:
    try:
        conn = connect_database(path)
    except OSError as exc:
        raise RepositoryError("数据库连接失败：无法访问数据库文件。") from exc
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def create_conversation(
    title: str = "新对话",
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """创建会话并返回完整记录；默认生成 UUID 与 UTC 时间。"""
    if not isinstance(title, str):
        raise RepositoryError("会话标题必须是字符串。")
    conversation_id = _new_id()
    now = _utc_now_iso()
    try:
        with _connection(path) as conn:
            conn.execute(
                "INSERT INTO conversations "
                "(id, title, created_at, updated_at, archived) "
                "VALUES (?, ?, ?, ?, 0)",
                (conversation_id, title, now, now),
            )
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话创建失败：数据库错误。") from exc
    return {
        "id": conversation_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "archived": 0,
    }


def list_conversations(
    *,
    include_archived: bool = False,
    path: DatabasePath | None = None,
) -> list[dict[str, Any]]:
    """按 updated_at 倒序列出会话；默认排除已归档会话。"""
    query = (
        "SELECT id, title, created_at, updated_at, archived "
        "FROM conversations"
    )
    if not include_archived:
        query += " WHERE archived = 0"
    query += " ORDER BY updated_at DESC, created_at DESC, rowid DESC"
    try:
        with _connection(path) as conn:
            rows = conn.execute(query).fetchall()
            return [_row_to_dict(row) for row in rows]
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话列表读取失败：数据库错误。") from exc


def get_conversation(
    conversation_id: str,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any] | None:
    """按 ID 读取单个会话；不存在时返回 None。"""
    try:
        with _connection(path) as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at, archived "
                "FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            return _row_to_dict(row)
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话读取失败：数据库错误。") from exc


def rename_conversation(
    conversation_id: str,
    title: str,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """重命名会话并刷新 updated_at；会话不存在时抛出 RepositoryError。"""
    if not isinstance(title, str):
        raise RepositoryError("会话标题必须是字符串。")
    now = _utc_now_iso()
    try:
        with _connection(path) as conn:
            cursor = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? "
                "WHERE id = ?",
                (title, now, conversation_id),
            )
            if cursor.rowcount == 0:
                raise RepositoryError("会话不存在，无法重命名。")
            row = conn.execute(
                "SELECT id, title, created_at, updated_at, archived "
                "FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            return _row_to_dict(row)
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话重命名失败：数据库错误。") from exc


def delete_conversation(
    conversation_id: str,
    *,
    path: DatabasePath | None = None,
) -> bool:
    """删除会话；消息、运行、状态和摘要由外键级联删除。

    learning_memories 不受影响。返回是否删除成功。
    """
    try:
        with _connection(path) as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            return cursor.rowcount > 0
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话删除失败：数据库错误。") from exc


def get_conversation_summary(
    conversation_id: str,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any] | None:
    """读取会话摘要；不存在时返回 None。"""
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise RepositoryError("conversation_id 必须为非空字符串。")
    try:
        with _connection(path) as conn:
            row = conn.execute(
                f"SELECT {_CONVERSATION_SUMMARY_SELECT} "
                "FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id.strip(),),
            ).fetchone()
            return _conversation_summary_from_row(row)
    except (ValidationError, ValueError) as exc:
        raise RepositoryError("会话摘要记录损坏。") from exc
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话摘要读取失败：数据库错误。") from exc


def upsert_conversation_summary(
    conversation_id: str,
    summary_text: str,
    *,
    covered_until_message_id: str | None = None,
    covered_turn_count: int = 0,
    model_name: str | None = None,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """新增或更新会话摘要，并由数据库原子维护 revision。"""
    try:
        summary = _validate_summary_upsert_input(
            conversation_id,
            summary_text,
            covered_until_message_id,
            covered_turn_count,
            model_name,
        )
    except ValidationError as exc:
        raise RepositoryError("会话摘要参数不合法。") from exc

    now = _utc_now_iso()
    try:
        with _connection(path) as conn:
            conn.execute(
                "INSERT INTO conversation_summaries "
                "(conversation_id, summary_text, covered_until_message_id, "
                "covered_turn_count, summary_revision, model_name, created_at, "
                "updated_at) VALUES (?, ?, ?, ?, 1, ?, ?, ?) "
                "ON CONFLICT(conversation_id) DO UPDATE SET "
                "summary_text = excluded.summary_text, "
                "covered_until_message_id = excluded.covered_until_message_id, "
                "covered_turn_count = excluded.covered_turn_count, "
                "summary_revision = conversation_summaries.summary_revision + 1, "
                "model_name = excluded.model_name, "
                "updated_at = excluded.updated_at",
                (
                    summary.conversation_id,
                    summary.summary_text,
                    summary.covered_until_message_id,
                    summary.covered_turn_count,
                    summary.model_name,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                f"SELECT {_CONVERSATION_SUMMARY_SELECT} "
                "FROM conversation_summaries WHERE conversation_id = ?",
                (summary.conversation_id,),
            ).fetchone()
            result = _conversation_summary_from_row(row)
            if result is None:
                raise RepositoryError("会话摘要保存后未找到记录。")
            return result
    except RepositoryError:
        raise
    except sqlite3.IntegrityError as exc:
        raise RepositoryError("会话摘要保存失败：会话不存在。") from exc
    except (ValidationError, ValueError) as exc:
        raise RepositoryError("会话摘要保存后记录不合法。") from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话摘要保存失败：数据库错误。") from exc


def delete_conversation_summary(
    conversation_id: str,
    *,
    path: DatabasePath | None = None,
) -> bool:
    """删除一个会话摘要；返回是否删除成功。"""
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise RepositoryError("conversation_id 必须为非空字符串。")
    try:
        with _connection(path) as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id.strip(),),
            )
            return cursor.rowcount > 0
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话摘要删除失败：数据库错误。") from exc


def _prepare_message(
    message: dict[str, Any],
    now: str,
) -> dict[str, Any]:
    """校验并规范化一条消息的插入值；不打开连接。"""
    conversation_id = message.get("conversation_id")
    role = message.get("role")
    display_content = message.get("display_content")
    model_content = message.get("model_content")
    if (
        not conversation_id
        or role is None
        or display_content is None
        or model_content is None
    ):
        raise ValueError(
            "消息缺少必要字段：conversation_id、role、display_content、model_content。"
        )
    try:
        image_metadata_json = _json_dumps(message.get("image_metadata_json"))
    except ValueError as exc:
        raise ValueError("消息图片元数据必须是 dict、list 或 JSON 字符串。") from exc
    return {
        "message_id": message.get("id") or _new_id(),
        "conversation_id": conversation_id,
        "role": role,
        "display_content": display_content,
        "model_content": model_content,
        "image_metadata_json": image_metadata_json,
        "created_at": message.get("created_at") or now,
    }


def _insert_message_on_conn(
    conn: sqlite3.Connection,
    prepared: dict[str, Any],
) -> None:
    conn.execute(
        "INSERT INTO messages "
        "(id, conversation_id, role, display_content, model_content, "
        "image_metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            prepared["message_id"],
            prepared["conversation_id"],
            prepared["role"],
            prepared["display_content"],
            prepared["model_content"],
            prepared["image_metadata_json"],
            prepared["created_at"],
        ),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (prepared["created_at"], prepared["conversation_id"]),
    )


def _prepare_agent_run(
    run: dict[str, Any],
    now: str,
    *,
    assistant_message_id: str | None = None,
) -> dict[str, Any]:
    """校验并规范化一次 AgentRun 的插入值；不打开连接。"""
    conversation_id = run.get("conversation_id")
    user_message_id = run.get("user_message_id")
    status = run.get("status")
    if not conversation_id or not user_message_id or not status:
        raise ValueError(
            "AgentRun 缺少必要字段：conversation_id、user_message_id、status。"
        )
    try:
        sources_json = _json_dumps(run.get("sources_json"))
        tool_records_json = _json_dumps(run.get("tool_records_json"))
        trace_json = _json_dumps(run.get("trace_json"))
        analysis_json = _json_dumps(run.get("analysis_json"))
    except ValueError as exc:
        raise ValueError(
            "AgentRun 的 JSON 字段必须是 dict、list 或 JSON 字符串。"
        ) from exc
    resolved_assistant_message_id = (
        assistant_message_id
        if assistant_message_id is not None
        else run.get("assistant_message_id")
    )
    return {
        "run_id": run.get("run_id") or _new_id(),
        "conversation_id": conversation_id,
        "user_message_id": user_message_id,
        "assistant_message_id": resolved_assistant_message_id,
        "status": status,
        "teaching_mode": run.get("teaching_mode"),
        "use_rag": int(bool(run.get("use_rag", False))),
        "use_tools": int(bool(run.get("use_tools", False))),
        "total_model_requests": int(run.get("total_model_requests", 0)),
        "total_duration_ms": float(run.get("total_duration_ms", 0.0)),
        "sources_json": sources_json,
        "tool_records_json": tool_records_json,
        "trace_json": trace_json,
        "analysis_json": analysis_json,
        "created_at": run.get("created_at") or now,
    }


def _insert_agent_run_on_conn(
    conn: sqlite3.Connection,
    prepared: dict[str, Any],
) -> None:
    conn.execute(
        "INSERT INTO agent_runs "
        "(run_id, conversation_id, user_message_id, assistant_message_id, "
        "status, teaching_mode, use_rag, use_tools, total_model_requests, "
        "total_duration_ms, sources_json, tool_records_json, trace_json, "
        "analysis_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            prepared["run_id"],
            prepared["conversation_id"],
            prepared["user_message_id"],
            prepared["assistant_message_id"],
            prepared["status"],
            prepared["teaching_mode"],
            prepared["use_rag"],
            prepared["use_tools"],
            prepared["total_model_requests"],
            prepared["total_duration_ms"],
            prepared["sources_json"],
            prepared["tool_records_json"],
            prepared["trace_json"],
            prepared["analysis_json"],
            prepared["created_at"],
        ),
    )


def insert_message(
    message: dict[str, Any],
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """插入一条消息并刷新所属会话的 updated_at；返回完整存储记录。"""
    now = message.get("created_at") or _utc_now_iso()
    try:
        prepared = _prepare_message(message, now)
    except ValueError as exc:
        raise RepositoryError(str(exc)) from exc
    try:
        with _connection(path) as conn:
            _insert_message_on_conn(conn, prepared)
    except RepositoryError:
        raise
    except sqlite3.IntegrityError as exc:
        raise RepositoryError("消息保存失败：会话不存在或字段不合法。") from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("消息保存失败：数据库错误。") from exc
    return {
        "id": prepared["message_id"],
        "conversation_id": prepared["conversation_id"],
        "role": prepared["role"],
        "display_content": prepared["display_content"],
        "model_content": prepared["model_content"],
        "image_metadata_json": message.get("image_metadata_json"),
        "created_at": prepared["created_at"],
    }


def list_messages(
    conversation_id: str,
    *,
    path: DatabasePath | None = None,
) -> list[dict[str, Any]]:
    """按时间正序列出会话内全部消息，JSON 字段自动反序列化。"""
    try:
        with _connection(path) as conn:
            rows = conn.execute(
                "SELECT id, conversation_id, role, display_content, model_content, "
                "image_metadata_json, created_at FROM messages "
                "WHERE conversation_id = ? "
                "ORDER BY created_at ASC, rowid ASC",
                (conversation_id,),
            ).fetchall()
            return [_message_from_row(row) for row in rows]
    except RepositoryError:
        raise
    except ValueError as exc:
        raise RepositoryError("消息记录包含无法解析的 JSON。") from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("消息列表读取失败：数据库错误。") from exc


def get_recent_messages(
    conversation_id: str,
    limit: int = 10,
    *,
    path: DatabasePath | None = None,
) -> list[dict[str, Any]]:
    """返回最近 limit 条消息，返回顺序仍为从旧到新。"""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise RepositoryError("limit 必须是正整数。")
    try:
        with _connection(path) as conn:
            rows = conn.execute(
                "SELECT id, conversation_id, role, display_content, model_content, "
                "image_metadata_json, created_at FROM ("
                "  SELECT id, conversation_id, role, display_content, model_content, "
                "  image_metadata_json, created_at, rowid AS _rowid FROM messages "
                "  WHERE conversation_id = ? "
                "  ORDER BY created_at DESC, rowid DESC LIMIT ?"
                ") ORDER BY created_at ASC, _rowid ASC",
                (conversation_id, limit),
            ).fetchall()
            return [_message_from_row(row) for row in rows]
    except RepositoryError:
        raise
    except ValueError as exc:
        raise RepositoryError("消息记录包含无法解析的 JSON。") from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("最近消息读取失败：数据库错误。") from exc


def insert_agent_run(
    run: dict[str, Any],
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """插入一次 AgentRun；JSON 字段自动序列化，返回完整存储记录。"""
    now = run.get("created_at") or _utc_now_iso()
    try:
        prepared = _prepare_agent_run(run, now)
    except ValueError as exc:
        raise RepositoryError(str(exc)) from exc
    try:
        with _connection(path) as conn:
            _insert_agent_run_on_conn(conn, prepared)
    except RepositoryError:
        raise
    except sqlite3.IntegrityError as exc:
        raise RepositoryError("AgentRun 保存失败：会话不存在或字段不合法。") from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("AgentRun 保存失败：数据库错误。") from exc
    return {
        "run_id": prepared["run_id"],
        "conversation_id": prepared["conversation_id"],
        "user_message_id": prepared["user_message_id"],
        "assistant_message_id": prepared["assistant_message_id"],
        "status": prepared["status"],
        "teaching_mode": prepared["teaching_mode"],
        "use_rag": prepared["use_rag"],
        "use_tools": prepared["use_tools"],
        "total_model_requests": prepared["total_model_requests"],
        "total_duration_ms": prepared["total_duration_ms"],
        "sources_json": run.get("sources_json"),
        "tool_records_json": run.get("tool_records_json"),
        "trace_json": run.get("trace_json"),
        "analysis_json": run.get("analysis_json"),
        "created_at": prepared["created_at"],
    }


def get_agent_runs(
    conversation_id: str,
    *,
    path: DatabasePath | None = None,
) -> list[dict[str, Any]]:
    """按时间正序列出会话内 AgentRun，JSON 字段自动反序列化。"""
    try:
        with _connection(path) as conn:
            rows = conn.execute(
                "SELECT run_id, conversation_id, user_message_id, assistant_message_id, "
                "status, teaching_mode, use_rag, use_tools, total_model_requests, "
                "total_duration_ms, sources_json, tool_records_json, trace_json, "
                "analysis_json, created_at FROM agent_runs "
                "WHERE conversation_id = ? "
                "ORDER BY created_at ASC, rowid ASC",
                (conversation_id,),
            ).fetchall()
            return [_agent_run_from_row(row) for row in rows]
    except RepositoryError:
        raise
    except ValueError as exc:
        raise RepositoryError("AgentRun 记录包含无法解析的 JSON。") from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("AgentRun 列表读取失败：数据库错误。") from exc


def create_generation_job(
    job: dict[str, Any],
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """创建 pending 后台回答 Job；同一会话只能存在一个活动 Job。"""
    if not isinstance(job, dict):
        raise RepositoryError("Generation Job 必须是字典。")
    conversation_id = job.get("conversation_id")
    user_message_id = job.get("user_message_id")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise RepositoryError("Generation Job 缺少 conversation_id。")
    if not isinstance(user_message_id, str) or not user_message_id.strip():
        raise RepositoryError("Generation Job 缺少 user_message_id。")
    try:
        payload = GenerationJobPayload.model_validate(job.get("payload"))
    except ValidationError as exc:
        raise RepositoryError("Generation Job payload 不合法。") from exc

    job_id = job.get("id") or _new_id()
    if not isinstance(job_id, str) or not job_id.strip():
        raise RepositoryError("Generation Job id 必须为非空字符串。")
    created_at = job.get("created_at") or _utc_now_iso()
    if not isinstance(created_at, str) or not created_at.strip():
        raise RepositoryError("Generation Job created_at 必须为非空字符串。")
    payload_json = json.dumps(payload.model_dump(mode="json"), ensure_ascii=False)
    try:
        with _connection(path) as conn:
            conn.execute(
                "INSERT INTO generation_jobs "
                "(id, conversation_id, user_message_id, status, payload_json, "
                "attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, 'pending', ?, 0, ?, ?)",
                (
                    job_id,
                    conversation_id,
                    user_message_id,
                    payload_json,
                    created_at,
                    created_at,
                ),
            )
            row = conn.execute(
                f"SELECT {_GENERATION_JOB_SELECT} FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            return _generation_job_from_row(row)
    except RepositoryError:
        raise
    except sqlite3.IntegrityError as exc:
        raise RepositoryError(
            "Generation Job 创建失败：会话、用户消息不存在或会话已有活动 Job。"
        ) from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("Generation Job 创建失败：数据库错误。") from exc


def enqueue_generation_job(
    user_message: dict[str, Any],
    payload: dict[str, Any],
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """在同一事务中保存 user 消息和对应的 pending Generation Job。"""
    now = _utc_now_iso()
    try:
        prepared_message = _prepare_message(user_message, now)
        if prepared_message["role"] != "user":
            raise ValueError("Generation Job 只能关联 user 消息。")
        prepared_payload = GenerationJobPayload.model_validate(payload)
    except (ValueError, ValidationError) as exc:
        raise RepositoryError("Generation Job 入队参数不合法。") from exc

    job_id = _new_id()
    payload_json = json.dumps(
        prepared_payload.model_dump(mode="json"),
        ensure_ascii=False,
    )
    try:
        with _connection(path) as conn:
            _insert_message_on_conn(conn, prepared_message)
            conn.execute(
                "INSERT INTO generation_jobs "
                "(id, conversation_id, user_message_id, status, payload_json, "
                "attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, 'pending', ?, 0, ?, ?)",
                (
                    job_id,
                    prepared_message["conversation_id"],
                    prepared_message["message_id"],
                    payload_json,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                f"SELECT {_GENERATION_JOB_SELECT} FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            job = _generation_job_from_row(row)
    except RepositoryError:
        raise
    except sqlite3.IntegrityError as exc:
        raise RepositoryError(
            "Generation Job 入队失败：会话不存在或会话已有活动 Job。"
        ) from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("Generation Job 入队失败：数据库错误。") from exc

    return {
        "user_message": {
            "id": prepared_message["message_id"],
            "conversation_id": prepared_message["conversation_id"],
            "role": prepared_message["role"],
            "display_content": prepared_message["display_content"],
            "model_content": prepared_message["model_content"],
            "image_metadata_json": user_message.get("image_metadata_json"),
            "created_at": prepared_message["created_at"],
        },
        "generation_job": job,
    }


def get_generation_job(
    job_id: str,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any] | None:
    """按 ID 读取后台回答 Job；不存在时返回 None。"""
    try:
        with _connection(path) as conn:
            row = conn.execute(
                f"SELECT {_GENERATION_JOB_SELECT} FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            return _generation_job_from_row(row)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise RepositoryError("Generation Job 记录损坏。") from exc
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("Generation Job 读取失败：数据库错误。") from exc


def list_generation_jobs(
    conversation_id: str | None = None,
    *,
    status: GenerationJobStatus | str | None = None,
    path: DatabasePath | None = None,
) -> list[dict[str, Any]]:
    """按创建顺序列出 Job，可按会话和状态过滤。"""
    clauses: list[str] = []
    params: list[Any] = []
    if conversation_id is not None:
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise RepositoryError("conversation_id 必须为非空字符串。")
        clauses.append("conversation_id = ?")
        params.append(conversation_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(_generation_job_status_value(status))
    query = f"SELECT {_GENERATION_JOB_SELECT} FROM generation_jobs"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at ASC, rowid ASC"
    try:
        with _connection(path) as conn:
            rows = conn.execute(query, params).fetchall()
            return [_generation_job_from_row(row) for row in rows]
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise RepositoryError("Generation Job 列表包含损坏记录。") from exc
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("Generation Job 列表读取失败：数据库错误。") from exc


def get_active_generation_job(
    conversation_id: str,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any] | None:
    """读取会话唯一的 pending/running Job。"""
    try:
        with _connection(path) as conn:
            row = conn.execute(
                f"SELECT {_GENERATION_JOB_SELECT} FROM generation_jobs "
                "WHERE conversation_id = ? AND status IN ('pending', 'running') "
                "ORDER BY created_at ASC, rowid ASC LIMIT 1",
                (conversation_id,),
            ).fetchone()
            return _generation_job_from_row(row)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise RepositoryError("活动 Generation Job 记录损坏。") from exc
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("活动 Generation Job 读取失败：数据库错误。") from exc


def list_pending_generation_jobs(
    limit: int = 100,
    *,
    path: DatabasePath | None = None,
) -> list[dict[str, Any]]:
    """按创建顺序列出待领取 Job。"""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise RepositoryError("limit 必须是正整数。")
    try:
        with _connection(path) as conn:
            rows = conn.execute(
                f"SELECT {_GENERATION_JOB_SELECT} FROM generation_jobs "
                "WHERE status = 'pending' "
                "ORDER BY created_at ASC, rowid ASC LIMIT ?",
                (limit,),
            ).fetchall()
            return [_generation_job_from_row(row) for row in rows]
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise RepositoryError("待处理 Generation Job 列表包含损坏记录。") from exc
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("待处理 Generation Job 读取失败：数据库错误。") from exc


def claim_generation_job(
    job_id: str,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any] | None:
    """原子执行 pending → running；已被领取时返回 None。"""
    now = _utc_now_iso()
    try:
        with _connection(path) as conn:
            cursor = conn.execute(
                "UPDATE generation_jobs SET status = 'running', "
                "attempts = attempts + 1, started_at = ?, finished_at = NULL, "
                "error_type = NULL, error_message = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'pending'",
                (now, now, job_id),
            )
            if cursor.rowcount == 0:
                return None
            row = conn.execute(
                f"SELECT {_GENERATION_JOB_SELECT} FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            return _generation_job_from_row(row)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise RepositoryError("领取后的 Generation Job 记录损坏。") from exc
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("Generation Job 领取失败：数据库错误。") from exc


def _transition_running_job(
    job_id: str,
    target_status: GenerationJobStatus,
    *,
    error_type: str | None = None,
    error_message: str | None = None,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    now = _utc_now_iso()
    try:
        with _connection(path) as conn:
            cursor = conn.execute(
                "UPDATE generation_jobs SET status = ?, error_type = ?, "
                "error_message = ?, finished_at = ?, updated_at = ? "
                "WHERE id = ? AND status = 'running'",
                (
                    target_status.value,
                    error_type,
                    error_message,
                    now,
                    now,
                    job_id,
                ),
            )
            if cursor.rowcount == 0:
                existing = conn.execute(
                    "SELECT status FROM generation_jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
                if existing is None:
                    raise RepositoryError("Generation Job 不存在。")
                raise RepositoryError("Generation Job 不是 running，无法完成状态变更。")
            row = conn.execute(
                f"SELECT {_GENERATION_JOB_SELECT} FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            return _generation_job_from_row(row)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise RepositoryError("状态变更后的 Generation Job 记录损坏。") from exc
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("Generation Job 状态更新失败：数据库错误。") from exc


def complete_generation_job(
    job_id: str,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """执行 running → completed。"""
    return _transition_running_job(
        job_id,
        GenerationJobStatus.COMPLETED,
        path=path,
    )


def fail_generation_job(
    job_id: str,
    error_type: str,
    error_message: str,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """执行 running → failed，并保存简短安全错误摘要。"""
    if not isinstance(error_type, str) or not error_type.strip():
        raise RepositoryError("error_type 必须为非空字符串。")
    if not isinstance(error_message, str) or not error_message.strip():
        raise RepositoryError("error_message 必须为非空字符串。")
    return _transition_running_job(
        job_id,
        GenerationJobStatus.FAILED,
        error_type=error_type.strip(),
        error_message=error_message.strip(),
        path=path,
    )


def mark_running_interrupted(
    *,
    path: DatabasePath | None = None,
) -> int:
    """将进程遗留的全部 running Job 标记为 interrupted，返回数量。"""
    now = _utc_now_iso()
    try:
        with _connection(path) as conn:
            cursor = conn.execute(
                "UPDATE generation_jobs SET status = 'interrupted', "
                "error_type = 'worker_interrupted', "
                "error_message = '回答生成被服务中断。', "
                "finished_at = ?, updated_at = ? WHERE status = 'running'",
                (now, now),
            )
            return cursor.rowcount
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("Generation Job 中断标记失败：数据库错误。") from exc


def retry_generation_job(
    job_id: str,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """仅执行 failed/interrupted → pending；保留累计 attempts。"""
    now = _utc_now_iso()
    try:
        with _connection(path) as conn:
            cursor = conn.execute(
                "UPDATE generation_jobs SET status = 'pending', "
                "error_type = NULL, error_message = NULL, started_at = NULL, "
                "finished_at = NULL, updated_at = ? "
                "WHERE id = ? AND status IN ('failed', 'interrupted')",
                (now, job_id),
            )
            if cursor.rowcount == 0:
                existing = conn.execute(
                    "SELECT status FROM generation_jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
                if existing is None:
                    raise RepositoryError("Generation Job 不存在。")
                raise RepositoryError("只有 failed/interrupted Job 可以重试。")
            row = conn.execute(
                f"SELECT {_GENERATION_JOB_SELECT} FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            return _generation_job_from_row(row)
    except RepositoryError:
        raise
    except sqlite3.IntegrityError as exc:
        raise RepositoryError("Generation Job 重试失败：会话已有活动 Job。") from exc
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise RepositoryError("重试后的 Generation Job 记录损坏。") from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("Generation Job 重试失败：数据库错误。") from exc


def get_conversation_state(
    conversation_id: str,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any] | None:
    """读取会话状态；不存在时返回 None。"""
    try:
        with _connection(path) as conn:
            row = conn.execute(
                "SELECT conversation_id, active_problem_text, active_image_context, "
                "teaching_mode, hint_step, updated_at "
                "FROM conversation_states WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return _row_to_dict(row)
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话状态读取失败：数据库错误。") from exc


def _state_upsert_sql(
    state: dict[str, Any],
    now: str,
) -> tuple[str, tuple[Any, ...]]:
    """构造 conversation_states 的 upsert SQL 与参数；不打开连接。"""
    conversation_id = state.get("conversation_id")
    if not conversation_id:
        raise ValueError("会话状态缺少 conversation_id。")
    provided = [name for name in _STATE_OPTIONAL_FIELDS if name in state]
    if provided:
        columns = ", ".join(provided)
        placeholders = ", ".join("?" for _ in provided)
        updates = ", ".join(f"{name} = excluded.{name}" for name in provided)
        sql = (
            "INSERT INTO conversation_states "
            f"(conversation_id, {columns}, updated_at) VALUES (?, {placeholders}, ?) "
            f"ON CONFLICT(conversation_id) DO UPDATE SET "
            f"{updates}, updated_at = excluded.updated_at"
        )
        params = (conversation_id, *(state[name] for name in provided), now)
    else:
        sql = (
            "INSERT INTO conversation_states (conversation_id, updated_at) "
            "VALUES (?, ?) ON CONFLICT(conversation_id) DO UPDATE SET "
            "updated_at = excluded.updated_at"
        )
        params = (conversation_id, now)
    return sql, params


def upsert_conversation_state(
    state: dict[str, Any],
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """新增或按 conversation_id 冲突更新会话状态。

    只更新输入中出现的可选字段，未提供的字段保持不变；始终刷新 updated_at。
    """
    conversation_id = state.get("conversation_id")
    if not conversation_id:
        raise RepositoryError("会话状态缺少 conversation_id。")
    now = state.get("updated_at") or _utc_now_iso()
    try:
        insert_sql, params = _state_upsert_sql(state, now)
    except ValueError as exc:
        raise RepositoryError(str(exc)) from exc
    try:
        with _connection(path) as conn:
            conn.execute(insert_sql, params)
            row = conn.execute(
                "SELECT conversation_id, active_problem_text, active_image_context, "
                "teaching_mode, hint_step, updated_at "
                "FROM conversation_states WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return _row_to_dict(row)
    except RepositoryError:
        raise
    except sqlite3.IntegrityError as exc:
        raise RepositoryError("会话状态保存失败：会话不存在或字段不合法。") from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话状态保存失败：数据库错误。") from exc


def finalize_conversation_turn(
    conversation_id: str,
    assistant_message: dict[str, Any],
    agent_run: dict[str, Any],
    conversation_state: dict[str, Any] | None = None,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """在一个短事务内原子保存 assistant 消息、agent_run 与可选状态更新。

    assistant 消息、agent_run 与 conversation_state 要么全部成功，要么全部
    回滚；user 消息由调用方在事务外先保存。
    """
    now = _utc_now_iso()
    try:
        assistant = _prepare_message(assistant_message, now)
        if assistant["conversation_id"] != conversation_id:
            raise ValueError("assistant_message 的 conversation_id 与传入会话不一致。")
        run = _prepare_agent_run(
            agent_run,
            now,
            assistant_message_id=assistant["message_id"],
        )
        if run["conversation_id"] != conversation_id:
            raise ValueError("agent_run 的 conversation_id 与传入会话不一致。")
        state_sql: str | None = None
        state_params: tuple[Any, ...] | None = None
        if conversation_state is not None:
            normalized_state = dict(conversation_state)
            normalized_state["conversation_id"] = conversation_id
            state_sql, state_params = _state_upsert_sql(normalized_state, now)
    except ValueError as exc:
        raise RepositoryError(str(exc)) from exc

    try:
        with _connection(path) as conn:
            _insert_message_on_conn(conn, assistant)
            _insert_agent_run_on_conn(conn, run)
            if state_sql is not None:
                conn.execute(state_sql, state_params)
    except RepositoryError:
        raise
    except sqlite3.IntegrityError as exc:
        raise RepositoryError("会话轮次保存失败：会话不存在或字段不合法。") from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话轮次保存失败：数据库错误。") from exc

    return {
        "conversation_id": conversation_id,
        "user_message_id": run["user_message_id"],
        "assistant_message_id": assistant["message_id"],
        "agent_run_id": run["run_id"],
    }


def finalize_generation_job(
    job_id: str,
    conversation_id: str,
    assistant_message: dict[str, Any],
    agent_run: dict[str, Any],
    conversation_state: dict[str, Any] | None = None,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """原子保存成功结果，并执行 running → completed。"""
    now = _utc_now_iso()
    try:
        assistant = _prepare_message(assistant_message, now)
        if assistant["conversation_id"] != conversation_id:
            raise ValueError("assistant_message 的 conversation_id 与传入会话不一致。")
        run = _prepare_agent_run(
            agent_run,
            now,
            assistant_message_id=assistant["message_id"],
        )
        if run["conversation_id"] != conversation_id:
            raise ValueError("agent_run 的 conversation_id 与传入会话不一致。")
        state_sql: str | None = None
        state_params: tuple[Any, ...] | None = None
        if conversation_state is not None:
            normalized_state = dict(conversation_state)
            normalized_state["conversation_id"] = conversation_id
            state_sql, state_params = _state_upsert_sql(normalized_state, now)
    except ValueError as exc:
        raise RepositoryError(str(exc)) from exc

    try:
        with _connection(path) as conn:
            job_row = conn.execute(
                "SELECT conversation_id, user_message_id, status "
                "FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if job_row is None:
                raise RepositoryError("Generation Job 不存在。")
            if job_row["status"] != "running":
                raise RepositoryError("Generation Job 不是 running，无法保存结果。")
            if job_row["conversation_id"] != conversation_id:
                raise RepositoryError("Generation Job 与会话不匹配。")
            if run["user_message_id"] != job_row["user_message_id"]:
                raise RepositoryError("AgentRun 与 Generation Job 的用户消息不匹配。")

            _insert_message_on_conn(conn, assistant)
            _insert_agent_run_on_conn(conn, run)
            if state_sql is not None:
                conn.execute(state_sql, state_params)
            cursor = conn.execute(
                "UPDATE generation_jobs SET status = 'completed', "
                "error_type = NULL, error_message = NULL, finished_at = ?, "
                "updated_at = ? WHERE id = ? AND status = 'running'",
                (now, now, job_id),
            )
            if cursor.rowcount != 1:
                raise RepositoryError("Generation Job 完成状态更新失败。")
            completed_row = conn.execute(
                f"SELECT {_GENERATION_JOB_SELECT} FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            completed_job = _generation_job_from_row(completed_row)
    except RepositoryError:
        raise
    except sqlite3.IntegrityError as exc:
        raise RepositoryError("Generation Job 结果保存失败：关联数据不合法。") from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("Generation Job 结果保存失败：数据库错误。") from exc

    return {
        "conversation_id": conversation_id,
        "user_message_id": run["user_message_id"],
        "assistant_message_id": assistant["message_id"],
        "agent_run_id": run["run_id"],
        "generation_job": completed_job,
    }


def clear_conversation_contents(
    conversation_id: str,
    *,
    path: DatabasePath | None = None,
) -> bool:
    """在一个事务内清空消息、运行、状态和摘要，但保留 conversation 行。

    返回会话是否存在；不存在时不做任何删除并返回 False。
    """
    now = _utc_now_iso()
    try:
        with _connection(path) as conn:
            cursor = conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            if cursor.rowcount == 0:
                return False
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute(
                "DELETE FROM agent_runs WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute(
                "DELETE FROM conversation_states WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute(
                "DELETE FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id,),
            )
            return True
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话内容清空失败：数据库错误。") from exc


def reset_conversation_state(
    conversation_id: str,
    *,
    path: DatabasePath | None = None,
) -> bool:
    """删除会话当前状态；返回是否删除成功。"""
    try:
        with _connection(path) as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_states WHERE conversation_id = ?",
                (conversation_id,),
            )
            return cursor.rowcount > 0
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("会话状态重置失败：数据库错误。") from exc


def insert_or_merge_memory(
    memory: dict[str, Any],
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """按 memory_type + topic + normalized_content 插入或合并记忆。

    首次插入生成新 id；重复合并时保留原 id，evidence_count 加 1，并更新内容、
    置信度、来源、状态和 updated_at（created_at 保持不变）。
    """
    missing = [
        name
        for name in ("memory_type", "topic", "content", "normalized_content", "confidence")
        if name not in memory
    ]
    if missing:
        raise RepositoryError(f"记忆缺少必要字段：{', '.join(missing)}。")
    memory_id = memory.get("id") or _new_id()
    created_at = memory.get("created_at") or _utc_now_iso()
    now = memory.get("updated_at") or _utc_now_iso()
    try:
        with _connection(path) as conn:
            try:
                conn.execute(
                    "INSERT INTO learning_memories "
                    "(id, memory_type, topic, content, normalized_content, "
                    "evidence_count, confidence, source_conversation_id, "
                    "source_message_id, confirmed, active, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        memory_id,
                        memory["memory_type"],
                        memory["topic"],
                        memory["content"],
                        memory["normalized_content"],
                        int(memory.get("evidence_count", 1)),
                        float(memory["confidence"]),
                        memory.get("source_conversation_id"),
                        memory.get("source_message_id"),
                        int(memory.get("confirmed", 1)),
                        int(memory.get("active", 1)),
                        created_at,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                existing = conn.execute(
                    "SELECT id FROM learning_memories "
                    "WHERE memory_type = ? AND topic = ? AND normalized_content = ?",
                    (memory["memory_type"], memory["topic"], memory["normalized_content"]),
                ).fetchone()
                if existing is None:
                    raise
                memory_id = existing["id"]
                conn.execute(
                    "UPDATE learning_memories "
                    "SET content = ?, evidence_count = evidence_count + 1, "
                    "confidence = ?, source_conversation_id = ?, "
                    "source_message_id = ?, confirmed = ?, active = ?, "
                    "updated_at = ? WHERE id = ?",
                    (
                        memory["content"],
                        float(memory["confidence"]),
                        memory.get("source_conversation_id"),
                        memory.get("source_message_id"),
                        int(memory.get("confirmed", 1)),
                        int(memory.get("active", 1)),
                        now,
                        memory_id,
                    ),
                )
            row = conn.execute(
                "SELECT id, memory_type, topic, content, normalized_content, "
                "evidence_count, confidence, source_conversation_id, "
                "source_message_id, confirmed, active, created_at, updated_at "
                "FROM learning_memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            return _row_to_dict(row)
    except RepositoryError:
        raise
    except sqlite3.IntegrityError as exc:
        raise RepositoryError("记忆保存失败：id 冲突或数据不合法。") from exc
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("记忆保存失败：数据库错误。") from exc


def list_memories(
    *,
    memory_type: str | None = None,
    topic: str | None = None,
    include_inactive: bool = False,
    path: DatabasePath | None = None,
) -> list[dict[str, Any]]:
    """列出记忆；默认只返回 active=1，按 updated_at 倒序。"""
    conditions: list[str] = []
    params: list[Any] = []
    if memory_type is not None:
        conditions.append("memory_type = ?")
        params.append(memory_type)
    if topic is not None:
        conditions.append("topic = ?")
        params.append(topic)
    if not include_inactive:
        conditions.append("active = 1")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = (
        "SELECT id, memory_type, topic, content, normalized_content, "
        "evidence_count, confidence, source_conversation_id, source_message_id, "
        "confirmed, active, created_at, updated_at FROM learning_memories "
        f"{where} ORDER BY updated_at DESC, rowid DESC"
    )
    try:
        with _connection(path) as conn:
            rows = conn.execute(query, params).fetchall()
            return [_row_to_dict(row) for row in rows]
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("记忆列表读取失败：数据库错误。") from exc


def deactivate_memory(
    memory_id: str,
    *,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """停用记忆（active=0）并刷新 updated_at；记忆不存在时抛出 RepositoryError。"""
    now = _utc_now_iso()
    try:
        with _connection(path) as conn:
            cursor = conn.execute(
                "UPDATE learning_memories SET active = 0, updated_at = ? "
                "WHERE id = ?",
                (now, memory_id),
            )
            if cursor.rowcount == 0:
                raise RepositoryError("记忆不存在，无法停用。")
            row = conn.execute(
                "SELECT id, memory_type, topic, content, normalized_content, "
                "evidence_count, confidence, source_conversation_id, "
                "source_message_id, confirmed, active, created_at, updated_at "
                "FROM learning_memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            return _row_to_dict(row)
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("记忆停用失败：数据库错误。") from exc


def delete_memory(
    memory_id: str,
    *,
    path: DatabasePath | None = None,
) -> bool:
    """删除记忆；返回是否删除成功。"""
    try:
        with _connection(path) as conn:
            cursor = conn.execute(
                "DELETE FROM learning_memories WHERE id = ?",
                (memory_id,),
            )
            return cursor.rowcount > 0
    except RepositoryError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise RepositoryError("记忆删除失败：数据库错误。") from exc
