"""Persistent task metadata stored alongside LangGraph checkpoints."""

from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite

from project_advisor.errors import PersistenceError


JSON_FIELDS = {
    "candidates",
    "confirmed_plan",
    "pending_interrupt",
    "scores",
    "retrieved_evidences",
    "diagnostics",
}
UPDATABLE_FIELDS = {
    "status",
    "candidates",
    "confirmed_plan",
    "confirmed_candidates",
    "pending_interrupt",
    "report",
    "scores",
    "retrieved_evidences",
    "diagnostics",
    "error",
    "last_node",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class TaskStore:
    """Small async SQLite repository for task discovery and report recovery."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Open one configured connection and normalize SQLite failures."""
        try:
            async with aiosqlite.connect(self.path) as connection:
                connection.row_factory = aiosqlite.Row
                await connection.execute("PRAGMA busy_timeout=5000")
                yield connection
        except sqlite3.Error as error:
            raise PersistenceError(
                f"任务数据库操作失败：{type(error).__name__}"
            ) from error

    async def setup(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as connection:
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS advisor_tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    question TEXT NOT NULL,
                    candidates TEXT NOT NULL DEFAULT '[]',
                    allow_clarification INTEGER NOT NULL DEFAULT 0,
                    confirmed_plan TEXT,
                    confirmed_candidates INTEGER NOT NULL DEFAULT 0,
                    pending_interrupt TEXT,
                    report TEXT NOT NULL DEFAULT '',
                    scores TEXT NOT NULL DEFAULT '[]',
                    retrieved_evidences TEXT NOT NULL DEFAULT '[]',
                    diagnostics TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    last_node TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_advisor_tasks_updated "
                "ON advisor_tasks(updated_at DESC)"
            )
            await connection.commit()

    async def create(
        self,
        *,
        task_id: str,
        question: str,
        candidates: list[str],
        allow_clarification: bool,
        confirmed_plan: dict[str, Any] | None,
        confirmed_candidates: bool,
    ) -> dict[str, Any]:
        timestamp = _now()
        async with self._connect() as connection:
            await connection.execute(
                """
                INSERT INTO advisor_tasks (
                    task_id, status, question, candidates, allow_clarification,
                    confirmed_plan, confirmed_candidates, created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    question,
                    _encode(candidates),
                    int(allow_clarification),
                    _encode(confirmed_plan) if confirmed_plan is not None else None,
                    int(confirmed_candidates),
                    timestamp,
                    timestamp,
                ),
            )
            await connection.commit()
        created = await self.get(task_id)
        if created is None:
            raise RuntimeError("任务创建后无法读取。")
        return created

    async def recover_incomplete(self) -> int:
        """Mark orphaned in-process tasks resumable after a service restart."""
        timestamp = _now()
        async with self._connect() as connection:
            cursor = await connection.execute(
                """
                UPDATE advisor_tasks
                SET status = 'paused',
                    error = '服务进程中断，任务已转为可恢复状态。',
                    updated_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (timestamp,),
            )
            await connection.commit()
            return max(0, cursor.rowcount)

    async def update(self, task_id: str, **values: Any) -> dict[str, Any] | None:
        filtered = {key: value for key, value in values.items() if key in UPDATABLE_FIELDS}
        if not filtered:
            return await self.get(task_id)
        encoded: dict[str, Any] = {}
        for key, value in filtered.items():
            if key in JSON_FIELDS:
                encoded[key] = _encode(value) if value is not None else None
            elif key == "confirmed_candidates":
                encoded[key] = int(bool(value))
            else:
                encoded[key] = value
        encoded["updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in encoded)
        async with self._connect() as connection:
            await connection.execute(
                f"UPDATE advisor_tasks SET {assignments} WHERE task_id = ?",
                (*encoded.values(), task_id),
            )
            await connection.commit()
        return await self.get(task_id)

    async def get(self, task_id: str) -> dict[str, Any] | None:
        async with self._connect() as connection:
            cursor = await connection.execute(
                "SELECT * FROM advisor_tasks WHERE task_id = ?",
                (task_id,),
            )
            row = await cursor.fetchone()
        return self._decode_row(row) if row is not None else None

    async def list(self, limit: int = 20) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 50))
        async with self._connect() as connection:
            cursor = await connection.execute(
                "SELECT * FROM advisor_tasks ORDER BY updated_at DESC LIMIT ?",
                (safe_limit,),
            )
            rows = await cursor.fetchall()
        return [self._decode_row(row) for row in rows]

    @staticmethod
    def _decode_row(row: aiosqlite.Row) -> dict[str, Any]:
        value = dict(row)
        for key in JSON_FIELDS:
            raw = value.get(key)
            if raw is None:
                value[key] = None
            else:
                value[key] = json.loads(raw)
        value["allow_clarification"] = bool(value["allow_clarification"])
        value["confirmed_candidates"] = bool(value["confirmed_candidates"])
        return value
