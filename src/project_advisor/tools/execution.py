"""Bounded, observable execution policy shared by all Research Agent tools."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field, ValidationError

from project_advisor.configuration import Configuration
from project_advisor.errors import ToolReportedError
from project_advisor.observability.logging import log_event
from project_advisor.schemas.evidence import Evidence
from project_advisor.tools.evidence_factory import (
    build_evidences_from_tool_result,
    is_error_tool_result,
)
from project_advisor.usage_tracking import add_usage, usage_scope

logger = logging.getLogger(__name__)


class ToolExecutionRecord(BaseModel):
    """Secret-free execution trace suitable for checkpoints and diagnostics."""

    tool_name: str
    agent_run_id: str = ""
    call_id: str = ""
    project_name: str = ""
    status: Literal[
        "succeeded",
        "failed",
        "timed_out",
        "invalid_arguments",
        "unavailable",
        "rejected",
    ]
    latency_ms: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    error_type: str = ""


@dataclass(slots=True)
class ToolExecutionResult:
    observation: str
    evidences: list[Evidence]
    token_usage: dict[str, int]
    record: ToolExecutionRecord


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", getattr(tool, "__name__", "unknown")))


def _validate_arguments(tool: Any, args: Any) -> None:
    if not isinstance(args, dict):
        raise TypeError("工具参数必须是 JSON 对象。")
    schema = getattr(tool, "args_schema", None)
    if schema is not None and hasattr(schema, "model_validate"):
        schema.model_validate(args)


def _is_retryable(error: Exception) -> bool:
    """Retry only failures that are likely transient."""
    if isinstance(error, ToolReportedError):
        return error.retryable
    if isinstance(error, (asyncio.TimeoutError, httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(error, (ValidationError, TypeError, ValueError)):
        return False
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429 or status_code >= 500
    message = str(error).casefold()
    return any(marker in message for marker in (
        "rate limit",
        "temporarily unavailable",
        "connection reset",
        "connection refused",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
    ))


def _safe_error(error: Exception) -> str:
    detail = " ".join(str(error).split())[:500]
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _reported_error(result: Any) -> ToolReportedError | None:
    if not is_error_tool_result(result):
        return None
    message = str(result)
    normalized = message.casefold()
    timed_out = "超时" in normalized or "timeout" in normalized
    retryable = timed_out or any(marker in normalized for marker in (
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "api 频率限制",
        "temporarily unavailable",
    ))
    return ToolReportedError(message, retryable=retryable, timed_out=timed_out)


def _emit_record(record: ToolExecutionRecord) -> None:
    level = logging.INFO if record.status == "succeeded" else logging.WARNING
    log_event(logger, level, "tool_execution", **record.model_dump())


async def execute_tool(
    tool: Any,
    args: Any,
    config: RunnableConfig,
    *,
    call_id: str = "",
    requested_tool_name: str = "",
    project_name: str,
    research_topic: str,
) -> ToolExecutionResult:
    """Validate, time-bound, retry and normalize one tool invocation."""
    configurable = Configuration.from_runnable_config(config)
    agent_run_id = str(
        (config or {}).get("configurable", {}).get("thread_id", "")
    )
    started_at = time.perf_counter()
    name = requested_tool_name or (_tool_name(tool) if tool is not None else "unknown")
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    if tool is None:
        record = ToolExecutionRecord(
            tool_name=name,
            agent_run_id=agent_run_id,
            call_id=call_id,
            project_name=project_name,
            status="unavailable",
            latency_ms=0,
            retry_count=0,
            error_type="ToolNotFound",
        )
        _emit_record(record)
        return ToolExecutionResult("工具执行失败：工具未注册或无权访问。", [], total_usage, record)

    try:
        _validate_arguments(tool, args)
    except (ValidationError, TypeError, ValueError) as error:
        record = ToolExecutionRecord(
            tool_name=name,
            agent_run_id=agent_run_id,
            call_id=call_id,
            project_name=project_name,
            status="invalid_arguments",
            latency_ms=round((time.perf_counter() - started_at) * 1000),
            retry_count=0,
            error_type=type(error).__name__,
        )
        _emit_record(record)
        return ToolExecutionResult(
            f"工具参数校验失败：{_safe_error(error)}", [], total_usage, record
        )

    attempts = configurable.tool_max_retries + 1
    last_error: Exception | None = None
    last_attempt = 0
    for attempt in range(attempts):
        last_attempt = attempt
        nested_usage = {"input_tokens": 0, "output_tokens": 0}
        try:
            with usage_scope() as nested_usage:
                invocation = args
                if getattr(tool, "response_format", "") == "content_and_artifact":
                    invocation = {"type": "tool_call", "name": name, "id": call_id or name, "args": args}
                result = await asyncio.wait_for(
                    tool.ainvoke(invocation, config),
                    timeout=configurable.tool_timeout_seconds,
                )
            artifact = result.artifact if isinstance(result, ToolMessage) else None
            result = result.content if isinstance(result, ToolMessage) else result
            reported_error = _reported_error(result)
            if reported_error is not None:
                raise reported_error
            total_usage = add_usage(total_usage, nested_usage)
            record = ToolExecutionRecord(
                tool_name=name,
                agent_run_id=agent_run_id,
                call_id=call_id,
                project_name=project_name,
                status="succeeded",
                latency_ms=round((time.perf_counter() - started_at) * 1000),
                retry_count=attempt,
            )
            _emit_record(record)
            return ToolExecutionResult(
                str(result),
                build_evidences_from_tool_result(
                    tool_name=name,
                    args=args,
                    result=result,
                    artifact=artifact,
                    project_name=project_name,
                    research_topic=research_topic,
                ),
                total_usage,
                record,
            )
        except Exception as error:  # Normalized into an observation for replanning.
            total_usage = add_usage(total_usage, nested_usage)
            last_error = error
            if attempt >= attempts - 1 or not _is_retryable(error):
                break
            delay = configurable.tool_retry_backoff_seconds * (2**attempt)
            if delay:
                await asyncio.sleep(delay)

    assert last_error is not None
    timed_out = bool(getattr(last_error, "timed_out", False)) or isinstance(
        last_error, (asyncio.TimeoutError, httpx.TimeoutException)
    )
    record = ToolExecutionRecord(
        tool_name=name,
        agent_run_id=agent_run_id,
        call_id=call_id,
        project_name=project_name,
        status="timed_out" if timed_out else "failed",
        latency_ms=round((time.perf_counter() - started_at) * 1000),
        retry_count=last_attempt,
        error_type=type(last_error).__name__,
    )
    _emit_record(record)
    return ToolExecutionResult(
        f"工具执行失败：{_safe_error(last_error)}",
        [],
        total_usage,
        record,
    )
