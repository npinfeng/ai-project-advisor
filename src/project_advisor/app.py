"""FastAPI service for interactive, resumable Project Advisor runs."""

import asyncio
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Mapping
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from project_advisor import __version__
from project_advisor.api.schemas import (
    AdviceRequest,
    CandidateSuggestionRequest,
    TaskResumeRequest,
)
from project_advisor.configuration import Configuration
from project_advisor.errors import (
    AgentRunTimeoutError,
    ModelConfigurationError,
    PersistenceError,
    ProjectAdvisorError,
    StructuredOutputError,
)
from project_advisor.evaluation import evaluate_cases, load_evaluation_bundle
from project_advisor.mcp_client import get_mcp_diagnostics
from project_advisor.observability.diagnostics import (
    build_runtime_diagnostics as _build_runtime_diagnostics,
    collect_token_usage as _collect_token_usage,
    usage_values as _usage_values,
)
from project_advisor.observability.logging import bind_log_context, log_event
from project_advisor.persistence import TaskStore

load_dotenv()

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_FILE = PROJECT_ROOT / "evals" / "real_results.json"
graph: Any | None = None
_rate_windows: dict[str, list[float]] = {}
_rate_lock = asyncio.Lock()
_capacity_lock = asyncio.Lock()
_active_expensive_requests = 0


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Own persistent SQLite connections for the full application lifetime."""
    import aiosqlite
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from project_advisor.graph import compile_graph
    from project_advisor.schemas.evidence import (
        CandidateProject,
        CandidateRecommendation,
        EvaluationCriteria,
        Evidence,
        ProjectScore,
        Requirement,
        Requirements,
        ReviewResult,
    )
    from project_advisor.state import EvidenceGap, ResearchPlan, ResearchTask

    config = Configuration.from_runnable_config()
    checkpoint_path = _resolve_project_path(config.checkpoint_db_path)
    task_path = _resolve_project_path(config.task_db_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    task_store = TaskStore(task_path)
    await task_store.setup()
    recovered_tasks = await task_store.recover_incomplete()
    if recovered_tasks:
        log_event(
            logger,
            logging.WARNING,
            "orphaned_tasks_recovered",
            recovered_task_count=recovered_tasks,
        )
    serializer = JsonPlusSerializer(allowed_msgpack_modules=[
        Requirement,
        Requirements,
        CandidateProject,
        CandidateRecommendation,
        Evidence,
        EvaluationCriteria,
        ProjectScore,
        ReviewResult,
        ResearchPlan,
        ResearchTask,
        EvidenceGap,
    ])
    async with aiosqlite.connect(checkpoint_path) as connection:
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA busy_timeout=5000")
        checkpointer = AsyncSqliteSaver(connection, serde=serializer)
        await checkpointer.setup()
        application.state.checkpointer = checkpointer
        application.state.task_store = task_store
        application.state.graph = compile_graph(checkpointer=checkpointer)
        application.state.checkpoint_path = str(checkpoint_path)
        application.state.task_path = str(task_path)
        try:
            yield
        finally:
            for name in (
                "graph",
                "checkpointer",
                "task_store",
                "checkpoint_path",
                "task_path",
            ):
                if hasattr(application.state, name):
                    delattr(application.state, name)


app = FastAPI(
    title="AI Project Advisor",
    description="Multi-agent technology selection and open-source project evaluation",
    version=__version__,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(PersistenceError)
async def persistence_error_handler(
    request: Request,
    error: PersistenceError,
) -> JSONResponse:
    """Expose a stable service-level response without leaking database details."""
    log_event(
        logger,
        logging.ERROR,
        "persistence_error",
        path=request.url.path,
        error_type=type(error).__name__,
    )
    return JSONResponse(
        status_code=503,
        content={"detail": "任务持久化服务暂时不可用，请稍后重试。"},
    )


NODE_LABELS = {
    "clarify_requirements": "需求解析",
    "await_clarification": "等待需求补充",
    "plan_evaluation": "生成评估计划",
    "confirm_plan": "确认候选项目",
    "feasibility_check": "约束可行性预检",
    "parallel_research": "专业化并行研究",
    "evidence_coverage": "确定性证据检查",
    "supplemental_research": "受限补充研究",
    "review_and_score": "证据审查与评分",
    "generate_report": "确定性报告生成",
}


def _presented_api_key(request: Request) -> str:
    direct = request.headers.get("x-api-key", "").strip()
    if direct:
        return direct
    authorization = request.headers.get("authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def _require_api_key(request: Request, config: Configuration) -> None:
    if not config.advisor_api_key:
        return
    presented = _presented_api_key(request)
    if not presented or not secrets.compare_digest(
        presented, config.advisor_api_key
    ):
        raise HTTPException(status_code=401, detail="API 访问密钥无效。")


async def _protect_expensive_endpoint(request: Request) -> Configuration:
    """Apply optional authentication and per-client in-memory rate limiting."""
    config = Configuration.from_runnable_config()
    _require_api_key(request, config)

    client_host = request.client.host if request.client else "unknown"
    now = time.monotonic()
    cutoff = now - 60.0
    async with _rate_lock:
        recent = [stamp for stamp in _rate_windows.get(client_host, []) if stamp > cutoff]
        if len(recent) >= config.api_rate_limit_per_minute:
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试。")
        recent.append(now)
        _rate_windows[client_host] = recent
    return config


async def _try_acquire_capacity(limit: int) -> bool:
    global _active_expensive_requests
    async with _capacity_lock:
        if _active_expensive_requests >= limit:
            return False
        _active_expensive_requests += 1
        return True


async def _release_capacity() -> None:
    global _active_expensive_requests
    async with _capacity_lock:
        _active_expensive_requests = max(0, _active_expensive_requests - 1)


async def _with_capacity_release(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    try:
        async for item in stream:
            yield item
    finally:
        await _release_capacity()


def _budget_violation(
    config: Configuration,
    input_tokens: int,
    output_tokens: int,
) -> str | None:
    total_tokens = input_tokens + output_tokens
    if config.max_run_tokens and total_tokens > config.max_run_tokens:
        return f"任务已达到 Token 上限（{config.max_run_tokens}）。"
    observed_cost = (
        input_tokens * config.input_price_per_million
        + output_tokens * config.output_price_per_million
    ) / 1_000_000
    if config.max_run_cost_usd and observed_cost > config.max_run_cost_usd:
        return f"任务已达到成本上限（${config.max_run_cost_usd:.4f}）。"
    return None


def _safe_model_error_message(error: Exception) -> str:
    if isinstance(error, ModelConfigurationError):
        return "模型配置无效，请检查 Provider、模型名称和 API Key。"
    if isinstance(error, StructuredOutputError):
        return "模型多次返回无效结构化结果，请稍后重试或更换模型。"
    message = str(error).casefold()
    if "api_key" in message or "api key" in message:
        return "模型 API Key 未配置，请检查 .env 文件。"
    if "jiter" in message or "应用程序控制策略" in message:
        return (
            "旧版 OpenAI SDK 的原生依赖被系统策略拦截。"
            "请重启服务以加载项目内置的纯 HTTPX 兼容客户端。"
        )
    return "候选项目生成失败，请检查模型配置和服务日志。"


def _sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _serialize_scores(scores: list[Any]) -> list[dict[str, Any]]:
    return [
        score.model_dump() if hasattr(score, "model_dump") else dict(score)
        for score in scores
    ]


def _serialize_evidences(evidences: list[Any]) -> list[dict[str, Any]]:
    """Expose only stable retrieval provenance needed by evaluation runners."""
    serialized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for evidence in evidences:
        value = (
            evidence.model_dump()
            if hasattr(evidence, "model_dump")
            else dict(evidence)
        )
        evidence_id = str(value.get("evidence_id", ""))
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        serialized.append({
            "evidence_id": evidence_id,
            "source_url": value.get("source_url", ""),
            "source_type": value.get("source_type", ""),
            "project_name": value.get("project_name", ""),
            "relevance": value.get("relevance", ""),
        })
    return serialized


def _build_question(payload: AdviceRequest) -> str:
    if not payload.candidates:
        return payload.question
    candidates = "、".join(payload.candidates)
    instruction = "仅评估用户已确认的以下候选项目" if payload.confirmed_candidates else "请优先评估以下候选项目"
    return f"{payload.question}\n\n{instruction}：{candidates}。"


async def _generate_candidate_plan(question: str) -> Any:
    """Generate the same structured plan later consumed by the workflow."""
    from langchain_core.messages import HumanMessage

    from project_advisor.agents.planner import generate_research_plan_with_usage

    return await generate_research_plan_with_usage(
        [HumanMessage(content=question)],
        {"configurable": {"allow_clarification": False}},
    )


def _unpack_generated_plan(generated: Any) -> tuple[dict[str, Any], dict[str, int]]:
    """Normalize Planner output for preview and execution-time plan reuse."""
    if isinstance(generated, tuple):
        plan, token_usage = generated
    else:
        plan, token_usage = generated, {"input_tokens": 0, "output_tokens": 0}
    result = plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)
    normalized_usage = {
        "input_tokens": int(token_usage.get("input_tokens", 0) or 0),
        "output_tokens": int(token_usage.get("output_tokens", 0) or 0),
    }
    return result, normalized_usage


def _get_graph() -> Any:
    """Load and compile the LangGraph workflow only when an evaluation starts."""
    global graph
    if graph is not None:
        return graph
    runtime_graph = getattr(app.state, "graph", None)
    if runtime_graph is not None:
        return runtime_graph
    if graph is None:
        from project_advisor.graph import graph as compiled_graph

        graph = compiled_graph
    return graph


def _task_store() -> TaskStore | None:
    return getattr(app.state, "task_store", None)


async def _update_task(task_id: str, **values: Any) -> dict[str, Any] | None:
    store = _task_store()
    return await store.update(task_id, **values) if store is not None else None


def _interrupt_value(update: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = update.get("__interrupt__")
    if not raw:
        return None
    first = raw[0] if isinstance(raw, (list, tuple)) else raw
    value = getattr(first, "value", first)
    if isinstance(value, Mapping):
        return dict(value)
    return {"kind": "input", "question": str(value)}


def _evaluation_file() -> Path:
    configured_path = os.getenv("EVALUATION_FILE", "").strip()
    return Path(configured_path).expanduser() if configured_path else DEFAULT_EVALUATION_FILE


_NO_RESUME = object()


async def _stream_graph(
    payload: AdviceRequest,
    http_request: Request,
    task_id: str,
    *,
    continue_from_checkpoint: bool = False,
    resume_value: Any = _NO_RESUME,
) -> AsyncIterator[str]:
    """Bind correlation context around the complete SSE generator lifetime."""
    with bind_log_context(
        request_id=task_id,
        task_id=task_id,
        agent_run_id=task_id,
    ):
        async for event in _stream_graph_impl(
            payload,
            http_request,
            task_id,
            continue_from_checkpoint=continue_from_checkpoint,
            resume_value=resume_value,
        ):
            yield event


async def _stream_graph_impl(
    payload: AdviceRequest,
    http_request: Request,
    task_id: str,
    *,
    continue_from_checkpoint: bool = False,
    resume_value: Any = _NO_RESUME,
) -> AsyncIterator[str]:
    question = _build_question(payload)
    runtime_config = Configuration.from_runnable_config(
        {"configurable": {"allow_clarification": payload.allow_clarification}}
    )
    config = {
        "configurable": {
            "allow_clarification": payload.allow_clarification,
            "thread_id": task_id,
        }
    }
    final_report = ""
    scores: list[dict[str, Any]] = []
    retrieved_evidences: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    stage_started_at = started_at
    record = await _task_store().get(task_id) if _task_store() is not None else None
    previous_diagnostics = (record or {}).get("diagnostics") or {}
    previous_total_ms = int(previous_diagnostics.get("total_duration_ms", 0) or 0)
    stage_durations_ms = dict(previous_diagnostics.get("stage_durations_ms", {}))
    tool_execution = {
        "total": 0,
        "succeeded": 0,
        "failed": 0,
        "timed_out": 0,
        "invalid_arguments": 0,
        "unavailable": 0,
        "rejected": 0,
        "retries": 0,
        "total_latency_ms": 0,
        **dict(previous_diagnostics.get("tool_execution", {})),
    }
    context_budget = dict(previous_diagnostics.get("context_budget", {}))
    deadline = started_at + runtime_config.agent_run_timeout_seconds
    execution_plan = payload.confirmed_plan
    if continue_from_checkpoint:
        previous_usage = previous_diagnostics.get("token_usage", {})
        input_tokens, output_tokens = _usage_values(previous_usage)
        scores = list((record or {}).get("scores") or [])
        retrieved_evidences = list((record or {}).get("retrieved_evidences") or [])
    else:
        planning_diagnostics = (
            execution_plan.get("planning_diagnostics", {}) if execution_plan else {}
        )
        input_tokens, output_tokens = _usage_values(
            planning_diagnostics.get("token_usage", {})
        )

    def diagnostics() -> dict[str, Any]:
        value = _build_runtime_diagnostics(
            started_at=started_at,
            stage_durations_ms=stage_durations_ms,
            candidate_count=len(scores) or len(payload.candidates),
            report=final_report,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_execution=tool_execution,
            config=runtime_config,
            context_budget=context_budget,
        )
        value["total_duration_ms"] += previous_total_ms
        value["checkpoint"] = {
            "enabled": getattr(app.state, "checkpointer", None) is not None,
            "thread_id": task_id,
        }
        return value

    def remaining_seconds() -> float:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise AgentRunTimeoutError(
                f"任务超过端到端超时 {runtime_config.agent_run_timeout_seconds:g}s。"
            )
        return remaining

    def collect_tool_executions(output: Mapping[str, Any]) -> None:
        for record_value in output.get("tool_executions", []) or []:
            record = (
                record_value.model_dump()
                if hasattr(record_value, "model_dump")
                else dict(record_value)
            )
            status = str(record.get("status", "failed"))
            tool_execution["total"] += 1
            if status in tool_execution:
                tool_execution[status] += 1
            else:
                tool_execution["failed"] += 1
            tool_execution["retries"] += int(record.get("retry_count", 0) or 0)
            tool_execution["total_latency_ms"] += int(
                record.get("latency_ms", 0) or 0
            )

    await _update_task(task_id, status="running", pending_interrupt=None, error="")
    log_event(
        logger,
        logging.INFO,
        "agent_run_started",
        resumed=continue_from_checkpoint,
        candidate_count=len(payload.candidates),
    )
    yield _sse_event(
        "started",
        {
            "task_id": task_id,
            "thread_id": task_id,
            "resumed": continue_from_checkpoint,
            "message": "评估任务已恢复" if continue_from_checkpoint else "评估任务已启动",
            "stages": list(NODE_LABELS.values()),
            "diagnostics": {
                "candidate_count": len(payload.candidates),
                "mcp": get_mcp_diagnostics(runtime_config),
            },
        },
    )

    try:
        if continue_from_checkpoint:
            if resume_value is _NO_RESUME:
                graph_input: Any = None
            else:
                from langgraph.types import Command

                graph_input = Command(resume=resume_value)
        else:
            defer_plan_to_graph = (
                payload.allow_clarification
                and execution_plan is None
                and not payload.confirmed_candidates
            )
            if execution_plan is None and not defer_plan_to_graph:
                try:
                    generated = await asyncio.wait_for(
                        _generate_candidate_plan(question),
                        timeout=remaining_seconds(),
                    )
                except asyncio.TimeoutError as error:
                    raise AgentRunTimeoutError(
                        f"任务超过端到端超时 "
                        f"{runtime_config.agent_run_timeout_seconds:g}s。"
                    ) from error
                execution_plan, generated_usage = _unpack_generated_plan(generated)
                execution_plan["planning_diagnostics"] = {
                    "token_usage": generated_usage,
                }
                input_tokens += generated_usage["input_tokens"]
                output_tokens += generated_usage["output_tokens"]
                await _update_task(task_id, confirmed_plan=execution_plan)
                budget_error = _budget_violation(
                    runtime_config, input_tokens, output_tokens
                )
                if budget_error:
                    await _update_task(
                        task_id, status="failed", error=budget_error,
                        diagnostics=diagnostics(),
                    )
                    yield _sse_event("error", {"task_id": task_id, "message": budget_error})
                    return

            graph_input = {
                "messages": [{"role": "user", "content": question}],
                "confirmed_candidates": (
                    payload.candidates if payload.confirmed_candidates else []
                ),
            }
            if execution_plan is not None:
                graph_input["confirmed_plan"] = execution_plan

        update_stream = _get_graph().astream(
            graph_input,
            config=config,
            stream_mode="updates",
        )
        update_iterator = update_stream.__aiter__()
        while True:
            try:
                update = await asyncio.wait_for(
                    anext(update_iterator),
                    timeout=remaining_seconds(),
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as error:
                raise AgentRunTimeoutError(
                    f"任务超过端到端超时 "
                    f"{runtime_config.agent_run_timeout_seconds:g}s。"
                ) from error
            if await http_request.is_disconnected():
                await _update_task(
                    task_id, status="paused", diagnostics=diagnostics()
                )
                return

            pending = _interrupt_value(update)
            if pending is not None:
                current_diagnostics = diagnostics()
                await _update_task(
                    task_id,
                    status="waiting_input",
                    pending_interrupt=pending,
                    diagnostics=current_diagnostics,
                )
                yield _sse_event(
                    "interrupt",
                    {"task_id": task_id, "thread_id": task_id, **pending},
                )
                return

            for node_name, node_output in update.items():
                if node_name not in NODE_LABELS:
                    continue

                output = node_output if isinstance(node_output, dict) else {}
                collect_tool_executions(output)
                update_input, update_output = _collect_token_usage(output)
                input_tokens += update_input
                output_tokens += update_output
                budget_error = _budget_violation(
                    runtime_config, input_tokens, output_tokens
                )
                if budget_error:
                    await _update_task(
                        task_id, status="failed", error=budget_error,
                        diagnostics=diagnostics(), last_node=node_name,
                    )
                    yield _sse_event("error", {"task_id": task_id, "message": budget_error})
                    return
                if node_name == "review_and_score":
                    scores = _serialize_scores(output.get("scores", []))
                    context_budget = dict(output.get("context_budget", {}))
                if output.get("evidences"):
                    combined = [*retrieved_evidences, *output.get("evidences", [])]
                    retrieved_evidences = _serialize_evidences(combined)
                if output.get("candidates"):
                    await _update_task(
                        task_id,
                        candidates=list(output.get("candidates", [])),
                        confirmed_candidates=(node_name == "confirm_plan"),
                    )
                if node_name == "generate_report":
                    final_report = output.get("final_report", "")

                now = time.perf_counter()
                stage_duration_ms = round((now - stage_started_at) * 1000)
                stage_durations_ms[node_name] = (
                    stage_durations_ms.get(node_name, 0) + stage_duration_ms
                )
                stage_started_at = now
                await _update_task(
                    task_id,
                    last_node=node_name,
                    scores=scores,
                    retrieved_evidences=retrieved_evidences,
                    diagnostics=diagnostics(),
                )
                log_event(
                    logger,
                    logging.INFO,
                    "workflow_node_completed",
                    node=node_name,
                    latency_ms=stage_duration_ms,
                    next=output.get("next"),
                )

                yield _sse_event(
                    "progress",
                    {
                        "task_id": task_id,
                        "node": node_name,
                        "label": NODE_LABELS[node_name],
                        "status": "completed",
                        "stage_duration_ms": stage_duration_ms,
                        "next": output.get("next"),
                        "scores": scores if node_name == "review_and_score" else [],
                    },
                )

        if final_report:
            final_diagnostics = diagnostics()
            result = {
                "task_id": task_id,
                "thread_id": task_id,
                "report": final_report,
                "scores": scores,
                "retrieved_evidences": retrieved_evidences,
                "diagnostics": final_diagnostics,
            }
            await _update_task(
                task_id,
                status="completed",
                pending_interrupt=None,
                report=final_report,
                scores=scores,
                retrieved_evidences=retrieved_evidences,
                diagnostics=final_diagnostics,
            )
            log_event(
                logger,
                logging.INFO,
                "agent_run_completed",
                total_duration_ms=final_diagnostics["total_duration_ms"],
                tool_execution=final_diagnostics["tool_execution"],
            )
            yield _sse_event("result", result)
        else:
            message = "工作流已结束，但没有生成报告。请检查模型和搜索配置。"
            await _update_task(
                task_id, status="failed", error=message, diagnostics=diagnostics()
            )
            yield _sse_event("error", {"task_id": task_id, "message": message})
    except asyncio.CancelledError:
        await _update_task(task_id, status="paused", diagnostics=diagnostics())
        raise
    except AgentRunTimeoutError as error:
        message = str(error)
        log_event(logger, logging.WARNING, "agent_run_timed_out", error=message)
        await _update_task(
            task_id,
            status="failed",
            error=message,
            diagnostics=diagnostics(),
        )
        yield _sse_event("error", {"task_id": task_id, "message": message})
    except ProjectAdvisorError as error:
        message = _safe_model_error_message(error)
        log_event(
            logger,
            logging.ERROR,
            "agent_run_domain_error",
            error_type=type(error).__name__,
        )
        await _update_task(
            task_id,
            status="failed",
            error=f"{type(error).__name__}: {error}",
            diagnostics=diagnostics(),
        )
        yield _sse_event("error", {"task_id": task_id, "message": message})
    except Exception as error:
        log_event(
            logger,
            logging.ERROR,
            "agent_run_failed",
            error_type=type(error).__name__,
        )
        logger.exception("Project evaluation stream failed")
        message = "评估执行失败，请检查 API Key、模型配置和服务日志。"
        await _update_task(
            task_id,
            status="failed",
            error=f"{type(error).__name__}: {error}",
            diagnostics=diagnostics(),
        )
        yield _sse_event("error", {"task_id": task_id, "message": message})


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict[str, Any]:
    config = Configuration.from_runnable_config()
    try:
        from project_advisor.utils import create_chat_model

        model = create_chat_model(
            config.research_model,
            max_tokens=min(config.research_model_max_tokens, 16),
            timeout_seconds=config.llm_timeout_seconds,
        )
        model_runtime = {
            "status": "ready",
            "provider": config.research_model.partition(":")[0],
            "client": type(model).__name__,
        }
    except Exception as error:
        model_runtime = {
            "status": "unavailable",
            "provider": config.research_model.partition(":")[0],
            "error_type": type(error).__name__,
        }
    service_status = "ok" if model_runtime["status"] == "ready" else "degraded"
    return {
        "status": service_status,
        "service": "project-advisor",
        "model_runtime": model_runtime,
        "persistence": {
            "status": "ready" if _task_store() is not None else "unavailable",
            "checkpoint_enabled": getattr(app.state, "checkpointer", None) is not None,
        },
    }


@app.get("/api/evaluation")
async def evaluation_dashboard() -> dict[str, Any]:
    """Return the latest reproducible offline-evaluation baseline."""
    evaluation_path = _evaluation_file()
    try:
        bundle = load_evaluation_bundle(evaluation_path)
        report = None
        status_message = ""
        if bundle.metadata.annotation_status == "reviewed":
            report = evaluate_cases(
                bundle.cases,
                k=bundle.k,
                metadata=bundle.metadata,
            )
        else:
            status_message = "真实运行已采集，等待独立人工审核后计算质量指标。"
        modified_at = datetime.fromtimestamp(
            evaluation_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="离线评测文件不存在。") from error
    except (ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=422, detail="离线评测文件格式无效。") from error
    return {
        "source": evaluation_path.name,
        "metadata": bundle.metadata.model_dump(),
        "k": bundle.k,
        "status_message": status_message,
        "updated_at": modified_at,
        "report": report.model_dump() if report is not None else None,
    }


def _task_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": record["task_id"],
        "status": record["status"],
        "question": record["question"],
        "candidates": record["candidates"],
        "pending_interrupt": record["pending_interrupt"],
        "last_node": record["last_node"],
        "has_report": bool(record["report"]),
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }


def _require_task_store(request: Request) -> TaskStore:
    config = Configuration.from_runnable_config()
    _require_api_key(request, config)
    store = _task_store()
    if store is None:
        raise HTTPException(status_code=503, detail="任务持久化服务尚未初始化。")
    return store


@app.get("/api/tasks")
async def list_tasks(request: Request, limit: int = 20) -> dict[str, Any]:
    store = _require_task_store(request)
    records = await store.list(limit=limit)
    return {"tasks": [_task_summary(record) for record in records]}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str, request: Request) -> dict[str, Any]:
    store = _require_task_store(request)
    record = await store.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return record


@app.post("/api/tasks/{task_id}/resume")
async def resume_task(
    task_id: str,
    payload: TaskResumeRequest,
    request: Request,
) -> StreamingResponse:
    config = await _protect_expensive_endpoint(request)
    store = _task_store()
    if store is None:
        raise HTTPException(status_code=503, detail="任务持久化服务尚未初始化。")
    record = await store.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在。")
    if record["status"] == "completed":
        raise HTTPException(status_code=409, detail="任务已经完成，无需恢复。")
    if record["status"] == "running":
        raise HTTPException(status_code=409, detail="任务仍在运行。")
    if record["status"] == "waiting_input" and payload.response is None:
        raise HTTPException(status_code=422, detail="当前任务正在等待用户输入。")
    if not await _try_acquire_capacity(config.max_concurrent_evaluations):
        raise HTTPException(status_code=429, detail="深度评估并发已满，请稍后重试。")

    advice = AdviceRequest(
        question=record["question"],
        candidates=record["candidates"] or [],
        allow_clarification=record["allow_clarification"],
        confirmed_plan=record["confirmed_plan"],
        confirmed_candidates=record["confirmed_candidates"],
    )
    resume_value = (
        payload.response if record["status"] == "waiting_input" else _NO_RESUME
    )
    return StreamingResponse(
        _with_capacity_release(_stream_graph(
            advice,
            request,
            task_id,
            continue_from_checkpoint=True,
            resume_value=resume_value,
        )),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/candidates/suggest")
async def suggest_candidates(
    payload: CandidateSuggestionRequest,
    request: Request,
) -> dict[str, Any]:
    """Preview structured requirements and AI-recommended candidates."""
    config = await _protect_expensive_endpoint(request)
    if not await _try_acquire_capacity(config.max_concurrent_evaluations):
        raise HTTPException(status_code=429, detail="服务繁忙，请稍后重试。")
    request_id = uuid4().hex
    with bind_log_context(request_id=request_id):
        try:
            generated = await _generate_candidate_plan(payload.question)
            result, token_usage = _unpack_generated_plan(generated)
            result["planning_diagnostics"] = {
                "token_usage": token_usage,
                "estimated_cost_usd": round(
                    (
                        token_usage.get("input_tokens", 0)
                        * config.input_price_per_million
                        + token_usage.get("output_tokens", 0)
                        * config.output_price_per_million
                    ) / 1_000_000,
                    6,
                ),
                "request_id": request_id,
            }
            log_event(logger, logging.INFO, "candidate_suggestion_completed")
            return result
        except Exception as error:
            log_event(
                logger,
                logging.ERROR,
                "candidate_suggestion_failed",
                error_type=type(error).__name__,
            )
            logger.exception("Candidate suggestion failed")
            raise HTTPException(
                status_code=503,
                detail=_safe_model_error_message(error),
            ) from error
        finally:
            await _release_capacity()


@app.post("/api/advice/stream")
async def stream_advice(
    payload: AdviceRequest,
    request: Request,
) -> StreamingResponse:
    config = await _protect_expensive_endpoint(request)
    if not await _try_acquire_capacity(config.max_concurrent_evaluations):
        raise HTTPException(status_code=429, detail="深度评估并发已满，请稍后重试。")
    task_id = uuid4().hex
    store = _task_store()
    try:
        if store is not None:
            await store.create(
                task_id=task_id,
                question=payload.question,
                candidates=payload.candidates,
                allow_clarification=payload.allow_clarification,
                confirmed_plan=payload.confirmed_plan,
                confirmed_candidates=payload.confirmed_candidates,
            )
    except Exception:
        await _release_capacity()
        raise
    return StreamingResponse(
        _with_capacity_release(_stream_graph(payload, request, task_id)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    """Run the web application with Uvicorn."""
    import uvicorn

    uvicorn.run(
        "project_advisor.app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
