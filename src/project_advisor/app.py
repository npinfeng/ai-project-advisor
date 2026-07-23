"""FastAPI application for the interactive Project Advisor demo."""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from project_advisor.configuration import Configuration
from project_advisor.evaluation import evaluate_cases, load_evaluation_bundle
from project_advisor.mcp_client import get_mcp_diagnostics

load_dotenv()

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_FILE = PROJECT_ROOT / "evals" / "real_results.json"
URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}]+")
graph: Any | None = None

app = FastAPI(
    title="AI Project Advisor",
    description="Multi-agent technology selection and open-source project evaluation",
    version="0.2.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AdviceRequest(BaseModel):
    """Input accepted by the streaming advisor endpoint."""

    question: str = Field(min_length=10, max_length=5000)
    candidates: list[str] = Field(default_factory=list, max_length=8)
    allow_clarification: bool = False
    confirmed_plan: dict[str, Any] | None = None
    confirmed_candidates: bool = False

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        return value.strip()

    @field_validator("candidates")
    @classmethod
    def normalize_candidates(cls, value: list[str]) -> list[str]:
        normalized = []
        for candidate in value:
            candidate_name = candidate.strip()
            if candidate_name and candidate_name not in normalized:
                normalized.append(candidate_name)
        return normalized


class CandidateSuggestionRequest(BaseModel):
    """Input for the candidate preview stage."""

    question: str = Field(min_length=10, max_length=5000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        return value.strip()


NODE_LABELS = {
    "clarify_requirements": "需求解析",
    "plan_evaluation": "生成评估计划",
    "research_supervisor": "并行证据研究",
    "review_and_score": "证据审查与评分",
    "generate_report": "生成最终报告",
}


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

    from project_advisor.agents.planner import generate_research_plan

    return await generate_research_plan(
        [HumanMessage(content=question)],
        {"configurable": {"allow_clarification": False}},
    )


def _get_graph() -> Any:
    """Load and compile the LangGraph workflow only when an evaluation starts."""
    global graph
    if graph is None:
        from project_advisor.graph import graph as compiled_graph

        graph = compiled_graph
    return graph


def _evaluation_file() -> Path:
    configured_path = os.getenv("EVALUATION_FILE", "").strip()
    return Path(configured_path).expanduser() if configured_path else DEFAULT_EVALUATION_FILE


def _count_citation_urls(report: str) -> int:
    return len({url.rstrip(".,;:!?，。；：！？\"'") for url in URL_PATTERN.findall(report)})


def _usage_values(usage: Mapping[str, Any]) -> tuple[int, int]:
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    return int(input_tokens or 0), int(output_tokens or 0)


def _collect_token_usage(
    value: Any,
    seen_objects: set[int] | None = None,
) -> tuple[int, int]:
    """Collect LangChain/OpenAI usage metadata from one graph update."""
    seen_objects = seen_objects if seen_objects is not None else set()
    object_id = id(value)
    if object_id in seen_objects:
        return 0, 0
    seen_objects.add(object_id)

    if isinstance(value, Mapping):
        direct_usage = value.get("usage_metadata") or value.get("token_usage")
        if isinstance(direct_usage, Mapping):
            return _usage_values(direct_usage)
        total_input = 0
        total_output = 0
        for child in value.values():
            child_input, child_output = _collect_token_usage(child, seen_objects)
            total_input += child_input
            total_output += child_output
        return total_input, total_output

    if isinstance(value, (list, tuple, set)):
        total_input = 0
        total_output = 0
        for child in value:
            child_input, child_output = _collect_token_usage(child, seen_objects)
            total_input += child_input
            total_output += child_output
        return total_input, total_output

    usage_metadata = getattr(value, "usage_metadata", None)
    if isinstance(usage_metadata, Mapping):
        return _usage_values(usage_metadata)
    response_metadata = getattr(value, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if isinstance(token_usage, Mapping):
            return _usage_values(token_usage)
    return 0, 0


def _build_runtime_diagnostics(
    *,
    started_at: float,
    stage_durations_ms: dict[str, int],
    candidate_count: int,
    report: str,
    input_tokens: int,
    output_tokens: int,
    config: Configuration,
) -> dict[str, Any]:
    estimated_cost = (
        input_tokens * config.input_price_per_million
        + output_tokens * config.output_price_per_million
    ) / 1_000_000
    return {
        "total_duration_ms": round((time.perf_counter() - started_at) * 1000),
        "stage_durations_ms": stage_durations_ms,
        "candidate_count": candidate_count,
        "citation_url_count": _count_citation_urls(report),
        "mcp": get_mcp_diagnostics(config),
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "collected": (input_tokens + output_tokens) > 0,
        },
        "estimated_cost_usd": round(estimated_cost, 6),
        "cost_configured": (
            config.input_price_per_million > 0 or config.output_price_per_million > 0
        ),
    }


async def _stream_graph(
    payload: AdviceRequest,
    http_request: Request,
) -> AsyncIterator[str]:
    question = _build_question(payload)
    runtime_config = Configuration.from_runnable_config(
        {"configurable": {"allow_clarification": payload.allow_clarification}}
    )
    config = {
        "configurable": {
            "allow_clarification": payload.allow_clarification,
        }
    }
    final_report = ""
    scores: list[dict[str, Any]] = []
    retrieved_evidences: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    stage_started_at = started_at
    stage_durations_ms: dict[str, int] = {}
    input_tokens = 0
    output_tokens = 0

    yield _sse_event(
        "started",
        {
            "message": "评估任务已启动",
            "stages": list(NODE_LABELS.values()),
            "diagnostics": {
                "candidate_count": len(payload.candidates),
                "mcp": get_mcp_diagnostics(runtime_config),
            },
        },
    )

    try:
        graph_input = {
            "messages": [{"role": "user", "content": question}],
            "confirmed_candidates": payload.candidates if payload.confirmed_candidates else [],
        }
        if payload.confirmed_plan is not None:
            graph_input["confirmed_plan"] = payload.confirmed_plan

        async for update in _get_graph().astream(
            graph_input,
            config=config,
            stream_mode="updates",
        ):
            if await http_request.is_disconnected():
                return

            for node_name, node_output in update.items():
                if node_name not in NODE_LABELS:
                    continue

                output = node_output if isinstance(node_output, dict) else {}
                update_input, update_output = _collect_token_usage(output)
                input_tokens += update_input
                output_tokens += update_output
                if node_name == "review_and_score":
                    scores = _serialize_scores(output.get("scores", []))
                if output.get("evidences"):
                    combined = [*retrieved_evidences, *output.get("evidences", [])]
                    retrieved_evidences = _serialize_evidences(combined)
                if node_name == "generate_report":
                    final_report = output.get("final_report", "")

                now = time.perf_counter()
                stage_duration_ms = round((now - stage_started_at) * 1000)
                stage_durations_ms[node_name] = stage_duration_ms
                stage_started_at = now

                yield _sse_event(
                    "progress",
                    {
                        "node": node_name,
                        "label": NODE_LABELS[node_name],
                        "status": "completed",
                        "stage_duration_ms": stage_duration_ms,
                        "scores": scores if node_name == "review_and_score" else [],
                    },
                )

        if final_report:
            yield _sse_event(
                "result",
                {
                    "report": final_report,
                    "scores": scores,
                    "retrieved_evidences": retrieved_evidences,
                    "diagnostics": _build_runtime_diagnostics(
                        started_at=started_at,
                        stage_durations_ms=stage_durations_ms,
                        candidate_count=len(scores) or len(payload.candidates),
                        report=final_report,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        config=runtime_config,
                    ),
                },
            )
        else:
            yield _sse_event(
                "error",
                {"message": "工作流已结束，但没有生成报告。请检查模型和搜索配置。"},
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Project evaluation stream failed")
        yield _sse_event(
            "error",
            {"message": "评估执行失败，请检查 API Key、模型配置和服务日志。"},
        )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "project-advisor"}


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


@app.post("/api/candidates/suggest")
async def suggest_candidates(payload: CandidateSuggestionRequest) -> dict[str, Any]:
    """Preview structured requirements and AI-recommended candidates."""
    try:
        plan = await _generate_candidate_plan(payload.question)
        return plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)
    except Exception as error:
        logger.exception("Candidate suggestion failed")
        raise HTTPException(
            status_code=503,
            detail="候选项目生成失败，请检查模型配置后重试。",
        ) from error


@app.post("/api/advice/stream")
async def stream_advice(
    payload: AdviceRequest,
    request: Request,
) -> StreamingResponse:
    return StreamingResponse(
        _stream_graph(payload, request),
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
