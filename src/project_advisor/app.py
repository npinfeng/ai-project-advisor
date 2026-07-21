"""FastAPI application for the interactive Project Advisor demo."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from project_advisor.graph import graph

load_dotenv()

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")

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


def _build_question(payload: AdviceRequest) -> str:
    if not payload.candidates:
        return payload.question
    candidates = "、".join(payload.candidates)
    return f"{payload.question}\n\n请优先评估以下候选项目：{candidates}。"


async def _stream_graph(
    payload: AdviceRequest,
    http_request: Request,
) -> AsyncIterator[str]:
    question = _build_question(payload)
    config = {
        "configurable": {
            "allow_clarification": payload.allow_clarification,
        }
    }
    final_report = ""
    scores: list[dict[str, Any]] = []

    yield _sse_event(
        "started",
        {
            "message": "评估任务已启动",
            "stages": list(NODE_LABELS.values()),
        },
    )

    try:
        async for update in graph.astream(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
            stream_mode="updates",
        ):
            if await http_request.is_disconnected():
                return

            for node_name, node_output in update.items():
                if node_name not in NODE_LABELS:
                    continue

                output = node_output if isinstance(node_output, dict) else {}
                if node_name == "review_and_score":
                    scores = _serialize_scores(output.get("scores", []))
                if node_name == "generate_report":
                    final_report = output.get("final_report", "")

                yield _sse_event(
                    "progress",
                    {
                        "node": node_name,
                        "label": NODE_LABELS[node_name],
                        "status": "completed",
                        "scores": scores if node_name == "review_and_score" else [],
                    },
                )

        if final_report:
            yield _sse_event(
                "result",
                {"report": final_report, "scores": scores},
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
