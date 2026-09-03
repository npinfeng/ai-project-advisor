"""Capture real workflow observations without auto-labeling their quality.

The suite contains ground truth written before execution. This runner only fills
observable fields (retrieved Evidence, generated citations, latency, tokens and
cost), then produces a pending file for independent human annotation.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from project_advisor.evaluation_suite import (
    golden_suite_sha256,
    load_golden_suite,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = PROJECT_ROOT / "evals" / "golden_cases.json"
URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}]+")
load_dotenv(PROJECT_ROOT / ".env")


def _extract_urls(text: str) -> list[str]:
    return sorted({
        url.rstrip(".,;:!?，。；：！？\"'")
        for url in URL_PATTERN.findall(text)
    })


def _load_suite(path: Path) -> dict[str, Any]:
    return load_golden_suite(path).model_dump()


async def _request_sse(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "task_id": "",
        "report": "",
        "diagnostics": {},
        "retrieved_evidences": [],
        "interrupt": None,
        "error": None,
    }
    async with client.stream(
        "POST", url, json=payload, timeout=timeout_seconds
    ) as response:
        response.raise_for_status()
        current_event = "message"
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                current_event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            event_data = json.loads(line[5:].strip())
            state["task_id"] = event_data.get("task_id", state["task_id"])
            if current_event == "result":
                state["report"] = event_data.get("report", "")
                state["diagnostics"] = event_data.get("diagnostics", {})
                state["retrieved_evidences"] = event_data.get(
                    "retrieved_evidences", []
                )
            elif current_event == "interrupt":
                state["interrupt"] = event_data
            elif current_event == "error":
                state["error"] = event_data.get("message", "工作流返回错误")
    return state


async def _run_case(
    client: httpx.AsyncClient,
    server: str,
    case: dict[str, Any],
    timeout_seconds: float,
    *,
    exercise_resume: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    final_report = ""
    diagnostics: dict[str, Any] = {}
    retrieved_evidences: list[dict[str, Any]] = []
    run_error: str | None = None
    interrupt_kinds: list[str] = []

    payload = {
        "question": case["question"],
        "candidates": case["candidates"],
        "allow_clarification": False,
        "confirmed_candidates": not exercise_resume,
    }
    try:
        result = await _request_sse(
            client,
            f"{server.rstrip('/')}/api/advice/stream",
            payload,
            timeout_seconds,
        )
        for _ in range(5):
            interrupt = result.get("interrupt")
            if not interrupt:
                break
            kind = str(interrupt.get("kind", "input"))
            interrupt_kinds.append(kind)
            if kind == "candidate_confirmation":
                resume_value: Any = {"candidates": case["candidates"]}
            else:
                resume_value = {"answer": case["question"]}
            task_id = result.get("task_id")
            if not task_id:
                raise ValueError("SSE interrupt 未返回 task_id，无法验证恢复链路。")
            result = await _request_sse(
                client,
                f"{server.rstrip('/')}/api/tasks/{task_id}/resume",
                {"response": resume_value},
                timeout_seconds,
            )
        if result.get("interrupt"):
            run_error = "恢复次数超过 5 次，工作流仍在等待输入。"
        else:
            run_error = result.get("error")
        final_report = result.get("report", "")
        diagnostics = result.get("diagnostics", {})
        retrieved_evidences = result.get("retrieved_evidences", [])
    except Exception as error:
        run_error = f"{type(error).__name__}: {error}"

    elapsed_ms = diagnostics.get(
        "total_duration_ms",
        round((time.perf_counter() - started_at) * 1000),
    )
    token_usage = diagnostics.get("token_usage", {})
    retrieved_documents = list(dict.fromkeys(
        str(item.get("source_url", ""))
        for item in retrieved_evidences
        if item.get("source_url")
    ))
    retrieved_evidence_ids = list(dict.fromkeys(
        str(item.get("evidence_id", ""))
        for item in retrieved_evidences
        if item.get("evidence_id")
    ))

    return {
        "case_id": case["case_id"],
        "relevant_documents": case["relevant_documents"],
        "retrieved_documents": retrieved_documents,
        "retrieved_evidence_ids": retrieved_evidence_ids,
        "expected_citations": case["expected_citations"],
        "generated_citations": _extract_urls(final_report),
        "supported_citations": [],
        "task_success": None,
        "latency_ms": elapsed_ms,
        "input_tokens": int(token_usage.get("input_tokens", 0) or 0),
        "output_tokens": int(token_usage.get("output_tokens", 0) or 0),
        "cost_usd": float(diagnostics.get("estimated_cost_usd", 0) or 0),
        "workflow_completed": bool(final_report) and run_error is None,
        "run_error": run_error,
        "report_sha256": (
            hashlib.sha256(final_report.encode("utf-8")).hexdigest()
            if final_report else None
        ),
        "annotation_context": {
            "success_criteria": case["success_criteria"],
            "run_error": run_error,
            "report_preview": final_report[:1500],
            "generated_report": final_report,
            "recovery_exercised": bool(interrupt_kinds),
            "interrupt_kinds": interrupt_kinds,
        },
    }


async def _preflight_release(
    client: httpx.AsyncClient,
    server: str,
    suite: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Verify health and exercise the candidate-suggestion model boundary."""
    health_response = await client.get(
        f"{server.rstrip('/')}/api/health", timeout=timeout_seconds
    )
    health_response.raise_for_status()
    health = health_response.json()
    if health.get("status") != "ok":
        raise RuntimeError(f"发布验收要求健康状态为 ok，当前为 {health.get('status')}。")
    if health.get("model_runtime", {}).get("status") != "ready":
        raise RuntimeError("发布验收要求真实模型运行时处于 ready。")
    if health.get("persistence", {}).get("status") != "ready":
        raise RuntimeError("发布验收要求任务持久化处于 ready。")
    if not health.get("persistence", {}).get("checkpoint_enabled"):
        raise RuntimeError("发布验收要求 LangGraph checkpoint 已启用。")

    first_case = suite["cases"][0]
    suggestion_response = await client.post(
        f"{server.rstrip('/')}/api/candidates/suggest",
        json={"question": first_case["question"]},
        timeout=timeout_seconds,
    )
    suggestion_response.raise_for_status()
    suggestion = suggestion_response.json()
    if not suggestion.get("candidates") or not suggestion.get("requirements"):
        raise RuntimeError("候选建议预检未返回结构化 candidates/requirements。")
    return {
        "health_status": health["status"],
        "model_runtime_status": health["model_runtime"]["status"],
        "persistence_status": health["persistence"]["status"],
        "checkpoint_enabled": True,
        "suggested_candidate_count": len(suggestion["candidates"]),
    }


async def _capture(args: argparse.Namespace) -> Path:
    suite_model = load_golden_suite(
        args.suite,
        require_reviewed=not args.allow_draft_suite,
    )
    suite = suite_model.model_dump()
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("run-%Y%m%dT%H%M%SZ")
    results = []

    api_key = os.getenv("ADVISOR_API_KEY", "").strip()
    headers = {"X-API-Key": api_key} if api_key else {}
    preflight: dict[str, Any] = {}
    async with httpx.AsyncClient(headers=headers) as client:
        if not args.skip_preflight:
            preflight = await _preflight_release(
                client, args.server, suite, args.timeout_seconds
            )
        for index, case in enumerate(suite["cases"], start=1):
            print(f"[{index}/{len(suite['cases'])}] {case['case_id']}")
            result = await _run_case(
                client,
                args.server,
                case,
                args.timeout_seconds,
                exercise_resume=(index == 1 and not args.skip_recovery_check),
            )
            results.append(result)
            if args.delay_seconds and index < len(suite["cases"]):
                await asyncio.sleep(args.delay_seconds)

    output = {
        "k": int(suite.get("k", 5)),
        "metadata": {
            "dataset_name": run_id,
            "display_name": f"真实运行 {run_id}（待审核）",
            "dataset_kind": "real_run",
            "annotation_status": "pending",
            "annotation_method": "none",
            "annotator": None,
            "is_publishable": False,
            "run_id": run_id,
            "model": os.getenv("RESEARCH_MODEL", "not-recorded"),
            "created_at": created_at.isoformat(),
            "golden_suite_name": suite["suite_name"],
            "golden_suite_version": suite["version"],
            "golden_suite_sha256": golden_suite_sha256(args.suite),
            "ground_truth_status": suite["ground_truth_status"],
            "ground_truth_reviewer": suite.get("reviewer"),
            "ground_truth_reviewed_at": suite.get("reviewed_at"),
            "candidate_suggestion_exercised": bool(preflight),
            "recovery_exercised": any(
                case.get("annotation_context", {}).get("recovery_exercised")
                for case in results
            ),
            "release_preflight": preflight,
            "notes": (
                f"suite={suite.get('suite_name', args.suite.name)}; "
                "supported_citations 和 task_success 必须独立人工审核。"
            ),
        },
        "cases": results,
    }

    output_path = args.output or (
        PROJECT_ROOT / "evals" / "runs" / f"{run_id}.pending.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture real Project Advisor observations for later review."
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--allow-draft-suite",
        action="store_true",
        help="Development only: capture a non-publishable run from draft labels.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip health/candidate checks; the resulting run cannot be publishable.",
    )
    parser.add_argument(
        "--skip-recovery-check",
        action="store_true",
        help="Skip checkpoint resume exercise; the resulting run cannot be publishable.",
    )
    args = parser.parse_args()
    output_path = asyncio.run(_capture(args))
    print(f"待审核运行结果：{output_path}")
    print(
        "下一步：运行 scripts/annotate_eval_results.py，"
        "不要直接将 pending 文件接入正式看板。"
    )


if __name__ == "__main__":
    main()
