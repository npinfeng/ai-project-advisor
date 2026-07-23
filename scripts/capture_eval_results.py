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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = PROJECT_ROOT / "evals" / "golden_cases.json"
URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}]+")


def _extract_urls(text: str) -> list[str]:
    return sorted({
        url.rstrip(".,;:!?，。；：！？\"'")
        for url in URL_PATTERN.findall(text)
    })


def _load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError("评测套件至少需要一个 case。")
    case_ids = [case.get("case_id") for case in cases]
    if None in case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("评测套件 case_id 必须存在且唯一。")
    for case in cases:
        required = (
            "question",
            "candidates",
            "relevant_documents",
            "expected_citations",
            "success_criteria",
        )
        missing = [key for key in required if not case.get(key)]
        if missing:
            raise ValueError(
                f"{case['case_id']} 缺少预设字段：{', '.join(missing)}"
            )
    return payload


async def _run_case(
    client: httpx.AsyncClient,
    server: str,
    case: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    final_report = ""
    diagnostics: dict[str, Any] = {}
    retrieved_evidences: list[dict[str, Any]] = []
    run_error: str | None = None

    payload = {
        "question": case["question"],
        "candidates": case["candidates"],
        "allow_clarification": False,
        "confirmed_candidates": True,
    }
    try:
        async with client.stream(
            "POST",
            f"{server.rstrip('/')}/api/advice/stream",
            json=payload,
            timeout=timeout_seconds,
        ) as response:
            response.raise_for_status()
            current_event = "message"
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:"):
                    event_data = json.loads(line[5:].strip())
                    if current_event == "result":
                        final_report = event_data.get("report", "")
                        diagnostics = event_data.get("diagnostics", {})
                        retrieved_evidences = event_data.get(
                            "retrieved_evidences", []
                        )
                    elif current_event == "error":
                        run_error = event_data.get("message", "工作流返回错误")
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
        "annotation_context": {
            "success_criteria": case["success_criteria"],
            "run_error": run_error,
            "report_preview": final_report[:1500],
        },
    }


async def _capture(args: argparse.Namespace) -> Path:
    suite = _load_suite(args.suite)
    suite_bytes = args.suite.read_bytes()
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("run-%Y%m%dT%H%M%SZ")
    results = []

    async with httpx.AsyncClient() as client:
        for index, case in enumerate(suite["cases"], start=1):
            print(f"[{index}/{len(suite['cases'])}] {case['case_id']}")
            result = await _run_case(
                client, args.server, case, args.timeout_seconds
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
            "notes": (
                f"suite={suite.get('suite_name', args.suite.name)}; "
                f"suite_sha256={hashlib.sha256(suite_bytes).hexdigest()}; "
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
    args = parser.parse_args()
    output_path = asyncio.run(_capture(args))
    print(f"待审核运行结果：{output_path}")
    print(
        "下一步：运行 scripts/annotate_eval_results.py，"
        "不要直接将 pending 文件接入正式看板。"
    )


if __name__ == "__main__":
    main()
