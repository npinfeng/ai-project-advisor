"""Run real evaluation cases through the Project Advisor workflow and capture metrics.

Usage:
    C:/miniconda/envs/agent/python.exe scripts/run_eval_cases.py

This script invokes the actual LangGraph workflow for each test case, collects
real latency, token usage, cost, retrieved documents, and generated citations,
then writes the results in EvaluationCase-compatible JSON format.
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Add project root so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from project_advisor.configuration import Configuration
from project_advisor.graph import graph
from project_advisor.utils import get_today_str

URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}]+")

TEST_CASES: list[dict[str, Any]] = [
    {
        "case_id": "frontend-framework-2025",
        "question": (
            "我们团队在选型前端框架，候选有 React 19、Vue 3.5、Angular 19、Svelte 5 和 SolidJS。"
            "请比较它们的性能、生态成熟度、TypeScript 支持质量和学习成本。团队现有 React 经验。"
        ),
    },
    {
        "case_id": "database-choice-oltp",
        "question": (
            "我们需要为一个高并发 OLTP 系统选数据库，候选包括 PostgreSQL 17、MySQL 9、"
            "CockroachDB 和 TiDB。重点比较它们的事务一致性、水平扩展能力、运维复杂度和社区支持。"
        ),
    },
    {
        "case_id": "message-queue-async",
        "question": (
            "我们需要为异步事件处理选择消息队列，候选有 Apache Kafka、Redpanda、RabbitMQ、"
            "NATS 和 Apache Pulsar。请比较消息投递保证、吞吐能力、运维复杂度和生态成熟度。"
        ),
    },
    {
        "case_id": "python-web-framework",
        "question": (
            "我们准备用 Python 构建一个 REST API 服务，候选框架有 FastAPI、Litestar、"
            "Flask 3.x 和 Django 5。请评估它们的异步性能、类型安全、生态插件和文档质量。"
        ),
    },
    {
        "case_id": "observability-stack",
        "question": (
            "我们要为分布式服务建设可观测性体系，候选有 OpenTelemetry、Grafana、Honeycomb "
            "和 Datadog APM。请评估链路追踪互操作性、指标日志集成、私有化能力和运维成本。"
        ),
    },
    {
        "case_id": "container-orchestration",
        "question": (
            "团队需要为边缘和私有云选择容器编排平台，请比较 Kubernetes、Nomad、K3s 和 "
            "Docker Swarm 的可靠性、资源开销、生态支持和运维复杂度。"
        ),
    },
    {
        "case_id": "search-engine-selection",
        "question": (
            "我们需要同时支持商品搜索和日志搜索，请比较 Elasticsearch、Meilisearch、Typesense、"
            "OpenSearch 和 Quickwit 的相关性、扩展能力、运维复杂度与自托管能力。"
        ),
    },
    {
        "case_id": "graphql-vs-rest",
        "question": (
            "团队正在为多客户端平台选择 API 风格，请比较 GraphQL、REST 和 gRPC 在 Schema "
            "演进、缓存、可观测性、客户端复杂度和服务端运维风险方面的差异。"
        ),
    },
    {
        "case_id": "distributed-cache",
        "question": (
            "我们需要一个分布式缓存方案，候选有 Redis 8、Dragonfly、Microsoft Garnet 和 "
            "Memcached。比较它们的吞吐量、集群模式、持久化能力和运维工具链。"
        ),
    },
    {
        "case_id": "llm-orchestration-framework",
        "question": (
            "团队想选一个 LLM Agent 编排框架，候选有 LangGraph、CrewAI、Microsoft AutoGen、"
            "DSPy 和 Haystack 2.x。重点评估它们的工作流灵活性、多 Agent 协作能力、调试可观测性和社区活跃度。"
        ),
    },
]


def _collect_token_usage(value: Any, seen: set[int] | None = None) -> tuple[int, int]:
    """Recursively collect LangChain/OpenAI usage metadata from graph outputs."""
    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return 0, 0
    seen.add(obj_id)

    if isinstance(value, dict):
        direct = value.get("usage_metadata") or value.get("token_usage")
        if isinstance(direct, dict):
            inp = int(direct.get("input_tokens", direct.get("prompt_tokens", 0)) or 0)
            out = int(direct.get("output_tokens", direct.get("completion_tokens", 0)) or 0)
            return inp, out
        total_in = 0
        total_out = 0
        for child in value.values():
            ci, co = _collect_token_usage(child, seen)
            total_in += ci
            total_out += co
        return total_in, total_out

    if isinstance(value, (list, tuple, set)):
        total_in = 0
        total_out = 0
        for item in value:
            ci, co = _collect_token_usage(item, seen)
            total_in += ci
            total_out += co
        return total_in, total_out

    for attr in ("usage_metadata", "response_metadata"):
        meta = getattr(value, attr, None)
        if isinstance(meta, dict):
            token_data = meta.get("token_usage") or meta.get("usage")
            if isinstance(token_data, dict):
                inp = int(token_data.get("input_tokens", token_data.get("prompt_tokens", 0)) or 0)
                out = int(token_data.get("output_tokens", token_data.get("completion_tokens", 0)) or 0)
                return inp, out
    return 0, 0


def _extract_urls(text: str) -> list[str]:
    """Extract unique cleaned URLs from text."""
    urls = URL_PATTERN.findall(text)
    return sorted({u.rstrip(".,;:!?，。；：！？\"'") for u in urls})


def _extract_doc_ids(text: str) -> list[str]:
    """Extract document identifiers from tool outputs.

    Looks for patterns like document titles, headings, or source identifiers
    in the raw research notes.
    """
    ids: list[str] = []
    # Extract markdown headings as document identifiers
    heading_pattern = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    ids.extend(h.strip() for h in heading_pattern.findall(text))

    # Extract bold markers: **ProjectName**
    bold_pattern = re.compile(r"\*\*(.+?)\*\*")
    ids.extend(b.strip() for b in bold_pattern.findall(text))

    # Deduplicate while preserving order
    seen = set()
    unique: list[str] = []
    for did in ids:
        if did.lower() not in seen and len(did) > 1:
            seen.add(did.lower())
            unique.append(did)
    return unique[:30]  # cap to avoid noise


async def run_one_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run a single evaluation case through the workflow and collect all metrics."""
    print(f"\n{'='*60}")
    print(f"Running: {case['case_id']}")
    print(f"{'='*60}")

    config = {"configurable": {"allow_clarification": False}}
    started_at = time.perf_counter()
    total_input_tokens = 0
    total_output_tokens = 0
    final_report = ""
    all_raw_notes: list[str] = []

    try:
        async for update in graph.astream(
            {"messages": [HumanMessage(content=case["question"])]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():
                node_in, node_out = _collect_token_usage(node_output)
                total_input_tokens += node_in
                total_output_tokens += node_out

                if isinstance(node_output, dict):
                    # Collect raw research notes for document extraction
                    raw = node_output.get("raw_notes", [])
                    if isinstance(raw, list):
                        all_raw_notes.extend(raw)
                    # Capture final report
                    report = node_output.get("final_report", "")
                    if report:
                        final_report = report

        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        success = bool(final_report)

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        print(f"  ERROR: {exc}")
        success = False
        final_report = ""

    # Extract data from results
    generated_citations = _extract_urls(final_report)
    raw_text = " ".join(str(n) for n in all_raw_notes)
    retrieved_docs = _extract_doc_ids(raw_text)

    # Calculate cost (DeepSeek chat pricing roughly $0.27/$1.10 per 1M tokens)
    # But since .env may have custom pricing, use the Configuration class
    cfg = Configuration()
    cost = (
        total_input_tokens * cfg.input_price_per_million
        + total_output_tokens * cfg.output_price_per_million
    ) / 1_000_000 if (cfg.input_price_per_million or cfg.output_price_per_million) else 0.0

    # For annotation fields: set defaults based on what the system produced.
    # These need HUMAN REVIEW before being treated as ground truth.
    result = {
        "case_id": case["case_id"],
        "relevant_documents": retrieved_docs.copy(),
        "retrieved_documents": retrieved_docs.copy(),
        "expected_citations": generated_citations.copy(),
        "generated_citations": generated_citations.copy(),
        "supported_citations": generated_citations.copy(),
        "task_success": success,
        "latency_ms": elapsed_ms,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cost_usd": round(cost, 6),
        # Metadata for human reviewer
        "_report_preview": final_report[:500] if final_report else "",
        "_note": "relevant_documents, expected_citations, supported_citations, task_success are auto-populated defaults — human review required",
    }

    print(f"  Latency:   {elapsed_ms}ms ({elapsed_ms/1000:.1f}s)")
    print(f"  Tokens:    {total_input_tokens} in / {total_output_tokens} out = {total_input_tokens + total_output_tokens} total")
    print(f"  Cost:      ${cost:.6f}")
    print(f"  Documents: {len(retrieved_docs)} retrieved")
    print(f"  Citations: {len(generated_citations)} generated")
    print(f"  Success:   {success}")

    return result


async def main() -> None:
    print(f"Project Advisor — Real Evaluation Runner")
    print(f"Date: {get_today_str()}")
    print(f"Test cases: {len(TEST_CASES)}")

    results = []
    for i, case in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] ", end="")
        result = await run_one_case(case)
        results.append(result)
        # Small delay between runs to avoid rate limiting
        if i < len(TEST_CASES) - 1:
            await asyncio.sleep(2)

    # Build output payload
    output = {
        "k": 5,
        "_metadata": {
            "run_date": get_today_str(),
            "description": "Real workflow execution results. Annotation fields (relevant_documents, expected_citations, supported_citations, task_success) are auto-populated defaults and require human review.",
            "model": "deepseek:deepseek-chat",
            "total_runs": len(results),
        },
        "cases": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ],
    }

    output_path = Path(__file__).resolve().parents[1] / "evals" / "real_run_results.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Results written to: {output_path}")
    print(f"{'='*60}")

    # Calculate summary stats
    successes = sum(1 for r in results if r["task_success"])
    latencies = [r["latency_ms"] for r in results]
    total_tokens = [r["input_tokens"] + r["output_tokens"] for r in results]
    total_cost = sum(r["cost_usd"] for r in results)

    print(f"\nSummary:")
    print(f"  Success rate: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
    print(f"  Latency range: {min(latencies)/1000:.1f}s — {max(latencies)/1000:.1f}s")
    print(f"  Avg tokens: {sum(total_tokens)/len(total_tokens):.0f}")
    print(f"  Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
