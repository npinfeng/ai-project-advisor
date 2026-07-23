"""Run evaluation cases through the running web server and capture real metrics.

Usage (server must be running):
    C:/miniconda/envs/agent/python.exe scripts/capture_eval_results.py
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

SERVER = "http://127.0.0.1:8000"
URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}]+")

TEST_CASES = [
    {
        "case_id": "frontend-framework-2025",
        "question": (
            "Compare React 19, Vue 3.5, Angular 19, Svelte 5, and SolidJS for a new frontend project. "
            "Evaluate performance, ecosystem maturity, TypeScript support, and learning curve. "
            "Team has prior React experience."
        ),
        "candidates": ["React", "Vue", "Angular", "Svelte", "SolidJS"],
    },
    {
        "case_id": "database-choice-oltp",
        "question": (
            "Select a database for a high-concurrency OLTP system. Compare PostgreSQL 17, MySQL 9, "
            "CockroachDB, and TiDB. Focus on transaction consistency, horizontal scaling, "
            "operational complexity, and community support."
        ),
        "candidates": ["PostgreSQL", "MySQL", "CockroachDB", "TiDB"],
    },
    {
        "case_id": "message-queue-async",
        "question": (
            "Select a message queue for asynchronous event processing. Compare Apache Kafka, "
            "Redpanda, RabbitMQ, NATS, and Apache Pulsar. Evaluate delivery guarantees, "
            "throughput, operational complexity, and ecosystem maturity."
        ),
        "candidates": ["Kafka", "Redpanda", "RabbitMQ", "NATS", "Pulsar"],
    },
    {
        "case_id": "python-web-framework",
        "question": (
            "Choose a Python web framework for a new REST API service. Compare FastAPI, Litestar, "
            "Flask 3.x, and Django 5. Evaluate async performance, type safety, plugin ecosystem, "
            "and documentation quality."
        ),
        "candidates": ["FastAPI", "Litestar", "Flask", "Django"],
    },
    {
        "case_id": "observability-stack",
        "question": (
            "Choose an observability stack for distributed services. Compare OpenTelemetry, "
            "Grafana, Honeycomb, and Datadog APM. Evaluate tracing interoperability, metrics "
            "and logs integration, self-hosting, and operational cost."
        ),
        "candidates": ["OpenTelemetry", "Grafana", "Honeycomb", "Datadog"],
    },
    {
        "case_id": "container-orchestration",
        "question": (
            "Select container orchestration for an edge and private-cloud deployment. Compare "
            "Kubernetes, Nomad, K3s, and Docker Swarm for reliability, resource overhead, "
            "ecosystem support, and operating complexity."
        ),
        "candidates": ["Kubernetes", "Nomad", "K3s", "Docker Swarm"],
    },
    {
        "case_id": "search-engine-selection",
        "question": (
            "Choose a search engine for product and log search. Compare Elasticsearch, "
            "Meilisearch, Typesense, OpenSearch, and Quickwit for relevance, scale, "
            "operational complexity, and self-hosting."
        ),
        "candidates": ["Elasticsearch", "Meilisearch", "Typesense", "OpenSearch", "Quickwit"],
    },
    {
        "case_id": "graphql-vs-rest",
        "question": (
            "Choose an API style for a multi-client platform. Compare GraphQL with REST and "
            "gRPC, including schema evolution, caching, observability, client complexity, "
            "and server-side operational risks."
        ),
        "candidates": ["GraphQL", "REST", "gRPC"],
    },
    {
        "case_id": "distributed-cache",
        "question": (
            "Select a distributed caching solution. Compare Redis 8, Dragonfly, Microsoft Garnet, "
            "and Memcached. Evaluate throughput, clustering modes, persistence capabilities, "
            "and operational tooling."
        ),
        "candidates": ["Redis", "Dragonfly", "Garnet", "Memcached"],
    },
    {
        "case_id": "llm-orchestration-framework",
        "question": (
            "Choose an LLM agent orchestration framework. Compare LangGraph, CrewAI, "
            "Microsoft AutoGen, DSPy, and Haystack 2.x. Evaluate workflow flexibility, "
            "multi-agent collaboration, debugging observability, and community activity."
        ),
        "candidates": ["LangGraph", "CrewAI", "AutoGen", "DSPy", "Haystack"],
    },
]


def _extract_urls(text: str) -> list[str]:
    urls = URL_PATTERN.findall(text)
    return sorted({u.rstrip(".,;:!?，。；：！？\"'") for u in urls})


def _extract_doc_ids(text: str) -> list[str]:
    """Extract document identifiers from report text."""
    ids: list[str] = []
    heading_pattern = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    ids.extend(h.strip() for h in heading_pattern.findall(text))
    bold_pattern = re.compile(r"\*\*(.+?)\*\*")
    ids.extend(b.strip() for b in bold_pattern.findall(text))
    seen = set()
    unique: list[str] = []
    for did in ids:
        if did.lower() not in seen and len(did) > 1:
            seen.add(did.lower())
            unique.append(did)
    return unique[:30]


async def run_one_case(
    client: httpx.AsyncClient,
    case: dict,
    index: int,
    total: int,
) -> dict:
    """Send one request to the SSE endpoint and capture the full result."""
    print(f"\n[{index}/{total}] {case['case_id']}")
    print(f"  Candidates: {case['candidates']}")

    payload = {
        "question": case["question"],
        "candidates": case["candidates"],
        "allow_clarification": False,
    }

    started_at = time.perf_counter()
    event_count = 0

    try:
        async with client.stream(
            "POST",
            f"{SERVER}/api/advice/stream",
            json=payload,
            timeout=180.0,
        ) as response:
            final_report = ""
            diagnostics = {}
            current_event = None

            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: "):
                    event_count += 1
                    data = json.loads(line[6:])
                    event_type = current_event or "unknown"

                    if event_type == "result":
                        final_report = data.get("report", "")
                        diagnostics = data.get("diagnostics", {})
                    elif event_type == "error":
                        print(f"  ERROR event: {data.get('message', 'unknown')}")
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        print(f"  HTTP/stream error: {exc}")
        return {
            "case_id": case["case_id"],
            "relevant_documents": [],
            "retrieved_documents": [],
            "expected_citations": [],
            "generated_citations": [],
            "supported_citations": [],
            "task_success": False,
            "latency_ms": elapsed_ms,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }

    elapsed_ms = diagnostics.get("total_duration_ms", round((time.perf_counter() - started_at) * 1000))
    token_usage = diagnostics.get("token_usage", {})
    input_tokens = token_usage.get("input_tokens", 0)
    output_tokens = token_usage.get("output_tokens", 0)
    cost = diagnostics.get("estimated_cost_usd", 0.0)

    generated_citations = _extract_urls(final_report)
    retrieved_docs = _extract_doc_ids(final_report)

    print(f"  Events: {event_count} | Latency: {elapsed_ms}ms | Tokens: {input_tokens}+{output_tokens} | Cost: ${cost}")

    return {
        "case_id": case["case_id"],
        "relevant_documents": retrieved_docs.copy(),
        "retrieved_documents": retrieved_docs.copy(),
        "expected_citations": generated_citations.copy(),
        "generated_citations": generated_citations.copy(),
        "supported_citations": generated_citations.copy(),
        "task_success": bool(final_report),
        "latency_ms": elapsed_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }


async def main():
    print("=" * 60)
    print("Project Advisor — Real Evaluation Data Capture")
    print(f"Server: {SERVER}")
    print(f"Cases: {len(TEST_CASES)}")
    print("=" * 60)

    total = len(TEST_CASES)
    results = []

    async with httpx.AsyncClient() as client:
        for i, case in enumerate(TEST_CASES, 1):
            result = await run_one_case(client, case, i, total)
            results.append(result)
            if i < total:
                print("  Waiting 3s before next case...")
                await asyncio.sleep(3)

    # Clean output — remove annotation fields (auto-populated, need human review)
    clean_results = []
    for r in results:
        clean_results.append({
            "case_id": r["case_id"],
            "relevant_documents": [],
            "retrieved_documents": r["retrieved_documents"],
            "expected_citations": [],
            "generated_citations": r["generated_citations"],
            "supported_citations": [],
            "task_success": None,
            "latency_ms": r["latency_ms"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "cost_usd": r["cost_usd"],
        })

    output = {
        "k": 5,
        "_metadata": {
            "description": (
                "Real LangGraph workflow execution results via SSE API. "
                "Automated fields (latency, tokens, cost, retrieved_documents, generated_citations) "
                "are from actual runs. Annotation fields (relevant_documents, expected_citations, "
                "supported_citations, task_success) are empty and REQUIRE HUMAN REVIEW."
            ),
            "model": "deepseek:deepseek-chat",
            "total_runs": len(clean_results),
        },
        "cases": clean_results,
    }

    output_path = Path(__file__).resolve().parents[1] / "evals" / "real_run_results.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Results written to: {output_path}")
    print(f"{'=' * 60}")

    successes = sum(1 for r in results if r["task_success"])
    costs = [r["cost_usd"] for r in results]
    tokens = [r["input_tokens"] + r["output_tokens"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    print(f"\nSummary:")
    print(f"  Case count: {len(results)}")
    print(f"  Success: {successes}/{len(results)}")
    print(f"  Latency: {min(latencies)/1000:.1f}s — {max(latencies)/1000:.1f}s")
    print(f"  Avg tokens: {sum(tokens)/len(tokens):.0f}")
    print(f"  Total cost: ${sum(costs):.4f}")
    print(f"\n⚠️  NEXT STEP: Open evals/real_run_results.json and manually fill in:")
    print(f"    - relevant_documents: which retrieved docs are actually relevant")
    print(f"    - expected_citations: which citations should have appeared")
    print(f"    - supported_citations: which generated citations have real evidence")
    print(f"    - task_success: true/false for each case")


if __name__ == "__main__":
    asyncio.run(main())
