"""Integration tests for the real MCP boundary and offline evaluation suite."""

import asyncio
import json
from pathlib import Path

import pytest

from project_advisor.configuration import Configuration
from project_advisor.evaluation import EvaluationCase, evaluate_cases, evaluate_file
from project_advisor.mcp_client import clear_mcp_tool_cache, get_mcp_tools


def test_offline_evaluation_metrics():
    cases = [
        EvaluationCase(
            case_id="case-1",
            relevant_documents=["a", "b"],
            retrieved_documents=["a", "x"],
            expected_citations=["u1", "u2"],
            generated_citations=["u1", "bad"],
            supported_citations=["u1"],
            task_success=True,
            latency_ms=100,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.01,
        )
    ]

    report = evaluate_cases(cases, k=2)

    assert report.recall_at_k == 0.5
    assert report.precision_at_k == 0.5
    assert report.mrr == 1.0
    assert report.citation_accuracy == 0.5
    assert report.citation_coverage == 0.5
    assert report.task_success_rate == 1.0
    assert report.average_tokens == 15


def test_sample_evaluation_file():
    sample = Path(__file__).parents[1] / "evals" / "sample_results.json"
    report = evaluate_file(sample)

    assert report.case_count == 3
    assert 0 <= report.ndcg_at_k <= 1
    assert report.latency_p95_ms >= report.latency_p50_ms


def test_real_stdio_mcp_tool_call():
    async def run():
        clear_mcp_tool_cache()
        tools = await get_mcp_tools(Configuration(), force_refresh=True)
        cost_tool = next(tool for tool in tools if tool.name.endswith("estimate_llm_cost"))
        result = await cost_tool.ainvoke(
            {
                "monthly_requests": 1000,
                "average_input_tokens": 2000,
                "average_output_tokens": 500,
                "input_price_per_million": 1.0,
                "output_price_per_million": 2.0,
            }
        )
        payload = json.loads(result) if isinstance(result, str) else result
        assert payload["monthly_cost_usd"] == pytest.approx(3.0)
        assert payload["monthly_input_tokens"] == 2_000_000

    asyncio.run(run())
