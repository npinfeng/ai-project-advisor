"""Integration tests for the real MCP boundary and offline evaluation suite."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from project_advisor.configuration import Configuration
from project_advisor.evaluation import (
    EvaluationCase,
    EvaluationMetadata,
    evaluate_cases,
    evaluate_file,
    load_evaluation_bundle,
    load_evaluation_file,
    normalize_evaluation_identifier,
)
from project_advisor.mcp_client import (
    clear_mcp_tool_cache,
    get_mcp_diagnostics,
    get_mcp_tools,
)
from scripts.annotate_eval_results import finalize_annotations
from scripts.capture_eval_results import _run_case


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


def test_extended_evaluation_file():
    extended = Path(__file__).parents[1] / "evals" / "real_results.json"
    bundle = load_evaluation_bundle(extended)
    cases, k = load_evaluation_file(extended)
    report = evaluate_file(extended)

    assert len(cases) == 10
    assert len({case.case_id for case in cases}) == 10
    assert k == 5
    assert report.case_count == 10
    assert 0 <= report.task_success_rate <= 1
    assert report.dataset_kind == "synthetic"
    assert report.is_publishable is False
    assert report.quality_warnings
    assert bundle.metadata.annotation_method == "fixture"
    with pytest.raises(ValueError, match="不是经过审核"):
        evaluate_file(extended, require_publishable=True)


def test_pending_and_unverified_baselines_are_rejected():
    pending = EvaluationMetadata(
        dataset_name="pending-run",
        dataset_kind="real_run",
        annotation_status="pending",
    )
    case = EvaluationCase(
        case_id="pending-case",
        relevant_documents=["https://docs.example.com/a/"],
        retrieved_documents=["https://docs.example.com/a"],
        expected_citations=["https://docs.example.com/a"],
        generated_citations=["https://docs.example.com/a/"],
        task_success=None,
        latency_ms=10,
    )
    with pytest.raises(ValueError, match="pending"):
        evaluate_cases([case], metadata=pending)

    case.task_success = True
    invalid_publishable = pending.model_copy(update={
        "annotation_status": "reviewed",
        "is_publishable": True,
    })
    with pytest.raises(ValueError, match="独立人工审核"):
        evaluate_cases([case], metadata=invalid_publishable)


def test_url_normalization_and_duplicate_retrieval_do_not_inflate_metrics():
    case = EvaluationCase(
        case_id="url-normalization",
        relevant_documents=["https://Docs.Example.com/guide/"],
        retrieved_documents=[
            "https://docs.example.com/guide",
            "https://docs.example.com/guide/",
        ],
        expected_citations=["https://docs.example.com/guide/"],
        generated_citations=["https://docs.example.com/guide"],
        supported_citations=["https://docs.example.com/guide/"],
        task_success=True,
        latency_ms=10,
    )
    report = evaluate_cases([case], k=2)

    assert normalize_evaluation_identifier(
        "https://Docs.Example.com/guide/#section"
    ) == "https://docs.example.com/guide"
    assert report.recall_at_k == 1.0
    assert report.precision_at_k == 0.5
    assert report.ndcg_at_k == 1.0
    assert report.citation_accuracy == 1.0


def test_golden_suite_is_defined_before_execution():
    suite_path = Path(__file__).parents[1] / "evals" / "golden_cases.json"
    payload = json.loads(suite_path.read_text(encoding="utf-8"))

    assert payload["ground_truth_status"] == "draft_human_review_required"
    assert len(payload["cases"]) >= 6
    for case in payload["cases"]:
        assert case["relevant_documents"]
        assert case["expected_citations"]
        assert case["success_criteria"]


def test_independent_annotation_is_required_before_publishable_evaluation():
    pending_payload = {
        "k": 5,
        "metadata": {
            "dataset_name": "run-test",
            "display_name": "真实运行 run-test（待审核）",
            "dataset_kind": "real_run",
            "annotation_status": "pending",
            "annotation_method": "none",
            "annotator": None,
            "is_publishable": False,
        },
        "cases": [{
            "case_id": "case-1",
            "relevant_documents": ["https://docs.example.com/guide"],
            "retrieved_documents": ["https://docs.example.com/guide"],
            "retrieved_evidence_ids": ["ev_123"],
            "expected_citations": ["https://docs.example.com/guide"],
            "generated_citations": ["https://docs.example.com/guide"],
            "supported_citations": [],
            "task_success": None,
            "latency_ms": 100,
            "annotation_context": {"success_criteria": "包含明确结论"},
        }],
    }
    reviewed = finalize_annotations(
        pending_payload,
        {
            "case-1": {
                "supported_citations": ["https://docs.example.com/guide"],
                "task_success": True,
                "review_notes": "人工核对通过",
            }
        },
        annotator="reviewer-a",
        ground_truth_confirmed=True,
    )
    metadata = EvaluationMetadata.model_validate(reviewed["metadata"])
    cases = [EvaluationCase.model_validate(item) for item in reviewed["cases"]]
    report = evaluate_cases(cases, metadata=metadata)

    assert metadata.annotation_status == "reviewed"
    assert metadata.annotation_method == "independent_human"
    assert metadata.is_publishable is True
    assert report.is_publishable is True


def test_annotation_cannot_claim_support_for_an_unproduced_citation():
    pending_payload = {
        "metadata": {
            "dataset_kind": "real_run",
            "annotation_status": "pending",
        },
        "cases": [{
            "case_id": "case-1",
            "generated_citations": ["https://docs.example.com/actual"],
        }],
    }
    with pytest.raises(ValueError, match="未生成"):
        finalize_annotations(
            pending_payload,
            {
                "case-1": {
                    "supported_citations": ["https://docs.example.com/fabricated"],
                    "task_success": True,
                }
            },
            annotator="reviewer-a",
            ground_truth_confirmed=False,
        )


def test_capture_records_sse_evidence_without_auto_labeling():
    sse_body = """event: result
data: {"report":"结论 https://docs.example.com/generated","retrieved_evidences":[{"evidence_id":"ev_real","source_url":"https://docs.example.com/retrieved"}],"diagnostics":{"total_duration_ms":25,"token_usage":{"input_tokens":10,"output_tokens":4},"estimated_cost_usd":0.01}}

"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/advice/stream"
        return httpx.Response(
            200,
            text=sse_body,
            headers={"content-type": "text/event-stream"},
        )

    async def capture() -> dict:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await _run_case(
                client,
                "http://testserver",
                {
                    "case_id": "case-1",
                    "question": "测试",
                    "candidates": ["Project A"],
                    "relevant_documents": ["https://gold.example.com/relevant"],
                    "expected_citations": ["https://gold.example.com/citation"],
                    "success_criteria": "人工判断",
                },
                10,
            )

    result = asyncio.run(capture())

    assert result["retrieved_documents"] == [
        "https://docs.example.com/retrieved"
    ]
    assert result["retrieved_evidence_ids"] == ["ev_real"]
    assert result["relevant_documents"] == [
        "https://gold.example.com/relevant"
    ]
    assert result["generated_citations"] == [
        "https://docs.example.com/generated"
    ]
    assert result["supported_citations"] == []
    assert result["task_success"] is None


def test_mcp_diagnostics_when_disabled():
    diagnostics = get_mcp_diagnostics(Configuration(enable_local_mcp=False))

    assert diagnostics == {
        "status": "disabled",
        "server_count": 0,
        "tool_count": 0,
        "error_type": None,
    }


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
        if isinstance(result, str):
            payload = json.loads(result)
        elif isinstance(result, list):
            payload = json.loads(result[0]["text"])
        else:
            payload = result
        assert payload["monthly_cost_usd"] == pytest.approx(3.0)
        assert payload["monthly_input_tokens"] == 2_000_000
        diagnostics = get_mcp_diagnostics(Configuration())
        assert diagnostics["status"] == "connected"
        assert diagnostics["tool_count"] == 2

    asyncio.run(run())
