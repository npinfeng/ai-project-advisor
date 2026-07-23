"""Tests for the connected workflow topology and web demo."""

import importlib
import json

from fastapi.testclient import TestClient

from project_advisor.graph import (
    researcher_subgraph_doc,
    researcher_subgraph_repo,
    supervisor_subgraph,
)
from project_advisor.agents.reviewer import _bind_scores_to_evidence
from project_advisor.schemas.evidence import (
    CandidateRecommendation,
    Evidence,
    ProjectScore,
    Requirements,
    ReviewResult,
)
from project_advisor.state import ResearchPlan
from project_advisor.tools.evidence_factory import build_evidences_from_tool_result


def test_research_subgraphs_have_execution_loops():
    supervisor_mermaid = supervisor_subgraph.get_graph().draw_mermaid()
    repository_mermaid = researcher_subgraph_repo.get_graph().draw_mermaid()
    documentation_mermaid = researcher_subgraph_doc.get_graph().draw_mermaid()

    assert "supervisor -.-> supervisor_tools" in supervisor_mermaid
    assert "supervisor_tools -.-> supervisor" in supervisor_mermaid
    assert "repository_analyst -.-> analyst_tools" in repository_mermaid
    assert "analyst_tools -.-> compress_research" in repository_mermaid
    assert "documentation_researcher -.-> doc_tools" in documentation_mermaid
    assert "doc_tools -.-> compress_research" in documentation_mermaid


def test_review_result_is_structured():
    result = ReviewResult(
        analysis="LangGraph is the best fit for this stateful workflow.",
        scores=[
            ProjectScore(
                project_name="LangGraph",
                feature_match=9,
                engineering_reliability=8,
                community_and_maintenance=8,
                documentation_quality=8,
                learning_cost=6,
                extensibility=9,
                deployment_cost=7,
            )
        ],
        evidence_gaps=["No production latency benchmark was found."],
    )

    assert result.scores[0].project_name == "LangGraph"
    assert result.evidence_gaps


def test_tool_output_is_normalized_and_score_is_bound_to_evidence():
    evidences = build_evidences_from_tool_result(
        tool_name="github_get_repo",
        args={"github_url": "https://github.com/langchain-ai/langgraph"},
        result={"stars": 100, "url": "https://github.com/langchain-ai/langgraph"},
        project_name="LangGraph",
        research_topic="维护状态",
    )
    scores, gaps = _bind_scores_to_evidence(
        [ProjectScore(project_name="LangGraph", feature_match=9)],
        ["LangGraph", "CrewAI"],
        evidences,
    )

    assert evidences[0].evidence_id.startswith("ev_")
    assert scores[0].evidence_ids == [evidences[0].evidence_id]
    assert scores[0].source_urls == ["https://github.com/langchain-ai/langgraph"]
    assert scores[1].project_name == "CrewAI"
    assert scores[1].evidence_confidence == "insufficient"
    assert gaps


class FakeGraph:
    async def astream(self, *args, **kwargs):
        yield {"clarify_requirements": {"next": "plan_evaluation"}}
        yield {"plan_evaluation": {"candidates": ["LangGraph"]}}
        yield {
            "research_supervisor": {
                "notes": ["Evidence"],
                "evidences": [
                    Evidence(
                        source_url="https://docs.example.com/langgraph",
                        source_type="official_documentation",
                        project_name="LangGraph",
                        content="Durable execution evidence.",
                        relevance="checkpoint",
                        retrieved_at="2026-07-23T08:00:00+00:00",
                    )
                ],
            }
        }
        yield {
            "review_and_score": {
                "scores": [
                    ProjectScore(
                        project_name="LangGraph",
                        feature_match=9,
                        engineering_reliability=8,
                        community_and_maintenance=8,
                        documentation_quality=8,
                        learning_cost=6,
                        extensibility=9,
                        deployment_cost=7,
                        weighted_total=8.15,
                    )
                ]
            }
        }
        yield {"generate_report": {"final_report": "# Test report"}}


def test_web_app_and_sse_stream(monkeypatch):
    app_module = importlib.import_module("project_advisor.app")
    monkeypatch.setattr(app_module, "graph", FakeGraph())
    client = TestClient(app_module.app)

    health_response = client.get("/api/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"

    page_response = client.get("/")
    assert page_response.status_code == 200
    assert "Project Advisor" in page_response.text
    assert "开始深度评估" in page_response.text
    assert "系统评测看板" in page_response.text
    assert "本次运行诊断" in page_response.text

    evaluation_response = client.get("/api/evaluation")
    assert evaluation_response.status_code == 200
    evaluation_payload = evaluation_response.json()
    assert evaluation_payload["source"] == "real_results.json"
    assert evaluation_payload["metadata"]["dataset_kind"] == "synthetic"
    assert evaluation_payload["metadata"]["is_publishable"] is False
    assert evaluation_payload["report"]["case_count"] == 10
    assert evaluation_payload["report"]["latency_p95_ms"] >= evaluation_payload["report"]["latency_p50_ms"]

    with client.stream(
        "POST",
        "/api/advice/stream",
        json={
            "question": "请评估适合 Python 多智能体应用的框架。",
            "candidates": ["LangGraph"],
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: started" in body
    assert "event: progress" in body
    assert "event: result" in body
    assert "# Test report" in body
    assert "8.15" in body
    assert '"candidate_count": 1' in body
    assert '"stage_duration_ms"' in body
    assert '"stage_durations_ms"' in body
    assert '"token_usage"' in body
    assert '"retrieved_evidences"' in body
    assert '"evidence_id"' in body


def test_pending_evaluation_is_visible_without_fake_metrics(
    monkeypatch, tmp_path
):
    app_module = importlib.import_module("project_advisor.app")
    pending_file = tmp_path / "pending.json"
    pending_file.write_text(
        json.dumps({
            "k": 5,
            "metadata": {
                "dataset_name": "run-pending",
                "display_name": "真实运行（待审核）",
                "dataset_kind": "real_run",
                "annotation_status": "pending",
                "annotation_method": "none",
                "is_publishable": False,
            },
            "cases": [{
                "case_id": "case-1",
                "latency_ms": 10,
                "task_success": None,
            }],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVALUATION_FILE", str(pending_file))

    response = TestClient(app_module.app).get("/api/evaluation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["annotation_status"] == "pending"
    assert payload["report"] is None
    assert "等待独立人工审核" in payload["status_message"]


def test_candidate_suggestion_preview(monkeypatch):
    app_module = importlib.import_module("project_advisor.app")
    plan = ResearchPlan(
        research_brief="评估 Python Agent 框架。",
        requirements=Requirements(
            language="Python",
            required_features=["multi_agent", "self_hosted"],
        ),
        candidates=[
            CandidateRecommendation(
                name="LangGraph",
                github_url="https://github.com/langchain-ai/langgraph",
                reason="状态管理和持久化能力匹配。",
            ),
            CandidateRecommendation(
                name="CrewAI",
                github_url="https://github.com/crewAIInc/crewAI",
                reason="多智能体角色协作开箱即用。",
            ),
        ],
        evaluation_focus=["工程可靠性", "学习成本"],
    )

    async def fake_generate_candidate_plan(question):
        assert "Python" in question
        return plan

    monkeypatch.setattr(
        app_module, "_generate_candidate_plan", fake_generate_candidate_plan
    )
    client = TestClient(app_module.app)
    response = client.post(
        "/api/candidates/suggest",
        json={"question": "请推荐适合 Python 多智能体应用的开源框架。"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["requirements"]["language"] == "Python"
    assert len(payload["candidates"]) == 2
    assert payload["candidates"][0]["reason"]
