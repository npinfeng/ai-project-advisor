"""Tests for the connected workflow topology and web demo."""

import importlib

from fastapi.testclient import TestClient

from project_advisor.graph import (
    researcher_subgraph_doc,
    researcher_subgraph_repo,
    supervisor_subgraph,
)
from project_advisor.schemas.evidence import ProjectScore, ReviewResult


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


class FakeGraph:
    async def astream(self, *args, **kwargs):
        yield {"clarify_requirements": {"next": "plan_evaluation"}}
        yield {"plan_evaluation": {"candidates": ["LangGraph"]}}
        yield {"research_supervisor": {"notes": ["Evidence"]}}
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
