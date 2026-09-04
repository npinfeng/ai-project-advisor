"""Tests for the connected workflow topology and web demo."""

import asyncio
import importlib
import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from project_advisor.graph import (
    detect_evidence_gaps,
    evidence_coverage,
    graph,
    researcher_subgraph_doc,
    researcher_subgraph_repo,
)
from project_advisor.agents.planner import build_research_tasks
from project_advisor.agents.reviewer import (
    _bind_scores_to_evidence,
    _consolidate_review_gaps,
)
from project_advisor.schemas.evidence import (
    CandidateRecommendation,
    Evidence,
    ProjectScore,
    Requirements,
    ReviewResult,
)
from project_advisor.state import ResearchPlan
from project_advisor.tools.evidence_factory import build_evidences_from_tool_result
from project_advisor.tools.constraint_analyzer import analyze_feasibility


def test_research_subgraphs_have_execution_loops():
    workflow_mermaid = graph.get_graph().draw_mermaid()
    repository_mermaid = researcher_subgraph_repo.get_graph().draw_mermaid()
    documentation_mermaid = researcher_subgraph_doc.get_graph().draw_mermaid()

    assert "research_supervisor" not in workflow_mermaid
    assert "parallel_research" in workflow_mermaid
    assert "await_clarification" in workflow_mermaid
    assert "confirm_plan" in workflow_mermaid
    assert "evidence_coverage" in workflow_mermaid
    assert "supplemental_research" in workflow_mermaid
    assert "repository_analyst -.-> analyst_tools" in repository_mermaid
    assert "summarize_evidence" in repository_mermaid
    assert "documentation_researcher -.-> doc_tools" in documentation_mermaid
    assert "summarize_evidence" in documentation_mermaid


def test_planner_expands_typed_tasks_and_gap_round_is_bounded():
    candidates = [CandidateRecommendation(
        name="LangGraph",
        github_url="https://github.com/langchain-ai/langgraph",
        reason="匹配状态化工作流。",
    )]
    tasks = build_research_tasks(candidates, ["人工审批"])

    assert [task.track for task in tasks] == ["repository", "documentation"]
    assert all(task.project_name == "LangGraph" for task in tasks)
    assert tasks[0].github_url == "https://github.com/langchain-ai/langgraph"

    gaps = detect_evidence_gaps(["LangGraph"], [])
    assert {gap.track for gap in gaps} == {"repository", "documentation"}
    first_gate = evidence_coverage({
        "candidates": ["LangGraph"],
        "candidate_recommendations": candidates,
        "evaluation_focus": ["人工审批"],
        "evidences": [],
        "supplemental_round_used": False,
    })
    assert first_gate["next"] == "supplemental_research"
    assert first_gate["supplemental_round_used"] is True
    assert all(task.round == 1 for task in first_gate["research_tasks"])

    final_gate = evidence_coverage({
        "candidates": ["LangGraph"],
        "evidences": [],
        "supplemental_round_used": True,
    })
    assert final_gate["next"] == "review_and_score"


def test_framework_rag_integration_satisfies_hard_constraint():
    report = analyze_feasibility(
        {"required_features": ["rag"]},
        ["LangGraph", "AutoGen", "CrewAI"],
    )

    assert report.is_feasible is True
    assert not any("不支持 'rag'" in item.description for item in report.violations)


def test_evidence_coverage_targets_missing_hard_capabilities_without_calling_them_unsupported():
    rag_evidence = Evidence(
        source_url="https://docs.example.com/langgraph/rag",
        source_type="official_documentation",
        project_name="LangGraph",
        content="Build RAG with a retriever and vector store inside a LangGraph workflow.",
        relevance="feature_match",
        retrieved_at="2026-09-04T00:00:00+00:00",
    )

    gaps = detect_evidence_gaps(
        ["LangGraph"],
        [rag_evidence],
        {"LangGraph": None},
        ["rag", "human_in_the_loop"],
    )

    assert len(gaps) == 1
    assert gaps[0].track == "documentation"
    assert "human_in_the_loop" in gaps[0].reason
    assert "rag" not in gaps[0].reason.split("：", 1)[1].split("。", 1)[0]
    assert "证据不足不代表框架不支持" in gaps[0].reason


def test_manual_execution_plan_does_not_create_a_planner_model(monkeypatch):
    planner_module = importlib.import_module("project_advisor.agents.planner")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("执行阶段不应重新调用 Planner 模型")

    monkeypatch.setattr(planner_module, "create_chat_model", fail_if_called)
    output = asyncio.run(planner_module.plan_evaluation({
        "messages": [HumanMessage(content="比较 LangGraph 与 CrewAI。")],
        "confirmed_candidates": ["LangGraph", "CrewAI"],
    }, {}))

    assert output["next"] == "confirm_plan"
    assert len(output["research_tasks"]) == 2
    assert all(
        "比较 LangGraph 与 CrewAI" in task.research_topic
        for task in output["research_tasks"]
    )
    assert {task.track for task in output["research_tasks"]} == {"documentation"}


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


def test_final_report_is_deterministic_and_does_not_create_a_model(monkeypatch):
    reviewer_module = importlib.import_module("project_advisor.agents.reviewer")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("确定性报告节点不应创建模型")

    monkeypatch.setattr(reviewer_module, "create_chat_model", fail_if_called)
    evidence = Evidence(
        source_url="https://docs.example.com/langgraph",
        source_type="official_documentation",
        project_name="LangGraph",
        content="Durable execution is documented.",
        relevance="feature_match",
        retrieved_at="2026-07-23T08:00:00+00:00",
    )
    output = asyncio.run(reviewer_module.generate_report({
        "research_brief": "评估状态化 Agent 框架。",
        "candidates": ["LangGraph"],
        "scores": [ProjectScore(
            project_name="LangGraph",
            feature_match=9,
            weighted_total=8.5,
            evidence_ids=[evidence.evidence_id],
            source_urls=[evidence.source_url],
        )],
        "evidences": [evidence],
        "review_analysis": "证据支持状态化执行能力。",
        "review_evidence_gaps": [],
    }, {}))

    assert output["final_report"].startswith("# 技术选型评估报告")
    assert evidence.evidence_id in output["final_report"]
    assert evidence.source_url in output["final_report"]


def test_context_compression_is_reported_as_process_note_not_evidence_gap(monkeypatch):
    reviewer_module = importlib.import_module("project_advisor.agents.reviewer")
    monkeypatch.setattr(
        reviewer_module,
        "create_chat_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected model call")),
    )
    output = asyncio.run(reviewer_module.generate_report({
        "research_brief": "评估框架。",
        "candidates": [],
        "scores": [],
        "evidences": [],
        "review_analysis": "暂无。",
        "review_evidence_gaps": [],
        "context_budget": {
            "compressed": True,
            "omitted_or_duplicate_count": 4,
        },
    }, {}))

    report = output["final_report"]
    gap_section = report.split("## 7. 证据缺口", 1)[1].split("## 8.", 1)[0]
    assert "上下文预算" not in gap_section
    assert "## 8. 研究过程说明" in report
    assert "不代表项目能力或证据缺失" in report


def test_review_gaps_are_deduplicated_and_bounded_per_candidate():
    gaps = [
        "LangGraph: 缺少 RAG 官方文档。",
        "LangGraph: 缺少 RAG 官方文档。",
        "LangGraph: 缺少 HITL 官方文档。",
        "LangGraph: 缺少部署文档。",
        "LangGraph: 缺少基准数据。",
        "CrewAI: 缺少持久化文档。",
    ]

    consolidated = _consolidate_review_gaps(gaps, ["LangGraph", "CrewAI"])

    assert consolidated.count("LangGraph: 缺少 RAG 官方文档。") == 1
    assert len([gap for gap in consolidated if "LangGraph" in gap]) == 3
    assert "CrewAI: 缺少持久化文档。" in consolidated


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


def test_failed_tool_output_is_not_evidence_and_source_metadata_is_precise():
    failed = build_evidences_from_tool_result(
        tool_name="github_get_repo",
        args={"github_url": "https://github.com/example/missing"},
        result="仓库不存在：example/missing",
        project_name="Missing",
        research_topic="维护状态",
    )
    assert failed == []

    fetched = build_evidences_from_tool_result(
        tool_name="web_fetch_tool",
        args={"url": "https://blog.example.com/post"},
        result=(
            "URL：https://blog.example.com/post\n文档类型：blog\n"
            "内容日期：2026-06-10T00:00:00Z\n正文：真实内容"
        ),
        project_name="Example",
        research_topic="文档质量",
    )
    assert len(fetched) == 1
    assert fetched[0].source_type == "blog"
    assert fetched[0].source_date == "2026-06-10T00:00:00Z"


def test_reviewer_score_rubric_and_token_usage(monkeypatch):
    reviewer_module = importlib.import_module("project_advisor.agents.reviewer")
    captured = {}

    class FakeStructuredModel:
        def with_structured_output(self, *args, **kwargs):
            assert kwargs["include_raw"] is True
            return self

        def with_retry(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            captured["prompt"] = messages[-1].content
            return {
                "parsed": ReviewResult(
                    analysis="证据有限。",
                    scores=[ProjectScore(project_name="LangGraph", learning_cost=8, deployment_cost=7)],
                    evidence_gaps=[],
                ),
                "raw": AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
                ),
                "parsing_error": None,
            }

    monkeypatch.setattr(reviewer_module, "create_chat_model", lambda *a, **k: FakeStructuredModel())
    result, usage = asyncio.run(reviewer_module._review_single_candidate(
        "LangGraph", [], "评估易用性", reviewer_module.Configuration()
    ))

    assert result.scores[0].learning_cost == 8
    assert "10 分=最容易上手" in captured["prompt"]
    assert "10 分=部署运维最简单" in captured["prompt"]
    assert "没有找到直接证据" in captured["prompt"]
    assert "evidence_gaps 最多列 2 条" in captured["prompt"]
    assert usage == {"input_tokens": 120, "output_tokens": 30}


class FakeGraph:
    async def astream(self, *args, **kwargs):
        yield {"clarify_requirements": {"next": "plan_evaluation"}}
        yield {"plan_evaluation": {"candidates": ["LangGraph"]}}
        yield {"feasibility_check": {"knowledge_stats": {"feasibility_check": {"is_feasible": True}}}}
        yield {
            "parallel_research": {
                "notes": ["Evidence"],
                "next": "evidence_coverage",
                "tool_executions": [{
                    "tool_name": "web_fetch_tool",
                    "status": "succeeded",
                    "latency_ms": 12,
                    "retry_count": 1,
                }],
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
                "token_usage": {"input_tokens": 100, "output_tokens": 25},
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
    confirmed_plan = ResearchPlan(
        research_brief="评估 Python 多智能体框架。",
        requirements=Requirements(language="Python"),
        candidates=[CandidateRecommendation(name="LangGraph", reason="用户确认")],
        evaluation_focus=["feature_match"],
    )
    client = TestClient(app_module.app)

    health_response = client.get("/api/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] in {"ok", "degraded"}
    assert health_response.json()["model_runtime"]["status"] in {
        "ready", "unavailable"
    }

    page_response = client.get("/")
    assert page_response.status_code == 200
    assert "Project Advisor" in page_response.text
    assert "开始深度评估" in page_response.text
    assert "系统评测看板" in page_response.text
    assert "本次运行诊断" in page_response.text
    assert "约束可行性预检" in page_response.text

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
            "confirmed_candidates": True,
            "confirmed_plan": confirmed_plan.model_dump(),
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
    assert '"total_tokens": 125' in body
    assert '"tool_execution": {"total": 1, "succeeded": 1' in body
    assert '"retries": 1' in body
    assert '"retrieved_evidences"' in body
    assert '"evidence_id"' in body


def test_agent_run_timeout_returns_explicit_sse_error(monkeypatch):
    app_module = importlib.import_module("project_advisor.app")

    class SlowGraph:
        async def astream(self, *args, **kwargs):
            await asyncio.sleep(0.1)
            yield {"generate_report": {"final_report": "should not finish"}}

    plan = ResearchPlan(
        research_brief="测试运行超时。",
        requirements=Requirements(language="Python"),
        candidates=[CandidateRecommendation(name="LangGraph", reason="测试")],
        evaluation_focus=["feature_match"],
    )
    monkeypatch.setattr(app_module, "graph", SlowGraph())
    monkeypatch.setenv("AGENT_RUN_TIMEOUT_SECONDS", "0.01")

    with TestClient(app_module.app).stream(
        "POST",
        "/api/advice/stream",
        json={
            "question": "请测试一个会超过执行时间上限的评估任务。",
            "confirmed_plan": plan.model_dump(),
            "confirmed_candidates": True,
            "candidates": ["LangGraph"],
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: error" in body
    assert "端到端超时" in body
    assert "should not finish" not in body


def test_health_reports_degraded_when_model_runtime_is_unavailable(monkeypatch):
    app_module = importlib.import_module("project_advisor.app")
    utils_module = importlib.import_module("project_advisor.utils")

    def unavailable_model(*args, **kwargs):
        raise ValueError("model API key is missing")

    monkeypatch.setattr(utils_module, "create_chat_model", unavailable_model)
    response = TestClient(app_module.app).get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_runtime"]["status"] == "unavailable"


def test_stream_prepares_structured_plan_for_manual_candidates(monkeypatch):
    app_module = importlib.import_module("project_advisor.app")
    plan = ResearchPlan(
        research_brief="评估适合私有化部署的 Python Agent 框架。",
        requirements=Requirements(
            language="Python",
            deployment="self_hosted",
            required_features=["human_in_the_loop"],
        ),
        candidates=[CandidateRecommendation(name="AutoCandidate", reason="自动推荐")],
        evaluation_focus=["feature_match", "deployment_cost"],
    )
    captured = {}

    async def fake_generate_candidate_plan(question):
        assert "LangGraph" in question
        return plan, {"input_tokens": 20, "output_tokens": 10}

    class CapturingGraph(FakeGraph):
        async def astream(self, graph_input, *args, **kwargs):
            captured["graph_input"] = graph_input
            async for update in super().astream(graph_input, *args, **kwargs):
                yield update

    monkeypatch.setattr(app_module, "_generate_candidate_plan", fake_generate_candidate_plan)
    monkeypatch.setattr(app_module, "graph", CapturingGraph())

    with TestClient(app_module.app).stream(
        "POST",
        "/api/advice/stream",
        json={
            "question": "请比较适合 Python 私有化部署的框架。",
            "candidates": ["LangGraph", "CrewAI"],
            "confirmed_candidates": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    confirmed_plan = captured["graph_input"]["confirmed_plan"]
    assert confirmed_plan["requirements"]["language"] == "Python"
    assert confirmed_plan["requirements"]["deployment"] == "self_hosted"
    assert captured["graph_input"]["confirmed_candidates"] == ["LangGraph", "CrewAI"]
    assert '"total_tokens": 155' in body


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
    assert "planning_diagnostics" in payload
    assert len(payload["planning_diagnostics"]["request_id"]) == 32


def test_api_key_rate_limit_and_capacity_protection(monkeypatch):
    app_module = importlib.import_module("project_advisor.app")
    app_module._rate_windows.clear()
    app_module._active_expensive_requests = 0
    monkeypatch.setenv("ADVISOR_API_KEY", "test-secret")
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "1")

    plan = ResearchPlan(
        research_brief="测试计划",
        requirements=Requirements(language="Python"),
        candidates=[CandidateRecommendation(name="LangGraph", reason="测试")],
        evaluation_focus=["feature_match"],
    )

    async def fake_plan(question):
        return plan

    monkeypatch.setattr(app_module, "_generate_candidate_plan", fake_plan)
    client = TestClient(app_module.app)
    payload = {"question": "请推荐一个适合 Python 的 Agent 框架。"}

    assert client.post("/api/candidates/suggest", json=payload).status_code == 401
    headers = {"X-API-Key": "test-secret"}
    assert client.post("/api/candidates/suggest", json=payload, headers=headers).status_code == 200
    assert client.post("/api/candidates/suggest", json=payload, headers=headers).status_code == 429

    app_module._rate_windows.clear()
    app_module._active_expensive_requests = 1
    monkeypatch.setenv("MAX_CONCURRENT_EVALUATIONS", "1")
    assert client.post("/api/candidates/suggest", json=payload, headers=headers).status_code == 429
    app_module._active_expensive_requests = 0


def test_observed_token_and_cost_budgets():
    app_module = importlib.import_module("project_advisor.app")
    config = app_module.Configuration(
        max_run_tokens=100,
        max_run_cost_usd=0.001,
        input_price_per_million=10,
        output_price_per_million=20,
    )

    assert "Token 上限" in app_module._budget_violation(config, 90, 20)
    cost_only = config.model_copy(update={"max_run_tokens": 0})
    assert "成本上限" in app_module._budget_violation(cost_only, 100, 10)
    assert app_module._budget_violation(cost_only, 10, 5) is None
