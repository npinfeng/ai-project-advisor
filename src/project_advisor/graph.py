"""Thin multi-agent, thick-workflow orchestration for Project Advisor.

Only two specialist Research Agent roles retain tool-using loops. Planning,
dispatch, evidence coverage, persistence, score binding and report routing are
deterministic workflow nodes. A single bounded supplemental round is allowed.
"""

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from project_advisor.agents.documentation_researcher import (
    compress_research as doc_summarize,
    doc_tools,
    documentation_researcher,
)
from project_advisor.agents.planner import (
    build_research_tasks,
    clarify_requirements,
    plan_evaluation,
)
from project_advisor.agents.repository_analyst import (
    analyst_tools,
    compress_research as repo_summarize,
    repository_analyst,
)
from project_advisor.agents.reviewer import generate_report, review_and_score
from project_advisor.configuration import Configuration
from project_advisor.observability.logging import bind_log_context, log_event
from project_advisor.rag.knowledge_store import persist_evidences
from project_advisor.schemas.evidence import CandidateRecommendation, Evidence
from project_advisor.state import (
    AgentInputState,
    AgentState,
    EvidenceGap,
    ResearcherOutputState,
    ResearcherState,
    ResearchTask,
)
from project_advisor.tools.constraint_analyzer import (
    analyze_feasibility,
    render_feasibility_report,
)
from project_advisor.tools.shared_cache import clear_shared_cache

logger = logging.getLogger(__name__)

REPOSITORY_SOURCE_TYPES = {"github", "release_note"}
DOCUMENTATION_SOURCE_TYPES = {
    "official_documentation",
    "blog",
    "community",
    "web_search",
    "local_rag",
    "mcp",
}


def _normalize_evidences(values: list[Any]) -> list[Evidence]:
    normalized: list[Evidence] = []
    seen: set[str] = set()
    for value in values:
        try:
            evidence = (
                value
                if isinstance(value, Evidence)
                else Evidence.model_validate(value)
            )
        except (TypeError, ValueError):
            continue
        if evidence.evidence_id in seen:
            continue
        seen.add(evidence.evidence_id)
        normalized.append(evidence)
    return normalized


def _evidence_for_project(
    project_name: str,
    evidences: list[Evidence],
) -> list[Evidence]:
    target = project_name.casefold().strip()
    return [
        evidence
        for evidence in evidences
        if target == evidence.project_name.casefold().strip()
        or target in evidence.project_name.casefold()
    ]


def detect_evidence_gaps(
    candidates: list[str],
    evidences: list[Any],
    github_url_map: dict[str, str | None] | None = None,
) -> list[EvidenceGap]:
    """Build a deterministic candidate-by-track evidence coverage matrix.

    对于没有 GitHub URL 的候选项目，不检查仓库证据缺口——它们完全可以
    依靠文档搜索和 Web 搜索完成评估。
    """
    normalized = _normalize_evidences(evidences)
    gaps: list[EvidenceGap] = []
    for candidate in candidates:
        project_evidence = _evidence_for_project(candidate, normalized)
        source_types = {item.source_type for item in project_evidence}
        # 只有明确有 GitHub 的项目才检查仓库证据
        has_github = github_url_map.get(candidate) if github_url_map else True
        if has_github and not source_types.intersection(REPOSITORY_SOURCE_TYPES):
            gaps.append(EvidenceGap(
                project_name=candidate,
                track="repository",
                reason="缺少 GitHub 仓库、Release 或 Issue 的结构化证据。",
            ))
        if not source_types.intersection(DOCUMENTATION_SOURCE_TYPES):
            gaps.append(EvidenceGap(
                project_name=candidate,
                track="documentation",
                reason="缺少官方文档、Web 搜索或只读 RAG 的结构化证据。",
            ))
    return gaps


async def _run_research_task(
    task: ResearchTask,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Route by a typed track rather than guessing from topic keywords."""
    task_input = {
        "researcher_messages": [HumanMessage(content=task.research_topic)],
        "research_topic": task.research_topic,
        "project_name": task.project_name,
    }
    with bind_log_context(
        research_task_id=task.task_id,
        candidate=task.project_name,
        track=task.track,
        node=f"{task.track}_research",
    ):
        log_event(logger, logging.INFO, "research_task_started")
        if task.track == "repository":
            result = await researcher_subgraph_repo.ainvoke(task_input, config)
        else:
            result = await researcher_subgraph_doc.ainvoke(task_input, config)
        log_event(logger, logging.INFO, "research_task_completed")
        return result


async def _execute_research_tasks(
    tasks: list[ResearchTask],
    config: RunnableConfig,
) -> tuple[list[Evidence], list[str], dict[str, int], list[dict[str, Any]]]:
    configurable = Configuration.from_runnable_config(config)
    concurrency = max(1, configurable.max_concurrent_research_units)
    semaphore = asyncio.Semaphore(concurrency)

    async def run(task: ResearchTask) -> tuple[ResearchTask, Any]:
        async with semaphore:
            try:
                return task, await _run_research_task(task, config)
            except Exception as error:
                with bind_log_context(
                    research_task_id=task.task_id,
                    candidate=task.project_name,
                    track=task.track,
                ):
                    log_event(
                        logger,
                        logging.ERROR,
                        "research_task_failed",
                        error_type=type(error).__name__,
                    )
                    logger.exception("Research task failed")
                return task, error

    results = await asyncio.gather(*(run(task) for task in tasks))
    evidences: list[Any] = []
    notes: list[str] = []
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    tool_executions: list[dict[str, Any]] = []
    for task, result in results:
        if isinstance(result, Exception):
            notes.append(
                f"[{task.task_id}] {task.project_name}/{task.track} 执行失败："
                f"{type(result).__name__}: {result}"
            )
            continue
        evidences.extend(result.get("evidences", []))
        result_usage = result.get("token_usage", {})
        tool_executions.extend(result.get("tool_executions", []))
        token_usage["input_tokens"] += int(
            result_usage.get("input_tokens", 0) or 0
        )
        token_usage["output_tokens"] += int(
            result_usage.get("output_tokens", 0) or 0
        )
        summary = str(result.get("compressed_research", "")).strip()
        if summary:
            notes.append(f"[{task.task_id}]\n{summary}")

    normalized = _normalize_evidences(evidences)
    if normalized:
        try:
            await asyncio.to_thread(persist_evidences, normalized)
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "evidence_persistence_failed",
                evidence_count=len(normalized),
            )
            logger.exception("Failed to persist reusable research evidence")
    return normalized, notes, token_usage, tool_executions


async def parallel_research(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Execute one bounded research round and return only new observations.

    首轮研究开始时清空共享缓存，研究员通过缓存共享共性发现。
    """
    tasks = [
        task if isinstance(task, ResearchTask) else ResearchTask.model_validate(task)
        for task in state.get("research_tasks", [])
    ]
    # 首轮研究开始时清空缓存，避免跨评估污染
    if state.get("research_round", 0) == 0:
        clear_shared_cache()
    evidences, notes, token_usage, tool_executions = await _execute_research_tasks(
        tasks, config
    )
    return {
        "evidences": evidences,
        "raw_notes": notes,
        "tool_executions": tool_executions,
        "token_usage": token_usage,
        "research_round": max((task.round for task in tasks), default=0),
        "next": "evidence_coverage",
    }


def evidence_coverage(state: AgentState) -> dict[str, Any]:
    """Allow at most one supplemental round for explicit coverage gaps."""
    candidates = state.get("candidates", [])
    github_url_map = {
        rec.name: rec.github_url
        for rec in state.get("candidate_recommendations", [])
        if hasattr(rec, "name")
    }
    gaps = detect_evidence_gaps(candidates, state.get("evidences", []), github_url_map)
    if not gaps or state.get("supplemental_round_used", False):
        return {
            "evidence_gaps": gaps,
            "next": "review_and_score",
        }

    recommendations = []
    recommendation_map: dict[str, CandidateRecommendation] = {}
    for value in state.get("candidate_recommendations", []):
        try:
            item = (
                value
                if isinstance(value, CandidateRecommendation)
                else CandidateRecommendation.model_validate(value)
            )
        except (TypeError, ValueError):
            continue
        recommendation_map[item.name] = item
    for candidate in candidates:
        recommendations.append(
            recommendation_map.get(candidate)
            or CandidateRecommendation(
                name=candidate,
                reason="用于确定性补充研究。",
            )
        )
    requested_tracks: dict[str, set[str]] = {}
    reasons: dict[tuple[str, str], str] = {}
    for gap in gaps:
        requested_tracks.setdefault(gap.project_name, set()).add(gap.track)
        reasons[(gap.project_name, gap.track)] = gap.reason

    tasks = build_research_tasks(
        recommendations,
        state.get("evaluation_focus", []),
        research_brief=state.get("research_brief", ""),
        round_number=1,
        requested_tracks=requested_tracks,
    )
    tasks = [
        task.model_copy(update={
            "research_topic": (
                f"{task.research_topic}\n补充研究目标："
                f"{reasons[(task.project_name, task.track)]}"
            )
        })
        for task in tasks
    ]
    return {
        "research_tasks": tasks,
        "evidence_gaps": gaps,
        "supplemental_round_used": True,
        "next": "supplemental_research",
    }


def _route_after_clarify(state: AgentState) -> str:
    return state.get("next", "plan_evaluation")


def _route_after_plan(state: AgentState) -> str:
    return state.get("next", "confirm_plan")


def await_clarification(state: AgentState) -> dict[str, Any]:
    """Pause safely until the user provides the next clarification answer."""
    question = state.get("pending_clarification", "") or "请补充你的关键约束。"
    answer: Any = interrupt({
        "kind": "clarification",
        "question": question,
        "round": state.get("clarification_round", 1),
        "max_rounds": state.get("max_clarification_rounds", 3),
    })
    if isinstance(answer, dict):
        answer = answer.get("answer", "")
    answer_text = str(answer or "").strip()
    while not answer_text:
        answer = interrupt({
            "kind": "clarification",
            "question": "回答不能为空，请补充你的关键约束。",
            "round": state.get("clarification_round", 1),
            "max_rounds": state.get("max_clarification_rounds", 3),
        })
        if isinstance(answer, dict):
            answer = answer.get("answer", "")
        answer_text = str(answer or "").strip()
    return {
        "messages": [HumanMessage(content=answer_text)],
        "pending_clarification": "",
        "next": "clarify_requirements",
    }


def confirm_plan(state: AgentState) -> dict[str, Any]:
    """Pause for candidate confirmation without repeating Planner side effects."""
    recommendations = [
        value
        if isinstance(value, CandidateRecommendation)
        else CandidateRecommendation.model_validate(value)
        for value in state.get("candidate_recommendations", [])
    ]
    existing_names = state.get("confirmed_candidates", [])
    if existing_names:
        selected_names = existing_names
    else:
        requirements = state.get("requirements")
        response: Any = interrupt({
            "kind": "candidate_confirmation",
            "question": "请确认、删除或补充候选项目后继续。",
            "requirements": (
                requirements.model_dump()
                if hasattr(requirements, "model_dump")
                else requirements or {}
            ),
            "candidates": [item.model_dump() for item in recommendations],
        })
        if isinstance(response, dict):
            response = response.get("candidates", [])
        if isinstance(response, str):
            response = response.replace("，", ",").split(",")
        selected_names = list(dict.fromkeys(
            str(name).strip() for name in (response or []) if str(name).strip()
        ))
        if not selected_names:
            selected_names = [item.name for item in recommendations]
    if not selected_names:
        raise ValueError("候选项目不能为空。")
    if len(selected_names) > 8:
        raise ValueError("候选项目最多 8 个。")

    recommendation_map = {item.name.casefold(): item for item in recommendations}
    selected_recommendations = [
        recommendation_map.get(name.casefold())
        or CandidateRecommendation(name=name, reason="用户在确认阶段手动添加。")
        for name in selected_names
    ]
    tasks = build_research_tasks(
        selected_recommendations,
        state.get("evaluation_focus", []),
        research_brief=state.get("research_brief", ""),
    )
    return {
        "confirmed_candidates": selected_names,
        "candidates": selected_names,
        "candidate_recommendations": selected_recommendations,
        "research_tasks": tasks,
        "next": "feasibility_check",
    }


def _route_researcher(state: ResearcherState, tool_node: str) -> str:
    return state.get("next", tool_node)


def _route_repo(state: ResearcherState) -> str:
    return _route_researcher(state, "analyst_tools")


def _route_doc(state: ResearcherState) -> str:
    return _route_researcher(state, "doc_tools")


def _route_after_coverage(state: AgentState) -> str:
    return state.get("next", "review_and_score")


def feasibility_check(state: AgentState) -> dict[str, Any]:
    """在研究投入前检测需求物理可行性。

    对 Planner 生成的结构化需求进行分析，识别：
    - 物理矛盾（离线+大模型API、低内存+大参数）
    - 约束冲突（候选框架不支持必须功能）
    - 降级路径（当理想方案不可行时的替代选择）

    预检结果注入 research_brief，供研究阶段和 Reviewer 参考。
    """
    requirements = state.get("requirements")
    candidates = state.get("candidates", [])
    research_brief = state.get("research_brief", "")

    req_dict = {}
    if requirements is not None:
        if hasattr(requirements, "model_dump"):
            req_dict = requirements.model_dump()
        elif isinstance(requirements, dict):
            req_dict = requirements

    report = analyze_feasibility(req_dict, candidates)
    feasibility_text = render_feasibility_report(report)

    augmented_brief = f"{research_brief}\n\n{feasibility_text}"

    return {
        "research_brief": augmented_brief,
        "knowledge_stats": {
            **state.get("knowledge_stats", {}),
            "feasibility_check": {
                "is_feasible": report.is_feasible,
                "violation_count": len(report.violations),
                "degradation_path_count": len(report.degradation_paths),
            },
        },
    }


repo_builder = StateGraph(
    ResearcherState,
    output_schema=ResearcherOutputState,
    context_schema=Configuration,
)
repo_builder.add_node("repository_analyst", repository_analyst)
repo_builder.add_node("analyst_tools", analyst_tools)
repo_builder.add_node("summarize_evidence", repo_summarize)
repo_builder.add_edge(START, "repository_analyst")
repo_builder.add_conditional_edges(
    "repository_analyst",
    _route_repo,
    {"analyst_tools": "analyst_tools"},
)
repo_builder.add_conditional_edges(
    "analyst_tools",
    _route_repo,
    {
        "repository_analyst": "repository_analyst",
        "compress_research": "summarize_evidence",
    },
)
repo_builder.add_edge("summarize_evidence", END)
researcher_subgraph_repo = repo_builder.compile()

doc_builder = StateGraph(
    ResearcherState,
    output_schema=ResearcherOutputState,
    context_schema=Configuration,
)
doc_builder.add_node("documentation_researcher", documentation_researcher)
doc_builder.add_node("doc_tools", doc_tools)
doc_builder.add_node("summarize_evidence", doc_summarize)
doc_builder.add_edge(START, "documentation_researcher")
doc_builder.add_conditional_edges(
    "documentation_researcher",
    _route_doc,
    {"doc_tools": "doc_tools"},
)
doc_builder.add_conditional_edges(
    "doc_tools",
    _route_doc,
    {
        "documentation_researcher": "documentation_researcher",
        "compress_research": "summarize_evidence",
    },
)
doc_builder.add_edge("summarize_evidence", END)
researcher_subgraph_doc = doc_builder.compile()


deep_researcher_builder = StateGraph(
    AgentState,
    input_schema=AgentInputState,
    context_schema=Configuration,
)
deep_researcher_builder.add_node("clarify_requirements", clarify_requirements)
deep_researcher_builder.add_node("await_clarification", await_clarification)
deep_researcher_builder.add_node("plan_evaluation", plan_evaluation)
deep_researcher_builder.add_node("confirm_plan", confirm_plan)
deep_researcher_builder.add_node("feasibility_check", feasibility_check)
deep_researcher_builder.add_node("parallel_research", parallel_research)
deep_researcher_builder.add_node("evidence_coverage", evidence_coverage)
deep_researcher_builder.add_node("supplemental_research", parallel_research)
deep_researcher_builder.add_node("review_and_score", review_and_score)
deep_researcher_builder.add_node("generate_report", generate_report)

deep_researcher_builder.add_edge(START, "clarify_requirements")
deep_researcher_builder.add_conditional_edges(
    "clarify_requirements",
    _route_after_clarify,
    {
        "await_clarification": "await_clarification",
        "plan_evaluation": "plan_evaluation",
    },
)
deep_researcher_builder.add_edge("await_clarification", "clarify_requirements")
deep_researcher_builder.add_conditional_edges(
    "plan_evaluation",
    _route_after_plan,
    {"confirm_plan": "confirm_plan"},
)
deep_researcher_builder.add_edge("confirm_plan", "feasibility_check")
deep_researcher_builder.add_edge("feasibility_check", "parallel_research")
deep_researcher_builder.add_edge("parallel_research", "evidence_coverage")
deep_researcher_builder.add_conditional_edges(
    "evidence_coverage",
    _route_after_coverage,
    {
        "supplemental_research": "supplemental_research",
        "review_and_score": "review_and_score",
    },
)
deep_researcher_builder.add_edge("supplemental_research", "evidence_coverage")
deep_researcher_builder.add_edge("review_and_score", "generate_report")
deep_researcher_builder.add_edge("generate_report", END)

def compile_graph(checkpointer: Any | None = None) -> Any:
    """Compile the main workflow with an optional persistent checkpointer."""
    return deep_researcher_builder.compile(checkpointer=checkpointer)


graph = compile_graph()
