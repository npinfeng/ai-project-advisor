"""Planner Agent — 需求解析、候选项目确定、研究计划生成。

Planner 是技术选型流程的第一个核心 Agent，负责：
1. 将用户自然语言需求转化为结构化 Requirements
2. 确定要评估的候选项目列表
3. 设计评估维度和优先级
4. 生成供确定性工作流展开的研究计划
"""

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig

from project_advisor.configuration import Configuration
from project_advisor.prompts import transform_messages_into_research_topic_prompt
from project_advisor.schemas.evidence import CandidateRecommendation, Requirements
from project_advisor.state import (
    AgentState,
    ResearchPlan,
    ResearchTask,
)
from project_advisor.utils import create_chat_model, get_today_str


async def clarify_requirements(state: AgentState, config: RunnableConfig):
    """Deterministically require a confirmed plan or candidate list."""
    if not state.get("confirmed_plan") and not state.get("confirmed_candidates"):
        return {
            "messages": [AIMessage(
                content="请先生成并确认候选项目，或在请求中提供候选项目列表。"
            )],
            "next": "__end__",
        }
    return {"next": "plan_evaluation"}


async def generate_research_plan(
    messages: list,
    config: RunnableConfig,
) -> ResearchPlan:
    """Generate the reusable structured plan used by preview and execution."""
    configurable = Configuration.from_runnable_config(config)
    research_model = (
        create_chat_model(
            configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
        )
        .with_structured_output(ResearchPlan, method="function_calling")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(messages),
        date=get_today_str(),
    )
    return await research_model.ainvoke([HumanMessage(content=prompt)])


def _apply_confirmed_candidates(
    plan: ResearchPlan,
    confirmed_candidates: list[str],
) -> ResearchPlan:
    """Keep user-confirmed names while preserving Planner explanations when possible."""
    if not confirmed_candidates:
        return plan
    recommendation_map = {
        item.name.casefold(): item for item in plan.candidates
    }
    recommendations = []
    for name in confirmed_candidates:
        existing = recommendation_map.get(name.casefold())
        recommendations.append(
            existing
            if existing is not None
            else CandidateRecommendation(
                name=name,
                reason="用户在候选确认阶段手动指定。",
            )
        )
    brief = (
        f"{plan.research_brief}\n\n"
        f"用户最终确认的候选项目：{'、'.join(confirmed_candidates)}。"
    )
    return plan.model_copy(
        update={"candidates": recommendations, "research_brief": brief}
    )


def build_research_tasks(
    candidates: list[CandidateRecommendation],
    evaluation_focus: list[str],
    *,
    research_brief: str = "",
    round_number: int = 0,
    requested_tracks: dict[str, set[str]] | None = None,
) -> list[ResearchTask]:
    """Expand a confirmed plan into typed specialist work without another model call."""
    tasks: list[ResearchTask] = []
    focus = list(dict.fromkeys(item.strip() for item in evaluation_focus if item.strip()))
    repository_dimensions = ["engineering_reliability", "community_and_maintenance"]
    documentation_dimensions = [
        "feature_match",
        "documentation_quality",
        "learning_cost",
        "extensibility",
        "deployment_cost",
        *focus,
    ]
    brief_context = research_brief.strip()[:2000] or "未提供额外业务约束。"

    for index, candidate in enumerate(candidates):
        allowed_tracks = (
            requested_tracks.get(candidate.name, set())
            if requested_tracks is not None
            else {"repository", "documentation"}
        )
        if "repository" in allowed_tracks:
            github_reference = candidate.github_url or "未提供；不得猜测仓库地址"
            tasks.append(ResearchTask(
                task_id=f"r{round_number}-repository-{index}",
                project_name=candidate.name,
                github_url=candidate.github_url,
                track="repository",
                dimensions=repository_dimensions,
                research_topic=(
                    f"项目：{candidate.name}\nGitHub：{github_reference}\n"
                    f"研究背景：{brief_context}\n"
                    "仅收集仓库工程证据：仓库状态、Release、Issue、README、"
                    "许可证与维护活跃度。"
                ),
                round=round_number,
            ))
        if "documentation" in allowed_tracks:
            tasks.append(ResearchTask(
                task_id=f"r{round_number}-documentation-{index}",
                project_name=candidate.name,
                github_url=candidate.github_url,
                track="documentation",
                dimensions=documentation_dimensions,
                research_topic=(
                    f"项目：{candidate.name}\n"
                    f"研究背景：{brief_context}\n"
                    f"评估维度：{'、'.join(documentation_dimensions)}\n"
                    "仅检索官方文档和可验证的技术资料，逐项保留来源 URL。"
                ),
                round=round_number,
            ))
    return tasks


async def plan_evaluation(state: AgentState, config: RunnableConfig):
    """Generate or reuse a confirmed structured research plan."""
    confirmed_plan = state.get("confirmed_plan")
    if confirmed_plan:
        response = (
            confirmed_plan
            if isinstance(confirmed_plan, ResearchPlan)
            else ResearchPlan.model_validate(confirmed_plan)
        )
    else:
        confirmed_candidates = state.get("confirmed_candidates", [])
        if not confirmed_candidates:
            raise ValueError("执行研究前必须确认候选项目。")
        brief = get_buffer_string(state.get("messages", []))
        response = ResearchPlan(
            research_brief=brief,
            requirements=Requirements(additional_notes=brief[:2000]),
            candidates=[
                CandidateRecommendation(
                    name=name,
                    reason="用户在执行前明确确认。",
                )
                for name in confirmed_candidates
            ],
            evaluation_focus=[
                "feature_match",
                "engineering_reliability",
                "community_and_maintenance",
                "documentation_quality",
                "learning_cost",
                "extensibility",
                "deployment_cost",
            ],
        )

    response = _apply_confirmed_candidates(
        response,
        state.get("confirmed_candidates", []),
    )
    candidate_names = [candidate.name for candidate in response.candidates]

    research_tasks = build_research_tasks(
        response.candidates,
        response.evaluation_focus,
        research_brief=response.research_brief,
    )

    return {
        "requirements": response.requirements,
        "research_brief": response.research_brief,
        "candidates": candidate_names,
        "candidate_recommendations": response.candidates,
        "evaluation_focus": response.evaluation_focus,
        "research_tasks": research_tasks,
        "research_round": 0,
        "supplemental_round_used": False,
        "evidence_gaps": [],
        "next": "parallel_research",
    }
