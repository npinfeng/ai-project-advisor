"""Planner Agent — 需求解析、候选项目确定、研究计划生成。

Planner 是技术选型流程的第一个核心 Agent，负责：
1. 将用户自然语言需求转化为结构化 Requirements
2. 确定要评估的候选项目列表
3. 设计评估维度和优先级
4. 生成可供 Supervisor 分发的研究计划
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig

from project_advisor.configuration import Configuration
from project_advisor.prompts import (
    clarify_with_user_instructions,
    planner_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from project_advisor.schemas.evidence import CandidateRecommendation
from project_advisor.state import AgentState, ClarifyWithUser, ResearchPlan
from project_advisor.utils import create_chat_model, get_today_str


async def clarify_requirements(state: AgentState, config: RunnableConfig):
    """分析用户需求并决定是否需要追问。

    如果澄清功能被禁用或需求足够明确，直接进入研究规划阶段。
    否则向用户提出追问以获取更完整的需求信息。
    """
    configurable = Configuration.from_runnable_config(config)

    # 如果禁用了澄清功能，直接跳过
    if not configurable.allow_clarification:
        return {"next": "plan_evaluation"}

    messages = state.get("messages", [])

    clarification_model = (
        create_chat_model(
            configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
        )
        .with_structured_output(ClarifyWithUser, method="function_calling")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str(),
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt)])

    if response.need_clarification:
        return {
            "messages": [AIMessage(content=response.question)],
            "next": "__end__",
        }

    return {
        "messages": [AIMessage(content=response.verification)],
        "next": "plan_evaluation",
    }


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
        response = await generate_research_plan(
            state.get("messages", []),
            config,
        )

    response = _apply_confirmed_candidates(
        response,
        state.get("confirmed_candidates", []),
    )
    candidate_names = [candidate.name for candidate in response.candidates]

    # 初始化 Supervisor 上下文
    supervisor_system_prompt = planner_system_prompt.format(
        date=get_today_str(),
    )

    return {
        "requirements": response.requirements,
        "research_brief": response.research_brief,
        "candidates": candidate_names,
        "candidate_recommendations": response.candidates,
        "evaluation_focus": response.evaluation_focus,
        "supervisor_messages": {
            "type": "override",
            "value": [
                SystemMessage(content=supervisor_system_prompt),
                HumanMessage(
                    content=(
                        f"{response.research_brief}\n\n"
                        f"候选项目：{'、'.join(candidate_names)}\n"
                        f"重点维度：{'、'.join(response.evaluation_focus)}"
                    )
                ),
            ],
        },
        "next": "research_supervisor",
    }
