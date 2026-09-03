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
from project_advisor.utils import (
    create_chat_model,
    get_message_token_usage,
    get_today_str,
    invoke_structured_with_retry,
)
from project_advisor.usage_tracking import add_usage


def _count_clarification_rounds(messages: list) -> int:
    """从对话历史中推导已进行的追问轮数。

    通过统计 Assistant 发出的追问消息数量来计算，
    这样可以在多次 API 调用之间保持追问轮数的连续性。
    """
    count = 0
    for msg in messages:
        content = ""
        if hasattr(msg, "content"):
            content = str(msg.content or "")
        elif isinstance(msg, dict):
            content = str(msg.get("content", ""))
        # 检测追问特征：包含"需要了解更多"或"请补充"等追问关键词
        if any(
            keyword in content
            for keyword in [
                "需要了解更多",
                "请补充",
                "能详细说明",
                "具体是",
                "能否确认",
                "澄清",
                "clarify",
                "你是想要",
                "哪种",
                "哪些",
            ]
        ):
            count += 1
    return count


async def clarify_requirements(state: AgentState, config: RunnableConfig):
    """多轮诊断式需求澄清。

    支持最多 max_clarification_rounds 轮迭代追问。
    每轮由 Planner LLM 诊断当前信息的充分性，
    提出针对性的追问（而非 checklist 式填表），
    直到关键硬约束都明确后才进入评估计划阶段。

    追问轮数从对话历史中自动推导，跨 API 调用保持连续。
    """
    # 如果已经有确认的计划或候选项目，跳过追问
    if state.get("confirmed_plan") or state.get("confirmed_candidates"):
        return {"next": "plan_evaluation"}

    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return {"next": "plan_evaluation"}

    from langchain_core.messages import get_buffer_string

    from project_advisor.prompts import clarify_with_user_instructions
    from project_advisor.utils import create_chat_model, get_today_str

    messages = state.get("messages", [])
    # 从对话历史推导追问轮数（跨调用持久）
    clarification_round = _count_clarification_rounds(messages)
    max_rounds = state.get("max_clarification_rounds", 3)

    prompt = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str(),
        round_number=clarification_round,
        max_rounds=max_rounds,
    )

    clarify_model = (
        create_chat_model(
            configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            timeout_seconds=configurable.llm_timeout_seconds,
        )
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    response = await clarify_model.ainvoke([HumanMessage(content=prompt)])
    token_usage = get_message_token_usage(response)

    # 尝试以 JSON 解析回复
    import json
    import re
    need_clarification = False
    question = ""
    verification = ""

    try:
        content = response.content if hasattr(response, "content") else str(response)
        # 提取 JSON 块
        json_match = re.search(r'\{[^{}]*"need_clarification"[^{}]*\}', content)
        if json_match:
            parsed = json.loads(json_match.group())
            need_clarification = parsed.get("need_clarification", False)
            question = parsed.get("question", "")
            verification = parsed.get("verification", "")
    except (json.JSONDecodeError, AttributeError):
        # 如果 JSON 解析失败，检查回复中是否包含问号（暗示在追问）
        content = response.content if hasattr(response, "content") else str(response)
        if "?" in content and len(content) < 500:
            need_clarification = True
            question = content

    if need_clarification and clarification_round < max_rounds:
        return {
            "messages": [AIMessage(content=question)],
            "pending_clarification": question,
            "clarification_round": clarification_round + 1,
            "token_usage": token_usage,
            "next": "await_clarification",
        }

    # 不需要追问或已达最大轮数，确认并继续
    confirm_msg = verification or "需求已确认，开始评估。"
    return {
        "messages": [AIMessage(content=confirm_msg)],
        "clarification_round": clarification_round + 1,
        "token_usage": token_usage,
        "next": "plan_evaluation",
    }


async def generate_research_plan(
    messages: list,
    config: RunnableConfig,
) -> ResearchPlan:
    """Generate the reusable structured plan used by preview and execution."""
    plan, _ = await generate_research_plan_with_usage(messages, config)
    return plan


async def generate_research_plan_with_usage(
    messages: list,
    config: RunnableConfig,
) -> tuple[ResearchPlan, dict[str, int]]:
    """Generate a structured plan while preserving provider usage metadata."""
    configurable = Configuration.from_runnable_config(config)
    research_model = (
        create_chat_model(
            configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            timeout_seconds=configurable.llm_timeout_seconds,
        )
        .with_structured_output(
            ResearchPlan,
            method="function_calling",
            include_raw=True,
        )
    )

    prompt = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(messages),
        date=get_today_str(),
    )
    plan, raw_responses = await invoke_structured_with_retry(
        research_model,
        [HumanMessage(content=prompt)],
        max_attempts=configurable.max_structured_output_retries,
    )
    usage_values = [get_message_token_usage(raw) for raw in raw_responses]
    return plan, add_usage(*usage_values)


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
        # Repository Agent 只有 GitHub 工具。缺少 URL 时由文档轨道负责发现，
        # 避免创建一个确定会失败并消耗一次 LLM 循环的伪任务。
        if "repository" in allowed_tracks and candidate.github_url:
            github_reference = candidate.github_url
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
    plan_usage = {"input_tokens": 0, "output_tokens": 0}
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
            response, plan_usage = await generate_research_plan_with_usage(
                state.get("messages", []), config
            )
        else:
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
        "token_usage": plan_usage,
        "next": "confirm_plan",
    }
