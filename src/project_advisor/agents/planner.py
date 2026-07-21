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


async def plan_evaluation(state: AgentState, config: RunnableConfig):
    """生成结构化研究计划和候选项目列表。

    将用户消息转化为：
    1. ResearchBrief — 详细的研究简报
    2. Candidates — 候选项目列表
    3. EvaluationFocus — 重点评估维度
    """
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
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str(),
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt)])

    # 初始化 Supervisor 上下文
    supervisor_system_prompt = planner_system_prompt.format(
        date=get_today_str(),
    )

    return {
        "research_brief": response.research_brief,
        "candidates": response.candidates,
        "supervisor_messages": {
            "type": "override",
            "value": [
                SystemMessage(content=supervisor_system_prompt),
                HumanMessage(content=response.research_brief),
            ],
        },
        "next": "research_supervisor",
    }
