"""主 LangGraph 工作流 — AI 技术选型与开源项目评估系统的核心编排。

工作流架构：
    clarify_requirements → plan_evaluation → research_supervisor → review_and_score → generate_report

其中 research_supervisor 是一个子图（循环），管理并行研究任务。
每个研究任务由一个专门的子研究员（Repository Analyst 或 Documentation Researcher）执行。
"""

import asyncio
import logging
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from project_advisor.agents.documentation_researcher import (
    compress_research as doc_compress,
    doc_tools,
    documentation_researcher,
)
from project_advisor.agents.planner import clarify_requirements, plan_evaluation
from project_advisor.agents.repository_analyst import (
    analyst_tools,
    compress_research as repo_compress,
    repository_analyst,
)
from project_advisor.agents.reviewer import generate_report, review_and_score
from project_advisor.configuration import Configuration
from project_advisor.rag.knowledge_store import persist_evidences
from project_advisor.state import (
    AgentInputState,
    AgentState,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    SupervisorState,
)
from project_advisor.utils import (
    create_chat_model,
    get_all_tools,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
    think_tool,
)

logger = logging.getLogger(__name__)

# ===== 研究主管 =====


async def supervisor(state: SupervisorState, config: RunnableConfig) -> dict:
    """研究主管 — 规划研究策略并将任务委派给子研究员。

    主管分析研究简报，使用 think_tool 规划策略，
    通过 ConductResearch 委派任务，通过 ResearchComplete 表示研究完成。
    """
    configurable = Configuration.from_runnable_config(config)

    supervisor_tool_list = [ConductResearch, ResearchComplete, think_tool]
    research_model = (
        create_chat_model(
            configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
        )
        .bind_tools(supervisor_tool_list)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)

    return {
        "supervisor_messages": [response],
        "research_iterations": state.get("research_iterations", 0) + 1,
        "next": "supervisor_tools",
    }


async def supervisor_tools(
    state: SupervisorState, config: RunnableConfig
) -> dict:
    """执行主管的工具调用 — 处理研究委派、反思和完成信号。"""
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent = supervisor_messages[-1]

    # 退出条件
    exceeded_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent.tool_calls
    research_complete = any(
        tc["name"] == "ResearchComplete" for tc in most_recent.tool_calls
    )

    if exceeded_iterations or no_tool_calls or research_complete:
        return {
            "notes": get_notes_from_tool_calls(supervisor_messages),
            "next": "__end__",
        }

    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # 处理 think_tool 调用
    think_calls = [
        tc for tc in most_recent.tool_calls if tc["name"] == "think_tool"
    ]
    for tc in think_calls:
        all_tool_messages.append(ToolMessage(
            content=f"反思已记录：{tc['args'].get('reflection', '')}",
            name="think_tool",
            tool_call_id=tc["id"],
        ))

    # 处理 ConductResearch 调用
    research_calls = [
        tc for tc in most_recent.tool_calls if tc["name"] == "ConductResearch"
    ]

    if research_calls:
        try:
            # 限制并发数
            allowed = research_calls[:configurable.max_concurrent_research_units]
            overflow = research_calls[configurable.max_concurrent_research_units:]

            # 并行执行研究任务
            research_tasks = [
                _run_researcher_task(tc, config) for tc in allowed
            ]
            tool_results = await asyncio.gather(*research_tasks)

            for result, tc in zip(tool_results, allowed):
                all_tool_messages.append(ToolMessage(
                    content=result.get("compressed_research", "研究合成出错"),
                    name="ConductResearch",
                    tool_call_id=tc["id"],
                ))

            # 溢出任务返回错误
            for tc in overflow:
                all_tool_messages.append(ToolMessage(
                    content=f"错误：并发研究单元已达上限（{configurable.max_concurrent_research_units}）。请减少研究任务数量后重试。",
                    name="ConductResearch",
                    tool_call_id=tc["id"],
                ))

            # 汇总原始笔记
            raw_notes_concat = "\n".join([
                "\n".join(r.get("raw_notes", [])) for r in tool_results
            ])
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

            evidences = [
                evidence
                for result in tool_results
                for evidence in result.get("evidences", [])
            ]
            if evidences:
                update_payload["evidences"] = evidences
                try:
                    update_payload["knowledge_stats"] = await asyncio.to_thread(
                        persist_evidences, evidences
                    )
                except Exception:
                    logger.exception("Failed to persist reusable research evidence")

        except Exception as e:
            if _is_token_error(e, configurable.research_model):
                return {
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "next": "__end__",
                }

    update_payload["supervisor_messages"] = all_tool_messages
    update_payload["next"] = "supervisor"
    return update_payload


async def _run_researcher_task(
    tool_call: dict, config: RunnableConfig
) -> dict:
    """执行单个研究任务 — 调用 Researcher 子图。"""
    topic = tool_call["args"].get("research_topic", "")
    project_name = tool_call["args"].get("project_name", "")

    # 根据研究主题选择研究员类型
    # 如果主题涉及 GitHub、仓库、stars、release 等关键词，使用 Repository Analyst
    github_keywords = ["github", "仓库", "repo", "star", "issue", "release", "维护", "contributor"]
    is_github_task = any(kw in topic.lower() for kw in github_keywords)

    if is_github_task:
        return await researcher_subgraph_repo.ainvoke({
            "researcher_messages": [HumanMessage(content=topic)],
            "research_topic": topic,
            "project_name": project_name,
        }, config)
    else:
        return await researcher_subgraph_doc.ainvoke({
            "researcher_messages": [HumanMessage(content=topic)],
            "research_topic": topic,
            "project_name": project_name,
        }, config)


def _is_token_error(exception: Exception, model_name: str) -> bool:
    """检查是否为 token 超限错误。"""
    return is_token_limit_exceeded(exception, model_name)


# ===== 简单路由函数 =====

def _route_after_clarify(state: AgentState) -> str:
    """需求澄清后的路由。"""
    next_step = state.get("next", "research_supervisor")
    if next_step == "__end__":
        return END
    return next_step


def _route_after_plan(state: AgentState) -> str:
    """研究规划后的路由。"""
    return state.get("next", "research_supervisor")


def _route_supervisor(state: SupervisorState) -> str:
    """主管节点后的路由。"""
    return state.get("next", "supervisor")


def _route_analyst(state: ResearcherState) -> str:
    """仓库分析员的路由。"""
    return state.get("next", "analyst_tools")


def _route_doc(state: ResearcherState) -> str:
    """文档研究员的路由。"""
    return state.get("next", "doc_tools")


def _route_after_review(state: AgentState) -> str:
    """审查评分后的路由。"""
    return state.get("next", "generate_report")


# ===== 研究子图构建 =====

supervisor_builder = StateGraph(SupervisorState, context_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_conditional_edges(
    "supervisor",
    _route_supervisor,
    {"supervisor_tools": "supervisor_tools"},
)
supervisor_builder.add_conditional_edges(
    "supervisor_tools",
    _route_supervisor,
    {"supervisor": "supervisor", END: END},
)
supervisor_subgraph = supervisor_builder.compile()

repo_builder = StateGraph(
    ResearcherState, output_schema=ResearcherOutputState, context_schema=Configuration
)
repo_builder.add_node("repository_analyst", repository_analyst)
repo_builder.add_node("analyst_tools", analyst_tools)
repo_builder.add_node("compress_research", repo_compress)
repo_builder.add_edge(START, "repository_analyst")
repo_builder.add_conditional_edges(
    "repository_analyst",
    _route_analyst,
    {"analyst_tools": "analyst_tools"},
)
repo_builder.add_conditional_edges(
    "analyst_tools",
    _route_analyst,
    {
        "repository_analyst": "repository_analyst",
        "compress_research": "compress_research",
    },
)
repo_builder.add_edge("compress_research", END)
researcher_subgraph_repo = repo_builder.compile()

doc_builder = StateGraph(
    ResearcherState, output_schema=ResearcherOutputState, context_schema=Configuration
)
doc_builder.add_node("documentation_researcher", documentation_researcher)
doc_builder.add_node("doc_tools", doc_tools)
doc_builder.add_node("compress_research", doc_compress)
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
        "compress_research": "compress_research",
    },
)
doc_builder.add_edge("compress_research", END)
researcher_subgraph_doc = doc_builder.compile()


# ===== 主 Graph 构建 =====

deep_researcher_builder = StateGraph(
    AgentState,
    input_schema=AgentInputState,
    context_schema=Configuration,
)

# 添加节点
deep_researcher_builder.add_node("clarify_requirements", clarify_requirements)
deep_researcher_builder.add_node("plan_evaluation", plan_evaluation)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("review_and_score", review_and_score)
deep_researcher_builder.add_node("generate_report", generate_report)

# 添加边
deep_researcher_builder.add_edge(START, "clarify_requirements")

# 条件路由：clarify 之后可能结束（追问用户）或继续
deep_researcher_builder.add_conditional_edges(
    "clarify_requirements",
    _route_after_clarify,
    {
        "plan_evaluation": "plan_evaluation",
        END: END,
    },
)

deep_researcher_builder.add_conditional_edges(
    "plan_evaluation",
    _route_after_plan,
    {
        "research_supervisor": "research_supervisor",
    },
)

deep_researcher_builder.add_edge("research_supervisor", "review_and_score")
deep_researcher_builder.add_edge("review_and_score", "generate_report")
deep_researcher_builder.add_edge("generate_report", END)

# 编译
graph = deep_researcher_builder.compile()
