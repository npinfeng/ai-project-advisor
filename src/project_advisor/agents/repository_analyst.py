"""Repository Analyst Agent — GitHub 仓库深度分析。

作为子研究员运行，接收 Supervisor 分配的研究任务，
使用 GitHub 工具收集仓库的工程数据并生成分析报告。
"""

import asyncio

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig

from project_advisor.configuration import Configuration
from project_advisor.prompts import repo_analyst_system_prompt
from project_advisor.state import ResearcherState
from project_advisor.utils import (
    create_chat_model,
    get_all_tools,
    get_today_str,
    think_tool,
)


async def repository_analyst(state: ResearcherState, config: RunnableConfig):
    """仓库分析研究员 — 使用 GitHub 工具收集项目数据。"""
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    tools = await get_all_tools(config)
    if not tools:
        raise ValueError("未配置任何工具。请在配置中启用搜索 API 或添加工具。")

    system_prompt = repo_analyst_system_prompt.format(date=get_today_str())

    research_model = (
        create_chat_model(
            configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
        )
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    messages = [SystemMessage(content=system_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    return {
        "researcher_messages": [response],
        "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        "next": "analyst_tools",
    }


async def execute_tool_safely(tool, args, config):
    """安全执行工具，带错误处理和状态码检查。"""
    if tool is None:
        return "工具执行出错：未找到对应工具。"
    try:
        result = await tool.ainvoke(args, config)
        return str(result)
    except Exception as e:
        return f"工具执行出错：{str(e)}"


async def analyst_tools(state: ResearcherState, config: RunnableConfig):
    """执行仓库分析员的工具调用并处理结果。"""
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent = researcher_messages[-1]

    # 如果没有工具调用则退出
    if not most_recent.tool_calls:
        return {"next": "compress_research"}

    research_complete_calls = [
        tc for tc in most_recent.tool_calls if tc["name"] == "ResearchComplete"
    ]
    executable_calls = [
        tc for tc in most_recent.tool_calls if tc["name"] != "ResearchComplete"
    ]

    tools = await get_all_tools(config)
    tools_by_name = {
        getattr(t, "name", getattr(t, "__name__", "unknown")): t
        for t in tools
    }

    # 并行执行所有工具调用
    tool_tasks = [
        execute_tool_safely(
            tools_by_name.get(tc["name"]),
            tc["args"],
            config,
        )
        for tc in executable_calls
    ]
    observations = await asyncio.gather(*tool_tasks)

    tool_messages = [
        ToolMessage(content=obs, name=tc["name"], tool_call_id=tc["id"])
        for obs, tc in zip(observations, executable_calls)
    ]
    tool_messages.extend(
        ToolMessage(
            content="研究完成信号已确认。",
            name="ResearchComplete",
            tool_call_id=tc["id"],
        )
        for tc in research_complete_calls
    )

    # 检查退出条件
    exceeded = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete = bool(research_complete_calls)

    if exceeded or research_complete:
        return {
            "researcher_messages": tool_messages,
            "next": "compress_research",
        }

    return {
        "researcher_messages": tool_messages,
        "next": "repository_analyst",
    }


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """压缩研究发现 — 清理重复信息，保留所有关键数据。"""
    configurable = Configuration.from_runnable_config(config)

    synthesizer = create_chat_model(
        configurable.compression_model,
        max_tokens=configurable.compression_model_max_tokens,
    )

    researcher_messages = state.get("researcher_messages", [])
    researcher_messages.append(
        HumanMessage(content="请清理以上研究发现。移除重复信息，但保留所有关键数据和来源 URL。")
    )

    compression_prompt = (
        "你是一名技术分析师，刚完成了一个 GitHub 仓库的研究。"
        "请整理你的发现，保留所有数据（Stars、Release、Issue 统计等）和 URL。"
        "移除重复内容，但不要丢失任何数据点。"
        f"当前日期：{get_today_str()}"
    )

    messages = [SystemMessage(content=compression_prompt)] + researcher_messages

    try:
        response = await synthesizer.ainvoke(messages)
        compressed = str(response.content)
    except Exception:
        # 压缩失败则返回原始内容
        compressed = "\n".join([
            str(m.content) for m in filter_messages(
                researcher_messages, include_types=["tool", "ai"]
            )
        ])

    raw_notes = "\n".join([
        str(m.content) for m in filter_messages(
            researcher_messages, include_types=["tool", "ai"]
        )
    ])

    return {
        "compressed_research": compressed,
        "raw_notes": [raw_notes],
    }
