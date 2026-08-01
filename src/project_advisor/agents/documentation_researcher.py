"""Documentation Researcher Agent — 官方文档搜索和技术能力确认。

作为子研究员运行，接收确定性工作流分配的文档研究任务，
使用搜索工具查找官方文档、博客和技术资料。
"""

import asyncio

from langchain_core.messages import (
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig

from project_advisor.configuration import Configuration
from project_advisor.prompts import doc_researcher_system_prompt
from project_advisor.state import ResearcherState
from project_advisor.tools.evidence_factory import build_evidences_from_tool_result
from project_advisor.usage_tracking import add_usage, usage_scope
from project_advisor.utils import (
    create_chat_model,
    get_documentation_tools,
    get_message_token_usage,
    get_today_str,
)


async def documentation_researcher(state: ResearcherState, config: RunnableConfig):
    """文档研究员 — 使用搜索工具查找和分析项目文档。"""
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    tools = await get_documentation_tools(config)
    if not tools:
        raise ValueError("未配置任何工具。请在配置中启用搜索 API。")

    system_prompt = doc_researcher_system_prompt.format(date=get_today_str())

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
        "token_usage": get_message_token_usage(response),
        "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        "next": "doc_tools",
    }


async def execute_tool_safely(
    tool, args, config, *, project_name: str, research_topic: str
):
    """安全执行工具。"""
    if tool is None:
        return "工具执行出错：未找到对应工具。", [], {
            "input_tokens": 0,
            "output_tokens": 0,
        }
    try:
        with usage_scope() as nested_usage:
            result = await tool.ainvoke(args, config)
        return str(result), build_evidences_from_tool_result(
            tool_name=getattr(tool, "name", getattr(tool, "__name__", "unknown")),
            args=args,
            result=result,
            project_name=project_name,
            research_topic=research_topic,
        ), dict(nested_usage)
    except Exception as e:
        return f"工具执行出错：{str(e)}", [], {
            "input_tokens": 0,
            "output_tokens": 0,
        }


async def doc_tools(state: ResearcherState, config: RunnableConfig):
    """执行文档研究员的工具调用并处理结果。"""
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent = researcher_messages[-1]

    if not most_recent.tool_calls:
        return {"next": "compress_research"}

    research_complete_calls = [
        tc for tc in most_recent.tool_calls if tc["name"] == "ResearchComplete"
    ]
    executable_calls = [
        tc for tc in most_recent.tool_calls if tc["name"] != "ResearchComplete"
    ]

    tools = await get_documentation_tools(config)
    tools_by_name = {
        getattr(t, "name", getattr(t, "__name__", "unknown")): t
        for t in tools
    }
    project_name = state.get("project_name", "")
    research_topic = state.get("research_topic", "")

    tool_tasks = [
        execute_tool_safely(
            tools_by_name.get(tc["name"]),
            tc["args"],
            config,
            project_name=project_name,
            research_topic=research_topic,
        )
        for tc in executable_calls
    ]
    observations = await asyncio.gather(*tool_tasks)

    tool_messages = [
        ToolMessage(content=obs[0], name=tc["name"], tool_call_id=tc["id"])
        for obs, tc in zip(observations, executable_calls)
    ]
    evidences = [evidence for _, batch, _ in observations for evidence in batch]
    nested_usage = add_usage(*(usage for _, _, usage in observations))
    tool_messages.extend(
        ToolMessage(
            content="研究完成信号已确认。",
            name="ResearchComplete",
            tool_call_id=tc["id"],
        )
        for tc in research_complete_calls
    )

    exceeded = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete = bool(research_complete_calls)

    if exceeded or research_complete:
        return {
            "researcher_messages": tool_messages,
            "evidences": evidences,
            "token_usage": nested_usage,
            "next": "compress_research",
        }

    return {
        "researcher_messages": tool_messages,
        "evidences": evidences,
        "token_usage": nested_usage,
        "next": "documentation_researcher",
    }


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Deterministically summarize normalized evidence without another model call."""
    researcher_messages = state.get("researcher_messages", [])
    raw_notes = "\n".join(
        str(m.content) for m in filter_messages(
            researcher_messages, include_types=["tool"]
        )
    )
    MAX_RAW_CHARS = 8000
    was_truncated = len(raw_notes) > MAX_RAW_CHARS
    truncated_notes = raw_notes[:MAX_RAW_CHARS]

    seen: set[str] = set()
    evidence_lines = []
    for evidence in state.get("evidences", []):
        evidence_id = getattr(evidence, "evidence_id", "")
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        evidence_lines.append(
            f"- [{evidence_id}] {getattr(evidence, 'source_url', '')} — "
            f"{str(getattr(evidence, 'content', ''))[:800]}"
        )
    compressed = "\n".join(evidence_lines) or truncated_notes or "未收集到文档证据。"
    if was_truncated and not evidence_lines:
        compressed += (
            f"\n[⚠ 原始工具输出 {len(raw_notes)} 字符，"
            f"压缩至 {MAX_RAW_CHARS} 字符。"
            f"部分细节可能丢失，完整证据已持久化至 DocumentStore。]"
        )

    return {
        "compressed_research": compressed,
        "raw_notes": [raw_notes],
    }
