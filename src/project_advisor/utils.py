"""实用工具函数 — 搜索、模型管理、日期处理等。"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Any, List, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from tavily import AsyncTavilyClient

from project_advisor.configuration import Configuration, SearchAPI
from project_advisor.prompts import clarify_with_user_instructions


# ===== 日期工具 =====

def get_today_str() -> str:
    """获取当前日期字符串，格式如 'Mon Jan 15, 2024'。"""
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


def get_iso_timestamp() -> str:
    """获取 ISO 格式的时间戳，用于证据记录。"""
    return datetime.now().isoformat()


# ===== 模型工具 =====

def create_chat_model(
    model_spec: str,
    *,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """创建 Chat Model 实例，直接支持 DeepSeek、OpenAI、Anthropic。

    格式：<provider>:<model_name>

    示例：
        create_chat_model("deepseek:deepseek-chat", max_tokens=10000)
        create_chat_model("openai:gpt-4o", max_tokens=4096)
        create_chat_model("anthropic:claude-sonnet-4", max_tokens=4096)

    DeepSeek 通过 OpenAI 兼容协议接入，自动从环境变量读取：
        DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL（默认 https://api.deepseek.com）
    """
    if ":" not in model_spec:
        raise ValueError(
            f"模型格式错误：'{model_spec}'，应为 '<provider>:<model_name>'，"
            f"如 'deepseek:deepseek-chat'"
        )

    provider, _, model_name = model_spec.partition(":")

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            max_tokens=max_tokens,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=max_tokens,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(
            f"不支持的模型提供商：'{provider}'，当前支持：deepseek, openai, anthropic"
        )


def get_tavily_api_key() -> Optional[str]:
    """获取 Tavily API Key。"""
    return os.getenv("TAVILY_API_KEY")


def get_github_token(config: RunnableConfig) -> Optional[str]:
    """获取 GitHub Token。"""
    configurable = Configuration.from_runnable_config(config)
    if configurable.github_token:
        return configurable.github_token
    return os.getenv("GITHUB_TOKEN")


# 模型 token 限制映射表
MODEL_TOKEN_LIMITS = {
    "deepseek:deepseek-chat": 131072,
    "deepseek:deepseek-reasoner": 65536,
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
}


def get_model_token_limit(model_string: str) -> Optional[int]:
    """查找模型的 token 限制。"""
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    return None


def get_config_value(value: Any) -> Any:
    """从配置中提取值，处理枚举、字典和 None。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value
    return value.value


# ===== 反思工具 =====

@tool(description="Strategic reflection tool for research planning and analysis.")
def think_tool(reflection: str) -> str:
    """策略反思工具 — 在工具调用之间暂停思考。

    使用方法：
    - 收到工具结果后：分析了哪些关键信息？
    - 制定下一步计划：还需要哪些信息？
    - 评估完整性：当前证据是否足够得出结论？

    Args:
        reflection: 详细反思内容

    Returns:
        确认反思已记录
    """
    return f"反思已记录：{reflection}"


# ===== 搜索工具 =====

async def tavily_search_async(
    search_queries: list[str],
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
    include_raw_content: bool = True,
) -> list[dict]:
    """异步执行多个 Tavily 搜索查询。"""
    api_key = get_tavily_api_key()
    if not api_key:
        raise ValueError("TAVILY_API_KEY 未设置，请检查 .env 文件")

    tavily_client = AsyncTavilyClient(api_key=api_key)
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        for query in search_queries
    ]
    return await asyncio.gather(*search_tasks)


async def get_search_tool(search_api: SearchAPI):
    """根据配置返回对应的搜索工具。"""
    if search_api == SearchAPI.TAVILY:
        from project_advisor.tools.search import tavily_search_tool
        search_tool = tavily_search_tool
        search_tool.metadata = {
            **(search_tool.metadata or {}),
            "type": "search",
            "name": "web_search",
        }
        return [search_tool]
    elif search_api == SearchAPI.DUCKDUCKGO:
        from project_advisor.tools.search import duckduckgo_search_tool
        return [duckduckgo_search_tool]
    elif search_api == SearchAPI.NONE:
        return []
    return []


async def get_all_tools(config: RunnableConfig):
    """组装完整的工具体系（搜索 + GitHub + RAG + MCP + 研究控制）。"""
    from project_advisor.mcp_client import get_mcp_tools
    from project_advisor.state import ResearchComplete
    from project_advisor.tools.github import (
        github_get_readme,
        github_get_repo,
        github_list_issues,
        github_list_releases,
    )
    from project_advisor.tools.document_collector import (
        batch_fetch_tool,
        web_fetch_tool,
    )
    from project_advisor.tools.rag_search import (
        rag_ingest,
        rag_rebuild,
        rag_search,
        rag_status,
    )

    tools = [
        think_tool,
        ResearchComplete,
        github_get_repo,
        github_list_releases,
        github_list_issues,
        github_get_readme,
        web_fetch_tool,
        batch_fetch_tool,
        rag_search,
        rag_ingest,
        rag_rebuild,
        rag_status,
    ]

    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    tools.extend(await get_mcp_tools(configurable))

    return tools


# ===== Token 限制处理 =====

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """判断异常是否表示 token/上下文超限。

    Args:
        exception: 待分析的异常
        model_name: 可选的模型名称，用于更精确的检测

    Returns:
        如果异常表示 token 超限则返回 True
    """
    error_str = str(exception).lower()

    # 检查常见 token 超限关键词
    token_keywords = [
        "token", "context", "length", "maximum context",
        "reduce", "prompt is too long", "context_length_exceeded",
        "too many tokens",
    ]
    if any(kw in error_str for kw in token_keywords):
        return True

    # 检查 OpenAI 特定错误
    if hasattr(exception, "code"):
        if getattr(exception, "code", "") == "context_length_exceeded":
            return True

    return False


def remove_up_to_last_ai_message(messages: list) -> list:
    """移除到最后一条 AI 消息为止的内容，用于 token 超限时截断。

    Args:
        messages: 消息列表

    Returns:
        截断后的消息列表
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]
    return messages


def get_notes_from_tool_calls(messages: list) -> list[str]:
    """从工具调用消息中提取笔记内容。

    Args:
        messages: 消息列表

    Returns:
        工具消息的内容列表
    """
    return [
        m.content for m in filter_messages(messages, include_types="tool")
    ]
