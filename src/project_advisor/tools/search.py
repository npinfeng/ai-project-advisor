"""搜索工具 — Tavily 和 DuckDuckGo 搜索封装。"""

from typing import Annotated, List, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from project_advisor.utils import tavily_search_async


@tool(description="A search engine optimized for comprehensive, accurate results. Useful for finding technical documentation and current information.")
async def tavily_search_tool(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news"], InjectedToolArg] = "general",
) -> str:
    """使用 Tavily API 搜索网络并返回格式化结果。

    Args:
        queries: 搜索查询列表
        max_results: 每个查询返回的最大结果数
        topic: 搜索主题类型（general 或 news）

    Returns:
        格式化的搜索结果字符串
    """
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # 按 URL 去重
    unique_results = {}
    for response in search_results:
        for result in response.get("results", []):
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = {**result, "query": response.get("query", "")}

    if not unique_results:
        return "未找到有效搜索结果。请尝试其他搜索词。"

    formatted = "搜索结果：\n\n"
    for i, (url, result) in enumerate(unique_results.items()):
        formatted += f"\n\n--- 来源 {i + 1}：{result.get('title', '无标题')} ---\n"
        formatted += f"URL：{url}\n\n"
        raw = result.get("raw_content") or result.get("content", "")
        formatted += f"内容摘要：\n{raw[:3000]}\n"
        formatted += "\n" + "-" * 80 + "\n"

    return formatted


@tool(description="DuckDuckGo web search. Use as a fallback when other search tools are unavailable.")
async def duckduckgo_search_tool(
    search_queries: List[str],
    config: RunnableConfig = None,
) -> str:
    """使用 DuckDuckGo 进行搜索（备用搜索工具）。

    Args:
        search_queries: 搜索查询列表

    Returns:
        格式化的搜索结果
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for query in search_queries[:3]:  # 限制查询数量
                ddg_results = list(ddgs.text(query, max_results=3))
                for r in ddg_results:
                    results.append(
                        f"标题：{r.get('title', '')}\n"
                        f"URL：{r.get('href', '')}\n"
                        f"摘要：{r.get('body', '')}\n"
                    )

        if not results:
            return "DuckDuckGo 未找到结果。"

        return "\n---\n".join(results)
    except ImportError:
        return "DuckDuckGo 搜索不可用。请安装 duckduckgo-search 包。"
    except Exception as e:
        return f"DuckDuckGo 搜索出错：{str(e)}"
