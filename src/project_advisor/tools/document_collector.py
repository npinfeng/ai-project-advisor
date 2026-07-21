"""文档采集工具 — 抓取、解析和存储网页内容。

支持：
- 从 URL 抓取网页并提取正文
- HTML → Markdown 转换
- 自动提取元数据（标题、日期）
- 内容去重和截断
- 批量采集和进度报告
"""

import asyncio
import hashlib
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from project_advisor.schemas.evidence import Evidence
from project_advisor.utils import get_iso_timestamp


# ===== 网页抓取 =====

async def fetch_webpage(url: str, timeout: int = 30) -> dict:
    """抓取单个网页并解析为文本。

    Args:
        url: 网页 URL
        timeout: 超时秒数

    Returns:
        包含 url、title、content、content_type、status 的字典
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=timeout
        ) as client:
            response = await client.get(url, headers=headers)

            if response.status_code != 200:
                return {
                    "url": url,
                    "title": "",
                    "content": f"HTTP {response.status_code}",
                    "content_type": "",
                    "status": response.status_code,
                }

            content_type = response.headers.get("content-type", "").lower()
            html = response.text

            # 使用 BeautifulSoup 提取正文
            soup = BeautifulSoup(html, "html.parser")

            # 移除脚本和样式
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            title = soup.title.string if soup.title else ""
            title = title.strip() if title else ""

            # 提取正文
            body = soup.find("body")
            if body:
                text = body.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            # 清理文本
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            # 去重连续空行
            cleaned = []
            for line in lines:
                if len(line) > 5:  # 过滤太短的行
                    cleaned.append(line)

            content = "\n".join(cleaned)

            return {
                "url": url,
                "title": title,
                "content": content,
                "content_type": content_type,
                "status": response.status_code,
            }

    except httpx.TimeoutException:
        return {
            "url": url,
            "title": "",
            "content": f"超时（>{timeout}s）",
            "content_type": "",
            "status": 0,
        }
    except Exception as e:
        return {
            "url": url,
            "title": "",
            "content": f"抓取失败：{str(e)}",
            "content_type": "",
            "status": 0,
        }


# ===== 内容处理 =====

def extract_domain(url: str) -> str:
    """从 URL 提取域名。"""
    parsed = urlparse(url)
    return parsed.netloc or parsed.hostname or ""


def detect_document_type(url: str, title: str, content: str) -> str:
    """根据 URL、标题和内容自动检测文档类型。

    Returns:
        'official_documentation', 'github', 'blog', 'community', 'release_note', 'unknown'
    """
    url_lower = url.lower()
    title_lower = title.lower()

    # GitHub 相关
    if "github.com" in url_lower:
        if "/releases/tag/" in url_lower or "release" in title_lower:
            return "release_note"
        return "github"

    # 官方文档
    doc_indicators = ["docs.", "documentation", "api-reference", "guide", "tutorial"]
    if any(ind in url_lower for ind in doc_indicators):
        return "official_documentation"

    # 博客
    blog_indicators = ["blog.", "medium.com", "dev.to", "substack"]
    if any(ind in url_lower for ind in blog_indicators):
        return "blog"

    # 社区
    community_indicators = ["stackoverflow", "reddit", "discord", "forum", "github.com/.*/discussions"]
    if any(ind in url_lower for ind in community_indicators):
        return "community"

    return "unknown"


def create_content_hash(content: str) -> str:
    """计算内容 SHA256 哈希，用于去重。"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def truncate_content(content: str, max_chars: int = 10000) -> str:
    """截断内容到指定字符数，保留完整的段落。"""
    if len(content) <= max_chars:
        return content
    truncated = content[:max_chars]
    # 尝试在最后一个换行处截断
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars * 0.8:
        return truncated[:last_nl] + "\n\n[... 内容过长，已截断 ...]"
    return truncated + "\n\n[... 内容过长，已截断 ...]"


# ===== 批量采集 =====

async def collect_documents(
    urls: list[str],
    project_name: str,
    max_concurrent: int = 3,
    max_chars_per_doc: int = 10000,
) -> list[Evidence]:
    """批量抓取 URL 并转换为 Evidence 对象。

    Args:
        urls: 要抓取的 URL 列表
        project_name: 关联的项目名称
        max_concurrent: 最大并发数
        max_chars_per_doc: 每个文档最大字符数

    Returns:
        Evidence 对象列表
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(url: str) -> Optional[Evidence]:
        async with semaphore:
            result = await fetch_webpage(url)
            if result["status"] != 200:
                return None

            content = truncate_content(result["content"], max_chars_per_doc)
            doc_type = detect_document_type(
                url, result["title"], result["content"]
            )

            return Evidence(
                source_url=url,
                source_type=doc_type,
                project_name=project_name,
                content=content,
                relevance="documentation",
                confidence="medium",
                retrieved_at=get_iso_timestamp(),
                version_info=None,
            )

    tasks = [fetch_one(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# ===== LangChain 工具 =====

@tool(description="Fetch and extract the main content from a web page. Returns cleaned text with title and metadata.")
async def web_fetch_tool(url: str) -> str:
    """抓取网页并提取正文。

    用于获取搜索结果的完整内容。
    自动清理 HTML、脚本和导航元素。

    Args:
        url: 要抓取的网页 URL

    Returns:
        格式化的网页内容字符串
    """
    result = await fetch_webpage(url)

    if result["status"] != 200:
        return f"无法抓取网页：{result['content']}"

    content = truncate_content(result["content"], 8000)

    return f"""--- 网页内容 ---
URL：{url}
标题：{result['title']}
域名：{extract_domain(url)}
文档类型：{detect_document_type(url, result['title'], result['content'])}
抓取时间：{get_iso_timestamp()}

正文：
{content}
"""


@tool(description="Batch fetch content from multiple URLs. Use this to gather documentation from several sources at once.")
async def batch_fetch_tool(
    urls: list[str],
    project_name: str = "",
) -> str:
    """批量抓取多个 URL 的内容。

    Args:
        urls: URL 列表
        project_name: 关联项目名称

    Returns:
        汇总的网页内容
    """
    if not urls:
        return "未提供 URL。"

    evidence_list = await collect_documents(
        urls,
        project_name=project_name or "unknown",
        max_concurrent=3,
    )

    if not evidence_list:
        return f"未能成功抓取 {len(urls)} 个 URL。"

    lines = [f"批量抓取完成。成功 {len(evidence_list)}/{len(urls)} 个：\n"]
    for i, ev in enumerate(evidence_list):
        lines.append(
            f"## 文档 {i + 1}\n"
            f"- URL：{ev.source_url}\n"
            f"- 类型：{ev.source_type}\n"
            f"- 抓取时间：{ev.retrieved_at}\n"
            f"- 内容摘要：{ev.content[:500]}...\n"
        )

    return "\n".join(lines)
