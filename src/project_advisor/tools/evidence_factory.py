"""Convert heterogeneous tool outputs into traceable Evidence records."""

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from project_advisor.schemas.evidence import Evidence


URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}\"']+")
ERROR_PREFIXES = (
    "工具执行出错",
    "无法抓取网页",
    "抓取失败",
    "仓库不存在",
    "api 频率限制",
    "github api 错误",
    "获取 release 失败",
    "获取 issue 失败",
    "获取 readme 失败",
    "duckduckgo 搜索出错",
    "本地知识库中还没有",
    "未在本地知识库中找到",
    "未找到有效搜索结果",
)
SOURCE_DATE_PATTERN = re.compile(
    r"(?:内容日期|发布日期|最近更新|最近推送|source_date|published_at|updated_at)"
    r"[：:\s\"']+([0-9]{4}-[0-9]{2}-[0-9]{2}(?:[T ][0-9:.+-Z]+)?)",
    re.IGNORECASE,
)


def _serialize_result(result: Any) -> str:
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


def is_error_tool_result(result: Any) -> bool:
    """Return True when a tool observation represents failure, not evidence."""
    if isinstance(result, dict):
        status = result.get("status")
        if result.get("success") is False or result.get("error"):
            return True
        if isinstance(status, int) and status >= 400:
            return True
    content = _serialize_result(result).strip().casefold()
    return any(content.startswith(prefix) for prefix in ERROR_PREFIXES)


def _document_type_from_content(content: str) -> str | None:
    match = re.search(
        r"文档类型[：:]\s*(official_documentation|github|blog|community|release_note|unknown)",
        content,
        re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


def _source_metadata(
    tool_name: str,
    source_url: str,
    content: str = "",
) -> tuple[str, str]:
    name = tool_name.lower()
    parsed = urlparse(source_url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    if host == "github.com" or host.endswith(".github.com") or "github" in name:
        if "/releases" in path or "release" in name:
            return "release_note", "high"
        return "github", "high"
    if "mcp" in name:
        return "mcp", "high"
    if "rag" in name or source_url.startswith("tool://"):
        return "local_rag", "medium"
    detected_type = _document_type_from_content(content)
    if detected_type and detected_type != "unknown":
        confidence = (
            "high"
            if detected_type in {"official_documentation", "release_note"}
            else "medium"
        )
        return detected_type, confidence
    if any(token in name for token in ("search", "tavily", "duckduckgo")):
        return "web_search", "medium"
    if any(token in name for token in ("fetch", "documentation", "docs")):
        return "web_search", "medium"
    return "tool_output", "low"


def _extract_source_date(result: Any, content: str) -> str | None:
    if isinstance(result, dict):
        for key in ("source_date", "published_at", "date_published", "updated_at"):
            value = result.get(key)
            if value:
                return str(value)
    match = SOURCE_DATE_PATTERN.search(content)
    return match.group(1) if match else None


def build_evidences_from_tool_result(
    *,
    tool_name: str,
    args: dict[str, Any] | None,
    result: Any,
    project_name: str,
    research_topic: str,
) -> list[Evidence]:
    """Normalize a tool observation into one Evidence object per source URL."""
    args = args or {}
    content = _serialize_result(result).strip()
    if not content or is_error_tool_result(result):
        return []

    urls: list[str] = []
    for candidate in (
        args.get("url"),
        args.get("github_url"),
        args.get("repository_url"),
    ):
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            urls.append(candidate)
    tool_name_lower = tool_name.casefold()
    if any(token in tool_name_lower for token in ("search", "batch", "rag", "mcp")):
        urls.extend(URL_PATTERN.findall(content))

    unique_urls: list[str] = []
    for url in urls:
        cleaned = url.rstrip(".,;:!?，。；：！？)")
        if cleaned and cleaned not in unique_urls:
            unique_urls.append(cleaned)
    if not unique_urls:
        unique_urls = [f"tool://{tool_name}"]

    resolved_project = (
        project_name.strip()
        or str(args.get("project_name", "")).strip()
        or str(args.get("repo", "")).strip()
        or "未明确项目"
    )
    relevance = research_topic.strip() or "候选项目技术评估"
    retrieved_at = datetime.now(timezone.utc).isoformat()
    source_date = _extract_source_date(result, content)

    evidences = []
    for source_url in unique_urls[:12]:
        source_type, confidence = _source_metadata(tool_name, source_url, content)
        evidences.append(
            Evidence(
                source_url=source_url,
                source_type=source_type,
                project_name=resolved_project,
                content=content[:6000],
                relevance=relevance[:1000],
                confidence=confidence,
                retrieved_at=retrieved_at,
                source_date=source_date,
            )
        )
    return evidences
