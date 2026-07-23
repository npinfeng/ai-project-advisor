"""Convert heterogeneous tool outputs into traceable Evidence records."""

import json
import re
from datetime import datetime, timezone
from typing import Any

from project_advisor.schemas.evidence import Evidence


URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}\"']+")


def _serialize_result(result: Any) -> str:
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


def _source_metadata(tool_name: str, source_url: str) -> tuple[str, str]:
    name = tool_name.lower()
    host = source_url.lower()
    if "github" in name or "github.com" in host:
        return "github", "high"
    if "mcp" in name:
        return "mcp", "high"
    if "rag" in name or source_url.startswith("tool://"):
        return "local_rag", "medium"
    if any(token in name for token in ("fetch", "documentation", "docs")):
        return "official_documentation", "medium"
    if any(token in name for token in ("search", "tavily", "duckduckgo")):
        return "web_search", "medium"
    return "tool_output", "low"


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
    if not content:
        return []

    urls: list[str] = []
    for candidate in (
        args.get("url"),
        args.get("github_url"),
        args.get("repository_url"),
    ):
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            urls.append(candidate)
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

    evidences = []
    for source_url in unique_urls[:12]:
        source_type, confidence = _source_metadata(tool_name, source_url)
        evidences.append(
            Evidence(
                source_url=source_url,
                source_type=source_type,
                project_name=resolved_project,
                content=content[:6000],
                relevance=relevance[:1000],
                confidence=confidence,
                retrieved_at=retrieved_at,
            )
        )
    return evidences
