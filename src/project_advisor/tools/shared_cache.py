"""跨研究员共享证据缓存 — 打破研究员信息孤岛。

并发研究员可以通过此缓存共享关键发现：
- 研究员完成后将摘要写入缓存
- 后续同批次研究员可以查询已发现的共性结论
- 避免重复搜索同一话题（如多个项目都支持的 MCP 协议）
"""

from threading import Lock
from typing import Optional

_lock = Lock()
_store: dict[str, dict] = {}


def publish_shared_finding(
    topic: str,
    finding: str,
    project_name: str = "",
    source_url: str = "",
) -> None:
    """研究员发布可在同批次中共享的发现。

    Args:
        topic: 共享话题（如 "mcp_support", "deployment_options"）
        finding: 发现的摘要
        project_name: 来源项目
        source_url: 来源 URL
    """
    key = topic.strip().lower()
    with _lock:
        if key not in _store:
            _store[key] = {
                "topic": topic,
                "findings": [],
                "projects": [],
            }
        entry = _store[key]
        if finding not in entry["findings"]:
            entry["findings"].append(finding)
        if project_name and project_name not in entry["projects"]:
            entry["projects"].append(project_name)


def query_shared_findings(topic: str) -> Optional[str]:
    """查询其他研究员已发布的共享发现。

    Args:
        topic: 查询话题

    Returns:
        已有发现的摘要文本，无则返回 None
    """
    key = topic.strip().lower()
    with _lock:
        entry = _store.get(key)
    if not entry or not entry["findings"]:
        return None

    lines = [
        f"[共享发现] 其他研究员已对 '{entry['topic']}' 做了研究："
    ]
    for i, finding in enumerate(entry["findings"][:5], 1):
        lines.append(f"  {i}. {finding}")
    if entry["projects"]:
        lines.append(f"  相关项目：{', '.join(entry['projects'])}")

    return "\n".join(lines)


def list_shared_topics() -> list[str]:
    """列出所有已发布的共享话题。"""
    with _lock:
        return list(_store.keys())


def clear_shared_cache() -> None:
    """清除共享缓存（每次评估开始时调用）。"""
    with _lock:
        _store.clear()
