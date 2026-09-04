"""GitHub API 工具 — 获取仓库元数据和可核验的仓库文件。"""

import base64
import os
from typing import Optional
from urllib.parse import quote

import httpx
from langchain_core.tools import tool

from project_advisor import __version__

GITHUB_API_BASE = "https://api.github.com"
MAX_FILE_CHARS = 12000


def _get_headers() -> dict[str, str]:
    """构建 GitHub API 请求头。"""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": f"project-advisor/{__version__}",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _extract_owner_repo(github_url: str) -> tuple[str, str]:
    """从 GitHub URL 提取 owner 和 repo 名称。

    支持格式：
    - https://github.com/owner/repo
    - https://github.com/owner/repo.git
    - owner/repo
    """
    url = github_url.rstrip("/").removesuffix(".git")
    if "github.com/" in url:
        parts = url.split("github.com/")[-1].split("/")
    else:
        parts = url.split("/")
    if len(parts) < 2:
        raise ValueError(f"无法从 URL 解析 owner/repo：{github_url}")
    return parts[0], parts[1]


def _normalize_repository_path(path: str, *, allow_empty: bool = False) -> str:
    """Validate a repository-relative path before sending it to GitHub."""
    normalized = path.strip().strip("/").replace("\\", "/")
    if not normalized and allow_empty:
        return ""
    invalid_parts = any(
        part in {".", ".."} for part in normalized.split("/")
    )
    if not normalized or invalid_parts:
        raise ValueError("path 必须是仓库内的有效相对路径。")
    if len(normalized) > 500:
        raise ValueError("path 过长。")
    return normalized


def _github_error(action: str, response: httpx.Response) -> str:
    if response.status_code == 404:
        return f"{action}失败：文件、目录、仓库或 ref 不存在（404）。"
    if response.status_code == 403:
        return f"{action}失败：GitHub API 频率或权限限制（403），请检查 GITHUB_TOKEN。"
    return f"{action}失败（{response.status_code}）：{response.text[:300]}"


@tool(description="获取 GitHub 仓库的基本信息：Stars、语言、许可证、描述、最近更新时间、是否归档等。")
async def github_get_repo(github_url: str) -> str:
    """获取 GitHub 仓库的基本信息。

    Args:
        github_url: GitHub 仓库 URL 或 owner/repo 格式

    Returns:
        格式化的仓库信息字符串
    """
    owner, repo = _extract_owner_repo(github_url)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}",
            headers=_get_headers(),
        )

        if response.status_code == 404:
            return f"仓库不存在：{owner}/{repo}"
        if response.status_code == 403:
            return f"API 频率限制。请设置 GITHUB_TOKEN 环境变量。状态码：403"
        if response.status_code != 200:
            return f"GitHub API 错误（{response.status_code}）：{response.text[:500]}"

        data = response.json()

        return f"""--- GitHub 仓库：{owner}/{repo} ---
名称：{data.get('full_name')}
描述：{data.get('description', '无')}
主页：{data.get('homepage', '无')}
主要语言：{data.get('language', '未知')}
许可证：{data.get('license', {}).get('spdx_id', '未指定') if data.get('license') else '未指定'}
Stars：{data.get('stargazers_count', 0)}
Forks：{data.get('forks_count', 0)}
Open Issues：{data.get('open_issues_count', 0)}
Watchers：{data.get('watchers_count', 0)}
创建时间：{data.get('created_at', '未知')}
最近更新：{data.get('updated_at', '未知')}
最近推送：{data.get('pushed_at', '未知')}
已归档：{data.get('archived', False)}
默认分支：{data.get('default_branch', 'unknown')}
Topics：{', '.join(data.get('topics', [])) if data.get('topics') else '无'}
"""


@tool(description="获取仓库的 Release 列表，包括版本号、发布日期和变更说明。")
async def github_list_releases(github_url: str, per_page: int = 5) -> str:
    """获取仓库的最近 Release 列表。

    Args:
        github_url: GitHub 仓库 URL 或 owner/repo 格式
        per_page: 返回的 Release 数量，默认 5

    Returns:
        格式化的 Release 列表
    """
    owner, repo = _extract_owner_repo(github_url)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/releases",
            headers=_get_headers(),
            params={"per_page": per_page},
        )

        if response.status_code != 200:
            return f"获取 Release 失败（{response.status_code}）：{response.text[:300]}"

        releases = response.json()
        if not releases:
            return f"仓库 {owner}/{repo} 没有发布过 Release。"

        lines = [f"--- {owner}/{repo} 最近 {len(releases)} 个 Release ---"]
        for rel in releases:
            name = rel.get("name") or rel.get("tag_name", "未知")
            published = rel.get("published_at", "未知")
            prerelease = " [预发布]" if rel.get("prerelease") else ""
            body = (rel.get("body") or "")[:200]
            lines.append(f"\n## {name}{prerelease}")
            lines.append(f"发布日期：{published}")
            if body:
                lines.append(f"变更摘要：{body}")

        return "\n".join(lines)


@tool(description="获取仓库的 Issue 列表，可过滤状态（open/closed）和标签。")
async def github_list_issues(
    github_url: str, state: str = "open", per_page: int = 10, labels: str = ""
) -> str:
    """获取仓库的 Issue 列表。

    Args:
        github_url: GitHub 仓库 URL 或 owner/repo 格式
        state: Issue 状态 — 'open'、'closed' 或 'all'
        per_page: 返回数量，默认 10
        labels: 按标签过滤，逗号分隔

    Returns:
        格式化的 Issue 列表
    """
    owner, repo = _extract_owner_repo(github_url)

    params = {"state": state, "per_page": per_page, "sort": "updated"}
    if labels:
        params["labels"] = labels

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues",
            headers=_get_headers(),
            params=params,
        )

        if response.status_code != 200:
            return f"获取 Issue 失败（{response.status_code}）：{response.text[:300]}"

        issues = response.json()
        if not issues:
            return f"仓库 {owner}/{repo} 没有 {state} 状态的 Issue。"

        # 仅统计真正的 Issue（排除 Pull Request）
        real_issues = [i for i in issues if "pull_request" not in i]

        lines = [f"--- {owner}/{repo} 的 {state} Issue（共 {len(real_issues)} 个）---"]

        for issue in real_issues[:per_page]:
            title = issue.get("title", "无标题")
            created = issue.get("created_at", "未知")
            updated = issue.get("updated_at", "未知")
            labels_list = [lb["name"] for lb in issue.get("labels", [])]
            label_str = f" [标签：{', '.join(labels_list)}]" if labels_list else ""
            comments = issue.get("comments", 0)
            lines.append(
                f"#{issue.get('number')}: {title}{label_str}\n"
                f"  创建：{created} | 更新：{updated} | 评论数：{comments}"
            )

        return "\n".join(lines)


@tool(description="获取仓库的 README 内容（Markdown 格式）。")
async def github_get_readme(github_url: str) -> str:
    """获取仓库 README 文件的内容。

    Args:
        github_url: GitHub 仓库 URL 或 owner/repo 格式

    Returns:
        README 的 Markdown 原始内容（截断到 8000 字符）
    """
    owner, repo = _extract_owner_repo(github_url)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme",
            headers=_get_headers(),
        )

        if response.status_code == 404:
            return f"仓库 {owner}/{repo} 没有 README 文件。"
        if response.status_code != 200:
            return f"获取 README 失败（{response.status_code}）：{response.text[:300]}"

        data = response.json()
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")

        if len(content) > 8000:
            content = content[:8000] + "\n\n... [README 过长，已截断至前 8000 字符]"

        return f"--- {owner}/{repo} README ---\n\n{content}"


@tool(description=(
    "读取 GitHub 仓库中的指定文本文件，用于核验依赖、CI、部署、许可证和架构配置；"
    "支持指定 branch/tag/commit ref。"
))
async def github_get_file(
    github_url: str,
    path: str,
    ref: str = "",
) -> str:
    """Read one repository text file through the GitHub Contents API."""
    owner, repo = _extract_owner_repo(github_url)
    normalized_path = _normalize_repository_path(path)
    params = {"ref": ref.strip()} if ref.strip() else None
    encoded_path = quote(normalized_path, safe="/")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{encoded_path}",
            headers=_get_headers(),
            params=params,
        )
        if response.status_code != 200:
            return _github_error("获取仓库文件", response)

        data = response.json()
        if not isinstance(data, dict) or data.get("type") != "file":
            return (
                f"获取仓库文件失败：{normalized_path} 不是文件；"
                "请使用 github_list_directory 查看目录。"
            )
        if data.get("encoding") != "base64" or not data.get("content"):
            return (
                f"获取仓库文件失败：GitHub 未以内联文本形式返回 {normalized_path}；"
                "文件可能过大或为二进制。"
            )

        raw = base64.b64decode(data["content"], validate=False)
        if b"\x00" in raw[:4096]:
            return (
                f"获取仓库文件失败：{normalized_path} 是二进制文件，"
                "无法作为文本证据读取。"
            )
        content = raw.decode("utf-8", errors="replace")
        truncated = len(content) > MAX_FILE_CHARS
        if truncated:
            content = content[:MAX_FILE_CHARS]
        suffix = (
            f"\n\n... [文件过长，已截断至前 {MAX_FILE_CHARS} 字符]"
            if truncated else ""
        )
        resolved_ref = ref.strip() or "默认分支"
        source_url = data.get("html_url") or (
            f"https://github.com/{owner}/{repo}/blob/{resolved_ref}/{normalized_path}"
        )
        return (
            f"--- {owner}/{repo} 文件：{normalized_path} ---\n"
            f"Ref：{resolved_ref}\n来源：{source_url}\n\n{content}{suffix}"
        )


@tool(description=(
    "列出 GitHub 仓库某个目录的文件和子目录，用于定位依赖清单、CI、Docker、"
    "示例和文档；支持指定 branch/tag/commit ref。"
))
async def github_list_directory(
    github_url: str,
    path: str = "",
    ref: str = "",
    limit: int = 100,
) -> str:
    """List one repository directory through the GitHub Contents API."""
    owner, repo = _extract_owner_repo(github_url)
    normalized_path = _normalize_repository_path(path, allow_empty=True)
    encoded_path = quote(normalized_path, safe="/")
    endpoint = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents"
    if encoded_path:
        endpoint += f"/{encoded_path}"
    params = {"ref": ref.strip()} if ref.strip() else None
    bounded_limit = max(1, min(int(limit), 200))

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(endpoint, headers=_get_headers(), params=params)
        if response.status_code != 200:
            return _github_error("获取仓库目录", response)
        items = response.json()
        if not isinstance(items, list):
            return (
                f"获取仓库目录失败：{normalized_path or '/'} 不是目录；"
                "请使用 github_get_file 读取文件。"
            )

        ordered = sorted(
            items,
            key=lambda item: (
                item.get("type") != "dir",
                item.get("name", ""),
            ),
        )
        lines = [
            f"--- {owner}/{repo} 目录：{normalized_path or '/'} ---",
            f"Ref：{ref.strip() or '默认分支'}",
        ]
        for item in ordered[:bounded_limit]:
            kind = "目录" if item.get("type") == "dir" else "文件"
            size = f" | {item.get('size', 0)} bytes" if kind == "文件" else ""
            item_path = item.get("path", item.get("name", ""))
            lines.append(f"- [{kind}] {item_path}{size}")
        if len(ordered) > bounded_limit:
            lines.append(f"... [其余 {len(ordered) - bounded_limit} 项未显示]")
        return "\n".join(lines)
