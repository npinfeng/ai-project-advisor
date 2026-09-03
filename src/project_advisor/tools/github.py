"""GitHub API 工具 — 获取仓库、Release、Issue 和 README 数据。"""

import os
from typing import Optional

import httpx
from langchain_core.tools import tool

from project_advisor import __version__

GITHUB_API_BASE = "https://api.github.com"


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

        import base64

        data = response.json()
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")

        if len(content) > 8000:
            content = content[:8000] + "\n\n... [README 过长，已截断至前 8000 字符]"

        return f"--- {owner}/{repo} README ---\n\n{content}"
