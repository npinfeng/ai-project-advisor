"""Tests for role skills and newly exposed research tools."""

import asyncio
import base64

from project_advisor.skills import load_skill, load_skills_for_role
from project_advisor.tools import github_get_file, github_list_directory, web_discover_links
from project_advisor.tools import document_collector
from project_advisor.tools import github as github_module
from project_advisor.tools.evidence_factory import is_error_tool_result
from project_advisor.utils import get_documentation_tools, get_repository_tools


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = "test response"

    def json(self):
        return self._payload


class _AsyncClient:
    payload = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, *args, **kwargs):
        return _Response(self.payload)


def test_role_skills_are_scoped_and_frontmatter_is_not_injected():
    repository = load_skills_for_role("repository")
    documentation = load_skills_for_role("documentation")

    assert "repository-due-diligence" in repository
    assert "rag-architecture-review" not in repository
    assert "rag-architecture-review" in documentation
    assert "parent-child" in documentation
    assert "description:" not in repository
    assert load_skill("official-doc-verification").startswith("# Official")


def test_new_tools_are_exposed_to_the_correct_researcher():
    async def run():
        config = {"configurable": {"search_api": "none"}}
        repository = {
            getattr(tool, "name", None) or getattr(tool, "__name__", "")
            for tool in await get_repository_tools(config)
        }
        documentation = {
            getattr(tool, "name", None) or getattr(tool, "__name__", "")
            for tool in await get_documentation_tools(config)
        }
        assert {"github_get_file", "github_list_directory"} <= repository
        assert "web_discover_links" in documentation
        assert "web_discover_links" not in repository

    asyncio.run(run())


def test_github_get_file_decodes_text(monkeypatch):
    payload = {
        "type": "file",
        "encoding": "base64",
        "content": base64.b64encode(b"[project]\nname='demo'\n").decode(),
        "html_url": "https://github.com/acme/demo/blob/main/pyproject.toml",
    }
    _AsyncClient.payload = payload
    monkeypatch.setattr(github_module.httpx, "AsyncClient", _AsyncClient)

    result = asyncio.run(github_get_file.ainvoke({
        "github_url": "acme/demo",
        "path": "pyproject.toml",
    }))
    assert "name='demo'" in result
    assert "https://github.com/acme/demo/blob/main/pyproject.toml" in result


def test_github_list_directory_orders_directories_first(monkeypatch):
    _AsyncClient.payload = [
        {"type": "file", "name": "pyproject.toml", "path": "pyproject.toml", "size": 42},
        {"type": "dir", "name": "src", "path": "src", "size": 0},
    ]
    monkeypatch.setattr(github_module.httpx, "AsyncClient", _AsyncClient)
    result = asyncio.run(github_list_directory.ainvoke({"github_url": "acme/demo"}))
    assert result.index("[目录] src") < result.index("[文件] pyproject.toml")


def test_web_discover_links_filters_domain_and_keyword(monkeypatch):
    async def fake_fetch(url):
        return {
            "status": 200,
            "url": "https://docs.example.com/",
            "links": [
                {"text": "Migration guide", "url": "https://docs.example.com/migrate"},
                {"text": "API", "url": "https://docs.example.com/api"},
                {"text": "External", "url": "https://other.example.net/migrate"},
            ],
        }

    monkeypatch.setattr(document_collector, "fetch_webpage", fake_fetch)
    result = asyncio.run(web_discover_links.ainvoke({
        "url": "https://docs.example.com/",
        "keyword": "migration",
    }))
    assert "docs.example.com/migrate" in result
    assert "other.example.net" not in result


def test_new_tool_failures_are_not_evidence():
    assert is_error_tool_result("获取仓库文件失败：不存在（404）。")
    assert is_error_tool_result("获取仓库目录失败：不是目录。")
    assert is_error_tool_result("无法发现链接：请求失败。")
    assert is_error_tool_result("页面中未发现符合条件的链接：https://example.com")
