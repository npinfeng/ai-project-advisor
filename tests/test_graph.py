"""验证 LangGraph 工作流可以正确编译和运行。"""

import os
import sys
import pytest

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_graph_compiles():
    """验证 Graph 可以成功编译。"""
    from project_advisor.graph import graph
    assert graph is not None
    print("✅ Graph compiled successfully")


def test_schemas_work():
    """验证 Schema 模块可以正确导入和使用。"""
    from project_advisor.schemas.evidence import (
        CandidateProject,
        EvaluationCriteria,
        Evidence,
        ProjectScore,
        Requirements,
        calculate_weighted_score,
    )

    req = Requirements(
        language="Python",
        required_features=["multi_agent", "mcp"],
        deployment="self_hosted",
    )
    assert req.language == "Python"
    assert "multi_agent" in req.required_features

    ev = Evidence(
        source_url="https://github.com/langchain-ai/langgraph",
        source_type="github",
        project_name="LangGraph",
        content="LangGraph is a framework for building stateful, multi-actor applications.",
        relevance="feature_match",
        retrieved_at="2026-07-20T10:00:00",
    )
    assert ev.source_type == "github"

    criteria = EvaluationCriteria()
    score = ProjectScore(
        project_name="TestProject",
        feature_match=8.0, engineering_reliability=7.0,
        community_and_maintenance=6.0, documentation_quality=7.5,
        learning_cost=5.0, extensibility=8.0, deployment_cost=6.0,
    )
    weighted = calculate_weighted_score(score, criteria)
    assert 0 <= weighted <= 10
    print(f"✅ Schemas work correctly. Test weighted score: {weighted}")


def test_scoring_engine():
    """验证评分引擎的计算逻辑。"""
    from project_advisor.schemas.evidence import EvaluationCriteria, ProjectScore
    from project_advisor.tools.scoring import (
        compare_projects,
        create_default_criteria,
        format_score_table,
    )

    criteria = create_default_criteria()
    assert abs(sum([
        criteria.feature_match, criteria.engineering_reliability,
        criteria.community_and_maintenance, criteria.documentation_quality,
        criteria.learning_cost, criteria.extensibility, criteria.deployment_cost,
    ]) - 1.0) < 0.01, "权重总和应为 1.0"

    scores = [
        ProjectScore(project_name="A", feature_match=8, engineering_reliability=6,
                     community_and_maintenance=7, documentation_quality=5,
                     learning_cost=4, extensibility=8, deployment_cost=7),
        ProjectScore(project_name="B", feature_match=9, engineering_reliability=8,
                     community_and_maintenance=8, documentation_quality=9,
                     learning_cost=7, extensibility=7, deployment_cost=5),
    ]
    ranked = compare_projects(scores, criteria)
    assert ranked[0].project_name == "B"

    table = format_score_table(ranked)
    assert "B" in table
    print("✅ Scoring engine works correctly")


def test_citation_tools():
    """验证引用验证工具。"""
    from project_advisor.schemas.evidence import Evidence
    from project_advisor.tools.citations import (
        check_source_freshness,
        validate_citation,
        detect_conflicts,
    )

    ev = Evidence(
        source_url="https://example.com/doc",
        source_type="official_documentation",
        project_name="Test",
        content="Detailed documentation content.",
        relevance="documentation",
        retrieved_at="2026-07-20T10:00:00",
        source_date="2026-07-20T10:00:00",
    )

    freshness = check_source_freshness(ev)
    assert freshness["is_fresh"] is True

    validation = validate_citation(ev, "This project has good docs.")
    assert validation["is_valid"] is True

    # 测试冲突检测
    ev2 = Evidence(
        source_url="https://example.com/doc",
        source_type="official_documentation",
        project_name="Test",
        content="Different content from same URL.",
        relevance="documentation",
        retrieved_at="2026-07-20T10:00:00",
    )
    conflicts = detect_conflicts([ev, ev2])
    assert len(conflicts) > 0
    print("✅ Citation tools work correctly")


def test_github_url_parsing():
    """验证 GitHub URL 解析。"""
    from project_advisor.tools.github import _extract_owner_repo

    assert _extract_owner_repo("https://github.com/langchain-ai/langgraph") == ("langchain-ai", "langgraph")
    assert _extract_owner_repo("https://github.com/crewAIInc/crewAI.git") == ("crewAIInc", "crewAI")
    assert _extract_owner_repo("langchain-ai/langgraph") == ("langchain-ai", "langgraph")
    print("✅ GitHub URL parsing works correctly")


# ===== Phase 2 新增测试 =====

def test_document_collector():
    """验证文档采集工具的 URL 解析和内容检测。"""
    from project_advisor.tools.document_collector import (
        detect_document_type,
        extract_source_date,
        extract_domain,
        truncate_content,
        create_content_hash,
        validate_public_web_url,
    )
    from bs4 import BeautifulSoup
    import asyncio

    # URL 域名提取
    assert extract_domain("https://docs.langchain.com/oss/python/langgraph") == "docs.langchain.com"

    dated = BeautifulSoup(
        '<html><meta property="article:published_time" content="2026-07-01T08:00:00Z"></html>',
        "html.parser",
    )
    assert extract_source_date(dated) == "2026-07-01T08:00:00Z"
    with pytest.raises(ValueError, match="禁止访问"):
        asyncio.run(validate_public_web_url("http://127.0.0.1/internal"))
    assert asyncio.run(validate_public_web_url("https://8.8.8.8/docs"))
    assert extract_domain("https://github.com/langchain-ai/langgraph") == "github.com"

    # 文档类型检测
    assert detect_document_type(
        "https://docs.langchain.com/oss/python/langgraph/overview",
        "LangGraph Overview",
        "LangGraph is a framework..."
    ) == "official_documentation"

    assert detect_document_type(
        "https://github.com/langchain-ai/langgraph/releases/tag/v1.0",
        "Release v1.0",
        "Release notes..."
    ) == "release_note"

    assert detect_document_type(
        "https://blog.langchain.dev/langgraph-v1/",
        "Introducing LangGraph v1",
        "We're excited to announce..."
    ) == "blog"

    # 内容截断
    long_text = "Hello\n" * 10000
    truncated = truncate_content(long_text, 100)
    assert len(truncated) <= 150  # 100 chars + truncation message

    # 内容哈希
    h1 = create_content_hash("test content")
    h2 = create_content_hash("test content")
    h3 = create_content_hash("different content")
    assert h1 == h2
    assert h1 != h3

    print("✅ Document collector tools work correctly")


def test_document_store():
    """验证文档存储的增删查。"""
    import tempfile
    import shutil
    from project_advisor.schemas.evidence import Evidence
    from project_advisor.rag.document_store import DocumentStore

    tmp_dir = tempfile.mkdtemp()
    try:
        store = DocumentStore(storage_dir=tmp_dir)

        # 添加文档
        ev = Evidence(
            source_url="https://docs.example.com/test1",
            source_type="official_documentation",
            project_name="TestProject",
            content="This is a test document about multi-agent systems.",
            relevance="feature_match",
            confidence="high",
            retrieved_at="2026-07-20T10:00:00",
            version_info="v2.0",
        )
        assert store.add(ev) is True
        assert store.add(ev) is False  # 去重：不应重复添加

        # 批量添加
        ev2 = Evidence(
            source_url="https://docs.example.com/test2",
            source_type="blog",
            project_name="TestProject",
            content="Another document about RAG capabilities.",
            relevance="documentation",
            retrieved_at="2026-07-20T11:00:00",
        )
        count = store.add_batch([ev2])
        assert count == 1

        # 查询
        results = store.get_by_project("TestProject")
        assert len(results) == 2

        # 按类型过滤
        docs = store.get_by_project("TestProject", doc_type="blog")
        assert len(docs) == 1

        # 关键词搜索
        search_results = store.search("multi-agent")
        assert len(search_results) == 1

        # 统计
        stats = store.get_stats()
        assert stats["total_documents"] == 2

        # 清除
        store.clear_project("TestProject")
        assert len(store.get_by_project("TestProject")) == 0

        print("✅ Document store works correctly")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_tools_registered():
    """验证两个 Research Agent 的工具权限真正隔离。"""
    from project_advisor.utils import (
        get_documentation_tools,
        get_repository_tools,
    )
    from langchain_core.runnables import RunnableConfig
    import asyncio

    config = RunnableConfig(configurable={"search_api": "none"})

    async def run():
        repository = await get_repository_tools(config)
        documentation = await get_documentation_tools(config)
        repository_names = {
            getattr(tool, "name", getattr(tool, "__name__", ""))
            for tool in repository
        }
        documentation_names = {
            getattr(tool, "name", getattr(tool, "__name__", ""))
            for tool in documentation
        }

        assert "github_get_repo" in repository_names
        assert "web_fetch_tool" not in repository_names
        assert "rag_search" not in repository_names
        assert "web_fetch_tool" in documentation_names
        assert "batch_fetch_tool" in documentation_names
        assert "rag_search" in documentation_names
        assert "github_get_repo" not in documentation_names
        assert "rag_ingest" not in documentation_names
        assert "rag_rebuild" not in documentation_names

    asyncio.run(run())


def test_rag_modules():
    """验证 RAG 核心模块（分块、BM25、RRF、查询改写）。"""
    import tempfile
    import shutil
    from project_advisor.rag.chunker import DocumentChunker
    from project_advisor.rag.bm25_retriever import BM25Retriever
    from project_advisor.rag.hybrid_retriever import reciprocal_rank_fusion
    from project_advisor.rag.query_rewriter import QueryRewriter

    tmp_dir = tempfile.mkdtemp()
    try:
        # 1. 测试分块
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        test_doc = {
            "content": "## LangGraph\n\nLangGraph is a framework for building stateful agents.\n\n"
                       "## Features\n\nMulti-agent support with MCP integration.\n\n"
                       "## Deployment\n\nSelf-hosted with Docker support.",
            "source_url": "https://docs.langchain.com/langgraph",
            "source_type": "official_documentation",
            "project_name": "LangGraph",
        }
        chunks = chunker.chunk_document(test_doc["content"], metadata={
            "source_url": test_doc["source_url"],
            "source_type": test_doc["source_type"],
            "project_name": test_doc["project_name"],
        })
        assert len(chunks) >= 1
        assert all("text" in c for c in chunks)
        assert all("metadata" in c for c in chunks)
        print(f"Chunker: {len(chunks)} chunks created")

        # 2. 测试批量分块
        batch_chunks = chunker.chunk_documents([test_doc])
        assert len(batch_chunks) == len(chunks)
        print(f"Batch chunker: {len(batch_chunks)} chunks")

        # 3. 测试 BM25（多文档索引后进行精确匹配）
        bm25 = BM25Retriever(storage_dir=os.path.join(tmp_dir, "bm25"))
        # 添加更多文档以提高 BM25 的 IDF 质量
        chunks2 = chunker.chunk_document(
            "CrewAI is a framework for orchestrating AI agents with role-based collaboration. "
            "It supports multi-agent systems and task delegation.",
            metadata={"source_url": "https://docs.crewai.com", "source_type": "official_documentation", "project_name": "CrewAI"},
        )
        chunks3 = chunker.chunk_document(
            "Microsoft Agent Framework provides enterprise-grade multi-agent orchestration "
            "with built-in security and Azure integration.",
            metadata={"source_url": "https://learn.microsoft.com/agent-framework", "source_type": "official_documentation", "project_name": "MSAgent"},
        )
        all_chunks = chunks + chunks2 + chunks3
        bm25.index("AllProjects", all_chunks)
        results = bm25.search("multi-agent support", project_name="AllProjects", top_k=3)
        assert len(results) > 0
        print(f"BM25: {len(results)} results for 'multi-agent support'")

        # 4. 测试 RRF 融合
        list_a = [
            {"id": "a1", "text": "doc A1", "score": 0.9},
            {"id": "a2", "text": "doc A2", "score": 0.7},
            {"id": "a3", "text": "doc A3", "score": 0.5},
        ]
        list_b = [
            {"id": "b1", "text": "doc B1", "score": 0.8},
            {"id": "a1", "text": "doc A1", "score": 0.6},
            {"id": "a2", "text": "doc A2", "score": 0.4},
        ]
        fused = reciprocal_rank_fusion([list_a, list_b])
        assert fused[0]["id"] == "a1"
        print(f"RRF: {len(fused)} fused results, top: {fused[0]['id']}")

        # 5. 测试加权 RRF
        fused_weighted = reciprocal_rank_fusion(
            [list_a, list_b], weight_vector=[0.7, 0.3]
        )
        assert fused_weighted[0]["id"] == "a1"
        print("Weighted RRF: top result unchanged as expected")

        # 6. 测试查询改写（需要 OPENAI_API_KEY，跳过）
        print("QueryRewriter: requires API key, skipping live test")

        print("RAG modules work correctly")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_graph_compiles()
    test_schemas_work()
    test_scoring_engine()
    test_citation_tools()
    test_github_url_parsing()
    test_document_collector()
    test_document_store()
    test_tools_registered()
    test_rag_modules()
    print("\nAll tests passed!")
