"""RAG 搜索工具 — 供 Agent 使用的 LangChain 工具。

提供：
- rag_search: 搜索本地知识库（已索引的项目文档）
- rag_ingest: 将收集到的文档摄入到 RAG 索引
- rag_status: 查看 RAG 索引状态
"""

from typing import Optional

from langchain_core.tools import tool

from project_advisor.rag.ingestion import IngestionPipeline
from project_advisor.rag.query_rewriter import QueryRewriter
from project_advisor.rag.reranker import Reranker

# 全局单例
_pipeline: Optional[IngestionPipeline] = None
_rewriter: Optional[QueryRewriter] = None
_reranker: Optional[Reranker] = None


def _get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


def _get_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def _sync_from_store(
    project_name: str = "",
    *,
    force: bool = False,
) -> list[dict]:
    """Synchronize persisted Evidence into both vector and BM25 indexes."""
    from project_advisor.rag.document_store import DocumentStore

    store = DocumentStore()
    available_projects = store.get_stats().get("projects", [])
    if project_name:
        resolved = next(
            (
                name for name in available_projects
                if name.casefold() == project_name.casefold()
            ),
            project_name,
        )
        targets = [resolved] if resolved in available_projects else []
    else:
        targets = available_projects
    if not targets:
        return []

    pipeline = _get_pipeline()
    sync_results = []
    for target in targets:
        if force:
            pipeline.hybrid.clear(target)
        stats = pipeline.ingest_from_store(store, target)
        sync_results.append({"project_name": target, **stats})
    return sync_results


@tool(description="Search the local knowledge base for technical documentation. Use this for precise queries about specific capabilities, APIs, or architecture details of candidate projects.")
def rag_search(
    query: str,
    project_name: str = "",
    top_k: int = 5,
) -> str:
    """搜索本地 RAG 知识库。

    使用混合检索（BM25 + 向量）查找相关技术文档。
    适用场景：
    - "LangGraph 如何实现 checkpoint？"
    - "CrewAI 支持哪些 LLM 提供商？"
    - "Microsoft Agent Framework 的部署方式"

    Args:
        query: 搜索查询
        project_name: 限制搜索某个项目（可选）
        top_k: 返回结果数

    Returns:
        格式化的搜索结果
    """
    sync_results = _sync_from_store(project_name)
    if not sync_results:
        scope = f"项目 {project_name}" if project_name else "任何项目"
        return f"本地知识库中还没有 {scope} 的持久化证据。"

    pipeline = _get_pipeline()
    rewriter = _get_rewriter()

    # 查询改写
    rewritten = rewriter.rewrite_sync(query)

    # 混合检索
    proj = sync_results[0]["project_name"] if project_name else None
    results = pipeline.search(
        query=rewritten,
        project_name=proj,
        top_k=top_k,
    )

    if not results:
        return f"未在本地知识库中找到与 '{query}' 相关的结果。知识库可能尚未索引相关文档。"

    # 重排序
    reranker = _get_reranker()
    reranked = reranker.rerank_sync(query, results, top_k=min(top_k, len(results)))

    # 格式化输出
    lines = [f"RAG 搜索结果（查询：{query}）：\n"]
    for i, result in enumerate(reranked):
        source = result.get("metadata", {}).get("source_url", "未知")
        proj = result.get("project", result.get("metadata", {}).get("project_name", ""))
        score = result.get("rerank_score", result.get("rrf_score", result.get("score", 0)))
        text = result.get("text", "")[:800]
        lines.append(
            f"## 结果 {i + 1}（相关度：{score:.2f}）\n"
            f"- 项目：{proj}\n"
            f"- 来源：{source}\n"
            f"- 内容：{text}\n"
        )

    return "\n".join(lines)


@tool(description="Ingest documents into the RAG knowledge base. Call this after collecting documentation to make it searchable.")
def rag_ingest(
    project_name: str,
) -> str:
    """将已存储的文档摄入到 RAG 知识库。

    从 DocumentStore 读取指定项目的所有文档，
    进行分块、嵌入和索引。

    Args:
        project_name: 项目名称

    Returns:
        索引统计信息
    """
    sync_results = _sync_from_store(project_name)
    stats = sync_results[0] if sync_results else {}

    return (
        f"RAG 索引完成。\n"
        f"- 项目：{project_name}\n"
        f"- 分块数：{stats.get('chunks', 0)}\n"
        f"- 向量索引：{stats.get('vector_indexed', 0)}\n"
        f"- 向量总数：{stats.get('vector_total', 0)}\n"
        f"- 清理旧向量：{stats.get('vector_removed', 0)}\n"
        f"- BM25 索引：{stats.get('bm25_indexed', 0)}"
    )


@tool(description="Rebuild persistent vector and BM25 indexes from stored Evidence. Use after changing chunking or embedding configuration.")
def rag_rebuild(project_name: str = "") -> str:
    """强制从 DocumentStore 重建一个项目或全部项目的检索索引。"""
    results = _sync_from_store(project_name, force=True)
    if not results:
        return "没有可重建的持久化证据。"
    lines = ["RAG 索引重建完成："]
    for result in results:
        lines.append(
            f"- {result['project_name']}：{result.get('chunks', 0)} 个分块，"
            f"{result.get('vector_total', 0)} 个向量，"
            f"{result.get('bm25_indexed', 0)} 个 BM25 文档"
        )
    return "\n".join(lines)


@tool(description="Check the status of the RAG knowledge base — which projects are indexed and how many documents.")
def rag_status() -> str:
    """查看 RAG 知识库的状态。

    Returns:
        格式化的状态信息
    """
    from project_advisor.rag.document_store import DocumentStore
    from project_advisor.rag.bm25_retriever import BM25Retriever
    from project_advisor.rag.vector_store import VectorStore

    store = DocumentStore()
    vs = VectorStore()
    bm25 = BM25Retriever()

    store_stats = store.get_stats()
    lines = [
        "RAG 知识库状态：\n",
        f"存储目录：{store_stats['storage_dir']}",
        f"文档总数：{store_stats['total_documents']}",
        f"向量分块总数：{vs.count()}",
        f"BM25 分块总数：{bm25.count()}",
        "\n各项目详情：",
    ]

    for proj, count in store_stats.get("docs_per_project", {}).items():
        vs_count = vs.count(proj)
        bm25_count = bm25.count(proj)
        state = "已建立双路索引" if vs_count and bm25_count else "待同步"
        lines.append(
            f"  - {proj}：{count} 个证据，{vs_count} 个向量，"
            f"{bm25_count} 个 BM25 分块（{state}）"
        )

    return "\n".join(lines)
