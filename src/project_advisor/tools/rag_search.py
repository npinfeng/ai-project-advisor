"""RAG 搜索工具 — 供 Agent 使用的 LangChain 工具。

提供：
- rag_search: 搜索本地知识库（已索引的项目文档）
- rag_ingest: 将收集到的文档摄入到 RAG 索引
- rag_status: 查看 RAG 索引状态
"""

from typing import Optional

from langchain_core.runnables import RunnableConfig

from langchain_core.tools import tool

from project_advisor.configuration import Configuration
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


def _get_rewriter(model_name: str) -> QueryRewriter:
    global _rewriter
    if _rewriter is None or _rewriter.model_name != model_name:
        _rewriter = QueryRewriter(model_name=model_name)
    return _rewriter


def _get_reranker(model_name: str) -> Reranker:
    global _reranker
    if _reranker is None or _reranker.model_name != model_name:
        _reranker = Reranker(model_name=model_name)
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
    config: RunnableConfig = None,
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
    configurable = Configuration.from_runnable_config(config)
    rewriter = _get_rewriter(configurable.research_model)

    # 生成多角度子查询
    proj = sync_results[0]["project_name"] if project_name else None
    sub_queries = rewriter.multi_query_sync(query)
    if len(sub_queries) <= 1:
        sub_queries = [rewriter.rewrite_sync(query)]

    # 用每个子查询并行搜索，按 chunk_id 去重合并
    seen_ids: set[str] = set()
    merged: list[dict] = []
    for sq in sub_queries:
        batch = pipeline.search(
            query=sq,
            project_name=proj,
            top_k=max(top_k, 3),
        )
        for item in batch:
            chunk_id = item.get("id", "")
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                # 出现在更多子查询中的结果得分加权
                item["multi_query_hits"] = item.get("multi_query_hits", 0) + 1
                merged.append(item)
            elif chunk_id in seen_ids:
                # 已在结果中，增加多查询命中计数
                for existing in merged:
                    if existing.get("id") == chunk_id:
                        existing["multi_query_hits"] = existing.get("multi_query_hits", 1) + 1
                        existing["score"] = max(existing.get("score", 0), item.get("score", 0))
                        break

    # 按多查询命中数 + 原始分数排序
    merged.sort(key=lambda x: (x.get("multi_query_hits", 1), x.get("score", 0)), reverse=True)

    if not merged:
        return f"未在本地知识库中找到与 '{query}' 相关的结果。知识库可能尚未索引相关文档。"

    # 重排序（取合并后的 Top 20）
    reranker = _get_reranker(configurable.research_model)
    candidates = merged[:20]
    reranked = reranker.rerank_sync(query, candidates, top_k=min(top_k, len(candidates)))

    # 格式化输出（含检索时间，供 Reviewer 评估新鲜度）
    lines = [f"RAG 搜索结果（查询：{query}）：\n"]
    for i, result in enumerate(reranked):
        metadata = result.get("metadata", {})
        source = metadata.get("source_url", "未知")
        proj = result.get("project", metadata.get("project_name", ""))
        score = result.get("rerank_score", result.get("rrf_score", result.get("score", 0)))
        retrieved_at = metadata.get("retrieved_at", "未知")
        text = result.get("text", "")[:800]
        time_decay = result.get("time_decay")
        freshness_note = ""
        if time_decay is not None and time_decay < 0.8:
            freshness_note = f"（新鲜度衰减：{time_decay:.2f}，可能已过时）"
        lines.append(
            f"## 结果 {i + 1}（相关度：{score:.2f}）{freshness_note}\n"
            f"- 项目：{proj}\n"
            f"- 来源：{source}\n"
            f"- 检索时间：{retrieved_at}\n"
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


@tool(description="Check the status of the RAG knowledge base — which projects are indexed, document freshness, and how many documents may be stale.")
def rag_status() -> str:
    """查看 RAG 知识库的状态，包含新鲜度信息。

    Returns:
        格式化的状态信息
    """
    from datetime import datetime, timezone

    from project_advisor.rag.document_store import DocumentStore
    from project_advisor.rag.bm25_retriever import BM25Retriever
    from project_advisor.rag.vector_store import VectorStore

    store = DocumentStore()
    vs = VectorStore()
    bm25 = BM25Retriever()

    store_stats = store.get_stats()
    now = datetime.now(timezone.utc)
    stale_threshold_days = 180

    lines = [
        "RAG 知识库状态：\n",
        f"存储目录：{store_stats['storage_dir']}",
        f"文档总数：{store_stats['total_documents']}",
        f"向量分块总数：{vs.count()}",
        f"BM25 分块总数：{bm25.count()}",
        "\n各项目详情：",
    ]

    total_stale = 0
    for proj, count in store_stats.get("docs_per_project", {}).items():
        vs_count = vs.count(proj)
        bm25_count = bm25.count(proj)
        state = "已建立双路索引" if vs_count and bm25_count else "待同步"

        # 检查该项目下的过期文档
        try:
            docs = store.get_by_project(proj, max_results=1000)
            stale_count = 0
            newest = "未知"
            oldest = "未知"
            for doc in docs:
                retrieved = getattr(doc, "retrieved_at", "")
                if retrieved:
                    try:
                        dt = datetime.fromisoformat(retrieved)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        age = (now - dt).days
                        if age > stale_threshold_days:
                            stale_count += 1
                        if newest == "未知" or (isinstance(newest, str)):
                            newest = retrieved[:10]
                        oldest = retrieved[:10]
                    except (ValueError, TypeError):
                        pass
            total_stale += stale_count
            freshness = f"最新：{newest}，最旧：{oldest}"
            if stale_count > 0:
                freshness += f"，⚠ 过期文档：{stale_count}/{count}"
            else:
                freshness += "，✓ 全部新鲜"
        except Exception:
            freshness = "无法获取新鲜度"

        lines.append(
            f"  - {proj}：{count} 个证据，{vs_count} 个向量，"
            f"{bm25_count} 个 BM25 分块（{state}）\n"
            f"    {freshness}"
        )

    if total_stale > 0:
        lines.append(
            f"\n⚠ 共 {total_stale} 条证据超过 {stale_threshold_days} 天，"
            f"建议对相关项目运行 rag_rebuild 以获取最新数据。"
        )

    return "\n".join(lines)
