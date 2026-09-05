"""Async RAG search tool used by the Documentation Researcher.

提供：
- rag_search: 搜索本地知识库（已索引的项目文档）
- rag_ingest: 将收集到的文档摄入到 RAG 索引
- rag_status: 查看 RAG 索引状态
- rag_maintain: 预览或执行 Evidence 过期清理并重建受影响索引
"""

import asyncio
import logging
from typing import Optional

from langchain_core.runnables import RunnableConfig

from langchain_core.tools import tool

from project_advisor.configuration import Configuration
from project_advisor.observability.logging import log_event
from project_advisor.rag.embedder import Embedder
from project_advisor.rag.ingestion import IngestionPipeline
from project_advisor.rag.query_rewriter import QueryRewriter
from project_advisor.rag.reranker import Reranker
from project_advisor.tools.source_result import source_tool, sourced

# 全局单例
_pipeline: Optional[IngestionPipeline] = None
_rewriter: Optional[QueryRewriter] = None
_reranker: Optional[Reranker] = None
logger = logging.getLogger(__name__)


def _get_pipeline(config: Optional[Configuration] = None) -> IngestionPipeline:
    global _pipeline
    configurable = config or Configuration.from_runnable_config()
    embedder = Embedder(
        model_name=configurable.embedding_model,
        provider=configurable.embedding_provider,
        normalize_embeddings=configurable.embedding_normalize,
        query_instruction=configurable.embedding_query_instruction,
    )
    if (
        _pipeline is None
        or _pipeline.embedder.index_identity != embedder.index_identity
    ):
        if _pipeline is not None:
            _pipeline.hybrid.vector_store.close()
        _pipeline = IngestionPipeline(embedder=embedder)
    return _pipeline


def _get_rewriter(config: Configuration) -> QueryRewriter:
    global _rewriter
    if (
        _rewriter is None
        or _rewriter.model_name != config.research_model
        or _rewriter.timeout_seconds != config.llm_timeout_seconds
        or _rewriter.max_attempts != config.max_structured_output_retries
    ):
        _rewriter = QueryRewriter(
            model_name=config.research_model,
            timeout_seconds=config.llm_timeout_seconds,
            max_attempts=config.max_structured_output_retries,
        )
    return _rewriter


def _get_reranker(config: Configuration) -> Reranker:
    global _reranker
    if (
        _reranker is None
        or _reranker.model_name != config.research_model
        or _reranker.timeout_seconds != config.llm_timeout_seconds
        or _reranker.max_attempts != config.max_structured_output_retries
        or _reranker.max_concurrency != config.max_concurrent_research_units
    ):
        _reranker = Reranker(
            model_name=config.research_model,
            timeout_seconds=config.llm_timeout_seconds,
            max_attempts=config.max_structured_output_retries,
            max_concurrency=config.max_concurrent_research_units,
        )
    return _reranker


def _sync_from_store(
    project_name: str = "",
    *,
    force: bool = False,
    config: RunnableConfig = None,
) -> list[dict]:
    """Synchronize persisted Evidence into both vector and BM25 indexes."""
    from project_advisor.rag.document_store import DocumentStore
    from project_advisor.rag.evidence_lifecycle import EvidenceLifecyclePolicy

    store = DocumentStore()
    configurable = Configuration.from_runnable_config(config)
    policy = EvidenceLifecyclePolicy(
        stale_after_days=configurable.evidence_stale_after_days,
        expire_after_days=configurable.evidence_expire_after_days,
    )
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

    pipeline = _get_pipeline(configurable)
    sync_results = []
    for target in targets:
        if force:
            pipeline.hybrid.clear(target)
        stats = pipeline.ingest_from_store(store, target, policy=policy)
        sync_results.append({"project_name": target, **stats})
    return sync_results


@source_tool(description=(
    "Search the local knowledge base for technical documentation. Use this for "
    "precise queries about capabilities, APIs, or architecture of candidates."
))
async def rag_search(
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
    query = query.strip()
    if not query:
        raise ValueError("RAG 查询不能为空。")
    top_k = max(1, min(int(top_k), 20))
    configurable = Configuration.from_runnable_config(config)
    # Keep synchronization callable with a single project argument so it remains
    # easy to replace in offline tests; environment policy is read inside it.
    sync_results = await asyncio.to_thread(
        _sync_from_store,
        project_name,
        config=config,
    )
    if not sync_results:
        scope = f"项目 {project_name}" if project_name else "任何项目"
        return f"本地知识库中还没有 {scope} 的持久化证据。"

    pipeline = _get_pipeline(configurable)
    rewriter = _get_rewriter(configurable)

    # 生成多角度子查询
    proj = sync_results[0]["project_name"] if project_name else None
    generated_queries = await rewriter.generate_multi_queries(query)
    if len(generated_queries) <= 1:
        generated_queries.append(await rewriter.rewrite(query))
    # The original wording is always searched. This matters for Chinese proper
    # nouns and exact phrases that an LLM rewrite may translate or paraphrase.
    sub_queries = []
    seen_queries = set()
    for value in [query, *generated_queries]:
        normalized = " ".join(str(value).split()).casefold()
        if normalized and normalized not in seen_queries:
            seen_queries.add(normalized)
            sub_queries.append(str(value).strip())
        if len(sub_queries) >= 6:
            break

    # 用每个子查询并行搜索，按 chunk_id 去重合并
    merged_by_id: dict[str, dict] = {}
    for query_index, sq in enumerate(sub_queries):
        batch = await asyncio.to_thread(
            pipeline.search,
            query=sq,
            project_name=proj,
            top_k=max(top_k, 3),
        )
        for rank, item in enumerate(batch, start=1):
            chunk_id = item.get("id", "")
            if not chunk_id:
                continue
            retrieval_score = float(item.get(
                "hybrid_score", item.get("rrf_score", item.get("score", 0))
            ))
            if chunk_id not in merged_by_id:
                merged_by_id[chunk_id] = {
                    **item,
                    "multi_query_hits": 0,
                    "multi_query_score_sum": 0.0,
                    "multi_query_best_score": 0.0,
                    "multi_query_rank_score": 0.0,
                    "original_query_hit": False,
                }
            existing = merged_by_id[chunk_id]
            existing["multi_query_hits"] += 1
            existing["multi_query_score_sum"] += retrieval_score
            existing["multi_query_best_score"] = max(existing["multi_query_best_score"], retrieval_score)
            existing["multi_query_rank_score"] += 1.0 / (60 + rank)
            existing["original_query_hit"] = existing["original_query_hit"] or query_index == 0

    merged = list(merged_by_id.values())
    query_count = max(1, len(sub_queries))
    for item in merged:
        average_score = item["multi_query_score_sum"] / item["multi_query_hits"]
        coverage = item["multi_query_hits"] / query_count
        item["multi_query_score"] = (
            0.60 * item["multi_query_best_score"]
            + 0.25 * average_score
            + 0.15 * coverage
        )
    merged.sort(key=lambda item: (
        -item["multi_query_score"],
        -int(item["original_query_hit"]),
        -item["multi_query_rank_score"],
        str(item["id"]),
    ))

    if not merged:
        return f"未在本地知识库中找到与 '{query}' 相关的结果。知识库可能尚未索引相关文档。"

    # 重排序（取合并后的 Top 20）
    reranker = _get_reranker(configurable)
    candidates = merged[:20]
    reranked = await reranker.rerank(
        query,
        candidates,
        top_k=min(top_k, len(candidates)),
    )
    log_event(
        logger,
        logging.INFO,
        "rag_search_completed",
        project_name=project_name,
        query_count=len(sub_queries),
        candidate_count=len(candidates),
        result_count=len(reranked),
    )

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

    from project_advisor.rag.document_store import DocumentStore
    evidence_ids = {item.get("metadata", {}).get("evidence_id", "") for item in reranked} - {""}
    originals = await asyncio.to_thread(lambda: DocumentStore().get_by_ids(evidence_ids)) if evidence_ids else []
    return sourced("\n".join(lines), reused_evidences=originals)


@tool(description=(
    "Ingest documents into the RAG knowledge base after collecting documentation."
))
def rag_ingest(
    project_name: str,
    config: RunnableConfig = None,
) -> str:
    """将已存储的文档摄入到 RAG 知识库。

    从 DocumentStore 读取指定项目的所有文档，
    进行分块、嵌入和索引。

    Args:
        project_name: 项目名称

    Returns:
        索引统计信息
    """
    sync_results = _sync_from_store(project_name, config=config)
    stats = sync_results[0] if sync_results else {}

    return (
        f"RAG 索引完成。\n"
        f"- 项目：{project_name}\n"
        f"- 分块数：{stats.get('chunks', 0)}\n"
        f"- 向量索引：{stats.get('vector_indexed', 0)}\n"
        f"- 向量总数：{stats.get('vector_total', 0)}\n"
        f"- 清理旧向量：{stats.get('vector_removed', 0)}\n"
        f"- Embedding 变化触发重建："
        f"{'是' if stats.get('vector_rebuilt_for_embedding_change') else '否'}\n"
        f"- BM25 索引：{stats.get('bm25_indexed', 0)}"
    )


@tool(description=(
    "Rebuild persistent vector and BM25 indexes from stored Evidence after changing "
    "chunking or embedding configuration."
))
def rag_rebuild(project_name: str = "", config: RunnableConfig = None) -> str:
    """强制从 DocumentStore 重建一个项目或全部项目的检索索引。"""
    results = _sync_from_store(project_name, force=True, config=config)
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


@tool(description=(
    "Preview or apply Evidence lifecycle cleanup. Applying removes invalid/expired "
    "records and rebuilds vector/BM25 indexes for affected projects."
))
def rag_maintain(
    project_name: str = "",
    apply: bool = False,
    stale_after_days: int | None = None,
    expire_after_days: int | None = None,
    config: RunnableConfig = None,
) -> str:
    """清理无效/过期证据；默认 dry-run，apply=true 才会删除并重建。"""
    from project_advisor.rag.document_store import DocumentStore
    from project_advisor.rag.evidence_lifecycle import EvidenceLifecyclePolicy

    configurable = Configuration.from_runnable_config(config)
    policy = EvidenceLifecyclePolicy(
        stale_after_days=(
            configurable.evidence_stale_after_days
            if stale_after_days is None else stale_after_days
        ),
        expire_after_days=(
            configurable.evidence_expire_after_days
            if expire_after_days is None else expire_after_days
        ),
    )
    store = DocumentStore()
    report = store.maintain(project_name, policy=policy, dry_run=not apply)
    rebuilt: list[dict] = []
    if apply and report["affected_projects"]:
        pipeline = _get_pipeline(configurable)
        for project in report["affected_projects"]:
            pipeline.hybrid.clear(project)
            current = store.get_current_by_project(project, policy=policy, max_results=100_000)
            if current:
                stats = pipeline.ingest_evidence(project, current)
                rebuilt.append({"project_name": project, **stats})

    counts = report["before"]["counts"]
    mode = "已执行" if apply else "仅预览"
    lines = [
        f"Evidence 生命周期维护（{mode}）：",
        f"- active/stale/expired/invalid：{counts['active']}/"
        f"{counts['stale']}/{counts['expired']}/{counts['invalid']}",
        f"- 将移除：{report['removed_count']} 条",
        f"- 版本冲突：{len(report['before']['version_conflicts'])} 组",
    ]
    if apply:
        lines.append(f"- 已重建索引：{len(rebuilt)} 个项目")
    else:
        lines.append("- 传入 apply=true 后才会删除，并自动重建受影响索引。")
    return "\n".join(lines)


@tool(description=(
    "Check indexed RAG projects, document freshness, and stale document counts."
))
def rag_status(config: RunnableConfig = None) -> str:
    """查看 RAG 知识库的状态，包含新鲜度信息。

    Returns:
        格式化的状态信息
    """
    from project_advisor.rag.document_store import DocumentStore
    from project_advisor.rag.evidence_lifecycle import EvidenceLifecyclePolicy

    store = DocumentStore()
    store_stats = store.get_stats()
    configurable = Configuration.from_runnable_config(config)
    pipeline = _get_pipeline(configurable)
    vs = pipeline.hybrid.vector_store
    bm25 = pipeline.hybrid.bm25
    policy = EvidenceLifecyclePolicy(
        stale_after_days=configurable.evidence_stale_after_days,
        expire_after_days=configurable.evidence_expire_after_days,
    )

    lines = [
        "RAG 知识库状态：\n",
        f"Embedding：{configurable.embedding_provider}/"
        f"{configurable.embedding_model}（归一化："
        f"{'是' if configurable.embedding_normalize else '否'}）",
        f"存储目录：{store_stats['storage_dir']}",
        f"文档总数：{store_stats['total_documents']}",
        f"向量分块总数：{vs.count()}",
        f"BM25 分块总数：{bm25.count()}",
        "\n各项目详情：",
    ]

    total_stale = 0
    total_expired = 0
    total_invalid = 0
    for proj, count in store_stats.get("docs_per_project", {}).items():
        vs_count = vs.count(proj)
        bm25_count = bm25.count(proj)
        state = "已建立双路索引" if vs_count and bm25_count else "待同步"

        try:
            lifecycle = store.lifecycle_status(proj, policy=policy)
            lifecycle_counts = lifecycle["counts"]
            stale_count = lifecycle_counts["stale"]
            expired_count = lifecycle_counts["expired"]
            invalid_count = lifecycle_counts["invalid"]
            total_stale += stale_count
            total_expired += expired_count
            total_invalid += invalid_count
            freshness = (
                f"active/stale/expired/invalid：{lifecycle_counts['active']}/"
                f"{stale_count}/{expired_count}/{invalid_count}，"
                f"可索引当前版本：{lifecycle['indexable_current_count']}，"
                f"版本冲突：{len(lifecycle['version_conflicts'])}"
            )
        except Exception:
            freshness = "无法获取新鲜度"

        lines.append(
            f"  - {proj}：{count} 个证据，{vs_count} 个向量，"
            f"{bm25_count} 个 BM25 分块（{state}）\n"
            f"    {freshness}"
        )

    if total_stale or total_expired or total_invalid:
        lines.append(
            f"\n⚠ stale {total_stale} 条、expired {total_expired} 条、"
            f"invalid {total_invalid} 条。先运行 rag_maintain 预览；"
            "确认后使用 apply=true 清理并重建索引。"
        )

    return "\n".join(lines)
