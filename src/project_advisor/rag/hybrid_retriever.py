"""混合检索引擎 — BM25 + 向量检索 + Reciprocal Rank Fusion + 时间衰减。

核心思路：
1. 同时执行 BM25 关键词检索和向量语义检索
2. 使用 RRF 算法融合两路结果（含时间衰减因子）
3. 对融合后的候选文档进行重排序
4. 支持元数据过滤（项目、文档类型、日期、新鲜度）

这是整个 RAG 系统的主要对外接口。
"""

import math
from datetime import datetime, timezone
from typing import Optional

from project_advisor.rag.bm25_retriever import BM25Retriever
from project_advisor.rag.chunker import DocumentChunker
from project_advisor.rag.embedder import Embedder
from project_advisor.rag.vector_store import VectorStore

# 时间衰减半衰期（天）：超过此天数的文档 RRF 得分衰减至一半
TIME_DECAY_HALF_LIFE_DAYS = 90
DEFAULT_CANDIDATE_POOL_FACTOR = 4
DEFAULT_MIN_RESULTS_PER_SOURCE = 1


def _compute_time_decay(
    retrieved_at: Optional[str],
    half_life_days: float = TIME_DECAY_HALF_LIFE_DAYS,
) -> float:
    """根据文档检索时间计算新鲜度衰减因子。

    公式：decay = 0.5 ^ (age_days / half_life_days)
    - 今天检索的文档：decay = 1.0
    - N 天前检索的文档：decay = 0.5 ^ (N / half_life)
    - 无法解析时间的文档：decay = 0.5（保守降权）

    Args:
        retrieved_at: ISO 格式的检索时间字符串
        half_life_days: 半衰期天数

    Returns:
        0.0~1.0 之间的衰减因子
    """
    if not retrieved_at:
        return 0.5
    try:
        retrieved_dt = datetime.fromisoformat(retrieved_at)
        now = datetime.now(timezone.utc)
        if retrieved_dt.tzinfo is None:
            retrieved_dt = retrieved_dt.replace(tzinfo=timezone.utc)
        age_days = (now - retrieved_dt).total_seconds() / 86400.0
        if age_days <= 0:
            return 1.0
        return math.pow(0.5, age_days / half_life_days)
    except (ValueError, TypeError):
        return 0.5


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    weight_vector: Optional[list[float]] = None,
    enable_time_decay: bool = True,
) -> list[dict]:
    """Reciprocal Rank Fusion（RRF）— 融合多路检索结果（含时间衰减）。

    公式：score(d) = sum(w_i / (k + rank_i(d))) * time_decay(d)
    其中 k 是平滑参数（默认 60），rank_i 是文档在第 i 路结果中的排名。
    time_decay(d) 根据文档的 retrieved_at 计算新鲜度惩罚。

    Args:
        result_lists: 多路检索结果列表
        k: RRF 平滑参数
        weight_vector: 各路结果的权重，如 [0.6, 0.4] 表示向量检索权重更高
        enable_time_decay: 是否启用时间衰减

    Returns:
        按 RRF 分数降序排列的融合结果
    """
    if not result_lists:
        return []

    if weight_vector is None:
        weight_vector = [1.0] * len(result_lists)

    scores: dict[str, dict] = {}

    for list_idx, (results, weight) in enumerate(
        zip(result_lists, weight_vector)
    ):
        for rank, item in enumerate(results, start=1):
            doc_id = item.get("id", f"unknown_{list_idx}_{rank}")
            if doc_id not in scores:
                scores[doc_id] = {
                    "item": item,
                    "rrf_score": 0.0,
                }
            scores[doc_id]["rrf_score"] += weight / (k + rank)

    # 时间衰减：越旧的文档得分越低
    if enable_time_decay:
        for doc_id, entry in scores.items():
            retrieved_at = (
                entry["item"]
                .get("metadata", {})
                .get("retrieved_at", "")
            )
            decay = _compute_time_decay(retrieved_at)
            entry["rrf_score"] *= decay
            entry["time_decay"] = decay

    ranked = sorted(
        scores.values(), key=lambda x: x["rrf_score"], reverse=True
    )

    return [
        {**entry["item"], "rrf_score": entry["rrf_score"]}
        for entry in ranked
    ]


def _normalize_result_scores(results: list[dict]) -> list[float]:
    """Normalize one retriever's scores to 0..1 within that source."""
    if not results:
        return []
    raw_scores = []
    for item in results:
        try:
            value = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        raw_scores.append(value if math.isfinite(value) else 0.0)
    minimum = min(raw_scores)
    maximum = max(raw_scores)
    if math.isclose(maximum, minimum):
        return [1.0 if maximum > 0 else 0.0] * len(raw_scores)
    scale = maximum - minimum
    return [(value - minimum) / scale for value in raw_scores]


def score_rank_fusion(
    result_lists: list[list[dict]],
    *,
    source_names: Optional[list[str]] = None,
    weight_vector: Optional[list[float]] = None,
    rank_weight: float = 0.5,
    score_weight: float = 0.5,
    rrf_k: int = 60,
    enable_time_decay: bool = True,
) -> list[dict]:
    """Fuse independently normalized scores with normalized rank evidence."""
    if not result_lists:
        return []
    source_names = source_names or [
        f"source_{index}" for index in range(len(result_lists))
    ]
    weight_vector = weight_vector or [1.0] * len(result_lists)
    if len(source_names) != len(result_lists) or len(weight_vector) != len(result_lists):
        raise ValueError("source names, weights, and result lists must have equal length")
    if rank_weight < 0 or score_weight < 0 or math.isclose(rank_weight + score_weight, 0):
        raise ValueError("rank and score weights must be non-negative with a positive sum")
    if any(weight < 0 for weight in weight_vector) or math.isclose(sum(weight_vector), 0):
        raise ValueError("source weights must be non-negative with a positive sum")

    source_total = sum(weight_vector)
    component_total = rank_weight + score_weight
    source_weights = [weight / source_total for weight in weight_vector]
    normalized_rank_weight = rank_weight / component_total
    normalized_score_weight = score_weight / component_total
    entries: dict[str, dict] = {}

    for list_index, results in enumerate(result_lists):
        source_name = source_names[list_index]
        source_weight = source_weights[list_index]
        normalized_scores = _normalize_result_scores(results)
        for rank, (item, normalized_score) in enumerate(
            zip(results, normalized_scores), start=1
        ):
            document_id = str(item.get("id", f"unknown_{list_index}_{rank}"))
            entry = entries.setdefault(document_id, {
                "item": item,
                "hybrid_score": 0.0,
                "rrf_score": 0.0,
                "fusion_sources": [],
                "source_ranks": {},
                "source_scores": {},
            })
            normalized_rank = (rrf_k + 1) / (rrf_k + rank)
            entry["hybrid_score"] += source_weight * (
                normalized_rank_weight * normalized_rank
                + normalized_score_weight * normalized_score
            )
            entry["rrf_score"] += source_weight / (rrf_k + rank)
            entry["fusion_sources"].append(source_name)
            entry["source_ranks"][source_name] = rank
            entry["source_scores"][source_name] = normalized_score

    for entry in entries.values():
        retrieved_at = entry["item"].get("metadata", {}).get("retrieved_at", "")
        decay = _compute_time_decay(retrieved_at) if enable_time_decay else 1.0
        entry["hybrid_score"] *= decay
        entry["rrf_score"] *= decay
        entry["time_decay"] = decay

    ranked = sorted(
        entries.values(),
        key=lambda entry: (entry["hybrid_score"], entry["rrf_score"]),
        reverse=True,
    )
    return [
        {
            **entry["item"],
            "hybrid_score": entry["hybrid_score"],
            "rrf_score": entry["rrf_score"],
            "time_decay": entry["time_decay"],
            "fusion_sources": entry["fusion_sources"],
            "source_ranks": entry["source_ranks"],
            "source_scores": entry["source_scores"],
        }
        for entry in ranked
    ]


def _select_with_source_quotas(
    ranked: list[dict],
    result_lists: list[list[dict]],
    *,
    top_k: int,
    min_results_per_source: int,
) -> list[dict]:
    """Preserve each retriever's strongest evidence, then fill by fused rank."""
    if top_k < 1 or not ranked:
        return []
    available_ids = {str(item.get("id")) for item in ranked}
    required_ids: list[str] = []
    if min_results_per_source > 0 and top_k >= len(result_lists):
        for results in result_lists:
            source_added = 0
            for item in results:
                document_id = str(item.get("id", ""))
                if not document_id or document_id not in available_ids:
                    continue
                if document_id not in required_ids:
                    required_ids.append(document_id)
                source_added += 1
                if source_added >= min_results_per_source:
                    break

    selected_ids = set(required_ids[:top_k])
    for item in ranked:
        if len(selected_ids) >= top_k:
            break
        selected_ids.add(str(item.get("id")))
    return [item for item in ranked if str(item.get("id")) in selected_ids][:top_k]


class HybridRetriever:
    """混合检索器 — BM25 + 向量 + RRF 融合。

    使用方式：
        retriever = HybridRetriever()
        retriever.index_documents(project_name, documents)  # 一次性索引
        results = retriever.search(query, project_name, top_k=10)
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        chunker: Optional[DocumentChunker] = None,
        vector_store: Optional[VectorStore] = None,
        bm25: Optional[BM25Retriever] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rank_weight: float = 0.5,
        score_weight: float = 0.5,
        candidate_pool_factor: int = DEFAULT_CANDIDATE_POOL_FACTOR,
        min_results_per_source: int = DEFAULT_MIN_RESULTS_PER_SOURCE,
    ):
        """初始化混合检索器。

        Args:
            embedder: 嵌入模型
            chunker: 文档分块器
            vector_store: 向量存储
            bm25: BM25 检索器
            vector_weight: 向量检索的 RRF 权重
            bm25_weight: BM25 检索的 RRF 权重
        """
        self.embedder = embedder or Embedder()
        self.chunker = chunker or DocumentChunker()
        self.vector_store = vector_store or VectorStore()
        self.bm25 = bm25 or BM25Retriever()
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rank_weight = rank_weight
        self.score_weight = score_weight
        self.candidate_pool_factor = max(2, candidate_pool_factor)
        self.min_results_per_source = max(0, min_results_per_source)

    def index_documents(
        self,
        project_name: str,
        documents: list[dict],
    ) -> dict:
        """对项目文档进行完整索引（分块 + 嵌入 + BM25 + 向量存储）。

        Args:
            project_name: 项目名称
            documents: 文档列表，每个包含 content、source_url、source_type 等

        Returns:
            索引统计信息
        """
        if not documents:
            self.clear(project_name)
            return {
                "chunks": 0,
                "vector_indexed": 0,
                "vector_removed": 0,
                "vector_total": 0,
                "bm25_indexed": 0,
            }

        # 1. 分块
        chunks = self.chunker.chunk_documents(documents)

        # 2. 仅向量化新增 chunk，删除内容更新后遗留的旧 chunk。
        expected_ids = {chunk["id"] for chunk in chunks}
        existing_ids = self.vector_store.document_ids(project_name)
        stale_ids = existing_ids - expected_ids
        missing_chunks = [chunk for chunk in chunks if chunk["id"] not in existing_ids]
        removed_count = self.vector_store.delete_documents(project_name, stale_ids)
        if missing_chunks:
            embeddings = self.embedder.embed_batch(
                [chunk["text"] for chunk in missing_chunks]
            )
            vector_count = self.vector_store.add_documents(
                project_name, missing_chunks, embeddings
            )
        else:
            vector_count = 0

        # 4. 构建 BM25 索引
        self.bm25.index(project_name, chunks)

        return {
            "chunks": len(chunks),
            "vector_indexed": vector_count,
            "vector_removed": removed_count,
            "vector_total": self.vector_store.count(project_name),
            "bm25_indexed": self.bm25.count(project_name),
        }

    def search(
        self,
        query: str,
        project_name: Optional[str] = None,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """混合检索。

        Args:
            query: 搜索查询
            project_name: 项目名称（可选）
            top_k: 返回数量
            filter_metadata: 元数据过滤条件

        Returns:
            融合后的搜索结果列表
        """
        # 1. 向量检索
        query_embedding = self.embedder.embed(query)
        candidate_count = max(top_k, top_k * self.candidate_pool_factor)
        vector_results = self.vector_store.search(
            query_embedding,
            project_name=project_name,
            top_k=candidate_count,
            filter_metadata=filter_metadata,
        )

        # 2. BM25 检索
        bm25_results = self.bm25.search(
            query,
            project_name=project_name,
            top_k=candidate_count,
        )

        # 3. 分数归一化 + 排名融合，并保留两路检索的最低候选配额
        fused = score_rank_fusion(
            [vector_results, bm25_results],
            source_names=["vector", "bm25"],
            weight_vector=[self.vector_weight, self.bm25_weight],
            rank_weight=self.rank_weight,
            score_weight=self.score_weight,
        )
        return _select_with_source_quotas(
            fused,
            [vector_results, bm25_results],
            top_k=top_k,
            min_results_per_source=self.min_results_per_source,
        )

    def clear(self, project_name: str):
        """清除项目的所有索引。"""
        self.vector_store.delete_project(project_name)
        self.bm25.clear_project(project_name)
