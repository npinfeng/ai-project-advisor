"""混合检索引擎 — BM25 + 向量检索 + Reciprocal Rank Fusion。

核心思路：
1. 同时执行 BM25 关键词检索和向量语义检索
2. 使用 RRF 算法融合两路结果
3. 对融合后的候选文档进行重排序
4. 支持元数据过滤（项目、文档类型、日期）

这是整个 RAG 系统的主要对外接口。
"""

from typing import Optional

from project_advisor.rag.bm25_retriever import BM25Retriever
from project_advisor.rag.chunker import DocumentChunker
from project_advisor.rag.embedder import Embedder
from project_advisor.rag.vector_store import VectorStore


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    weight_vector: Optional[list[float]] = None,
) -> list[dict]:
    """Reciprocal Rank Fusion（RRF）— 融合多路检索结果。

    公式：score(d) = sum(1 / (k + rank_i(d)))
    其中 k 是平滑参数（默认 60），rank_i 是文档在第 i 路结果中的排名。

    Args:
        result_lists: 多路检索结果列表
        k: RRF 平滑参数
        weight_vector: 各路结果的权重，如 [0.6, 0.4] 表示向量检索权重更高

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

    ranked = sorted(
        scores.values(), key=lambda x: x["rrf_score"], reverse=True
    )

    return [
        {**entry["item"], "rrf_score": entry["rrf_score"]}
        for entry in ranked
    ]


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
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
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
            return {"chunks": 0, "indexed": 0}

        # 1. 分块
        chunks = self.chunker.chunk_documents(documents)

        # 2. 批量嵌入
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        # 3. 存入向量存储
        vector_count = self.vector_store.add_documents(
            project_name, chunks, embeddings
        )

        # 4. 构建 BM25 索引
        self.bm25.index(project_name, chunks)

        return {
            "chunks": len(chunks),
            "vector_indexed": vector_count,
            "bm25_indexed": self.bm25.count(),
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
        vector_results = self.vector_store.search(
            query_embedding,
            project_name=project_name,
            top_k=top_k * 2,  # 多取一些供融合
            filter_metadata=filter_metadata,
        )

        # 2. BM25 检索
        bm25_results = self.bm25.search(
            query,
            project_name=project_name,
            top_k=top_k * 2,
        )

        # 3. RRF 融合
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            weight_vector=[self.vector_weight, self.bm25_weight],
        )

        return fused[:top_k]

    def clear(self, project_name: str):
        """清除项目的所有索引。"""
        self.vector_store.delete_project(project_name)
        self.bm25.clear_project(project_name)
