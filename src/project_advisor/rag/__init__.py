"""RAG 模块 — 完整的混合检索增强生成系统。

组件：
- DocumentChunker: 智能文档分块
- Embedder: 文本向量化（本地/OpenAI）
- VectorStore: ChromaDB 向量存储
- BM25Retriever: 关键词检索
- HybridRetriever: BM25 + 向量 + RRF 混合检索
- Reranker: LLM 精排
- QueryRewriter: 查询改写和多查询生成
"""

from project_advisor.rag.chunker import DocumentChunker
from project_advisor.rag.embedder import Embedder
from project_advisor.rag.vector_store import VectorStore
from project_advisor.rag.bm25_retriever import BM25Retriever
from project_advisor.rag.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from project_advisor.rag.reranker import Reranker
from project_advisor.rag.query_rewriter import QueryRewriter

__all__ = [
    "DocumentChunker",
    "Embedder",
    "VectorStore",
    "BM25Retriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "Reranker",
    "QueryRewriter",
]
