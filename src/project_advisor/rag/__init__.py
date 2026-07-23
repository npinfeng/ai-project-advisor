"""Hybrid RAG components exposed through lazy package attributes."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DocumentChunker": ("project_advisor.rag.chunker", "DocumentChunker"),
    "Embedder": ("project_advisor.rag.embedder", "Embedder"),
    "VectorStore": ("project_advisor.rag.vector_store", "VectorStore"),
    "BM25Retriever": ("project_advisor.rag.bm25_retriever", "BM25Retriever"),
    "HybridRetriever": ("project_advisor.rag.hybrid_retriever", "HybridRetriever"),
    "reciprocal_rank_fusion": (
        "project_advisor.rag.hybrid_retriever",
        "reciprocal_rank_fusion",
    ),
    "Reranker": ("project_advisor.rag.reranker", "Reranker"),
    "QueryRewriter": ("project_advisor.rag.query_rewriter", "QueryRewriter"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load optional RAG dependencies only when their component is requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
