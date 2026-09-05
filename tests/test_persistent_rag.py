"""Tests for the persistent Evidence and Hybrid RAG data loop."""

import pytest

from project_advisor.rag.bm25_retriever import BM25Retriever
from project_advisor.rag.chunker import DocumentChunker
from project_advisor.rag.document_store import DocumentStore
from project_advisor.rag.embedder import Embedder
from project_advisor.rag.hybrid_retriever import HybridRetriever
from project_advisor.rag.text_analysis import lexical_tokens
from project_advisor.rag.knowledge_store import persist_evidences
from project_advisor.rag.vector_store import VectorStore
from project_advisor.schemas.evidence import Evidence


class FakeEmbedder:
    """Small deterministic embedder that keeps persistence tests offline."""

    def __init__(self, identity: str = "fake-v1", dimensions: int = 3):
        self.index_identity = identity
        self.index_metadata = {
            "embedding_identity": identity,
            "embedding_model": identity,
        }
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        values = [
            float(len(text)),
            float(lowered.count("checkpoint")),
            float(lowered.count("agent")),
        ]
        return (values + [0.0] * self.dimensions)[:self.dimensions]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def test_multilingual_tokenizer_keeps_chinese_and_technical_compounds():
    tokens = lexical_tokens("LangGraph 支持中文检索与 multi-agent v2.0")

    assert {"langgraph", "lang", "graph", "中文", "文检", "检索"} <= set(tokens)
    assert {"multi-agent", "multi", "agent", "v2.0", "v2", "0"} <= set(tokens)


def test_embedder_separates_query_instruction_and_normalizes_vectors():
    class CapturingModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, *, normalize_embeddings):
            self.calls.append((list(texts), normalize_embeddings))
            return [[3.0, 4.0] for _ in texts]

    model = CapturingModel()
    embedder = Embedder(
        model_name="BAAI/bge-base-zh-v1.5",
        query_instruction="为这个句子生成表示以用于检索相关文章：",
    )
    embedder._model = model

    query_vector = embedder.embed_query("如何恢复工作流？")
    document_vectors = embedder.embed_documents(["Checkpoint 可以恢复工作流。"])

    assert model.calls == [
        (["为这个句子生成表示以用于检索相关文章：如何恢复工作流？"], True),
        (["Checkpoint 可以恢复工作流。"], True),
    ]
    assert query_vector == pytest.approx([0.6, 0.8])
    assert document_vectors[0] == pytest.approx([0.6, 0.8])
    assert embedder.index_metadata["embedding_model"] == "BAAI/bge-base-zh-v1.5"


def make_evidence(content: str = "LangGraph supports durable checkpoint state.") -> Evidence:
    return Evidence(
        source_url="https://docs.example.com/langgraph/checkpoint",
        source_type="official_documentation",
        project_name="Microsoft Agent Framework",
        content=content,
        relevance="state persistence",
        confidence="high",
        retrieved_at="2026-07-23T08:00:00+00:00",
    )


def test_document_store_survives_restart_and_versions_by_content(tmp_path):
    first = make_evidence()
    changed = make_evidence("LangGraph checkpoint storage supports durable execution.")
    result = persist_evidences(
        [first, first.model_dump(), changed], storage_dir=tmp_path
    )

    assert result["stored"] == 2
    restarted = DocumentStore(storage_dir=str(tmp_path))
    stored = restarted.get_by_project("Microsoft Agent Framework")
    assert len(stored) == 2
    assert {item.evidence_id for item in stored} == {
        first.evidence_id,
        changed.evidence_id,
    }


def test_chunk_and_bm25_ids_are_stable_after_restart(tmp_path):
    chunker = DocumentChunker(chunk_size=120, chunk_overlap=20)
    documents = [
        {
            "content": "LangGraph checkpoint persistence and durable execution.",
            "source_url": "https://docs.example.com/langgraph",
            "project_name": "LangGraph",
        },
        {
            "content": "CrewAI role collaboration and task delegation.",
            "source_url": "https://docs.example.com/crewai",
            "project_name": "CrewAI",
        },
        {
            "content": "Phoenix provides traces and evaluation datasets.",
            "source_url": "https://docs.example.com/phoenix",
            "project_name": "Phoenix",
        },
    ]
    chunks = chunker.chunk_documents(documents)
    assert [chunk["id"] for chunk in chunks] == [
        chunk["id"] for chunk in chunker.chunk_documents(documents)
    ]

    bm25_dir = tmp_path / "bm25"
    first = BM25Retriever(storage_dir=bm25_dir)
    first.index("AllProjects", chunks)
    restarted = BM25Retriever(storage_dir=bm25_dir)
    results = restarted.search("checkpoint", project_name="AllProjects")

    assert restarted.count("AllProjects") == len(chunks)
    assert results[0]["id"] == chunks[0]["id"]


def test_bm25_retrieves_and_ranks_chinese_in_a_single_project_index(tmp_path):
    bm25 = BM25Retriever(storage_dir=tmp_path / "bm25-zh")
    chunks = [
        {"id": "permissions", "text": "系统提供租户级权限过滤和访问控制。"},
        {"id": "retrieval", "text": "系统支持中文混合检索、向量召回与关键词排序。"},
        {"id": "observability", "text": "系统提供链路追踪和运行指标。"},
    ]
    bm25.index("中文项目", chunks)

    results = bm25.search("中文检索排序", project_name="中文项目", top_k=3)

    assert results
    assert results[0]["id"] == "retrieval"
    assert results[0]["lexical_overlap"] > 0
    assert all(result["id"] != "observability" for result in results)

    single = BM25Retriever(storage_dir=tmp_path / "bm25-single-zh")
    single.index("单文档", [{"id": "only", "text": "支持中文检索。"}])
    assert single.search("中文检索", project_name="单文档")[0]["id"] == "only"


def test_bm25_cross_project_sorting_uses_one_global_score_scale(tmp_path):
    bm25 = BM25Retriever(storage_dir=tmp_path / "bm25-cross-project")
    bm25.index("弱匹配", [{"id": "weak", "text": "仅介绍中文界面。"}])
    bm25.index("强匹配", [{"id": "strong", "text": "中文检索排序支持关键词召回。"}])

    results = bm25.search("中文检索排序", top_k=2)

    assert [result["id"] for result in results] == ["strong", "weak"]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["score"] > results[1]["score"]


def test_hybrid_indexes_are_idempotent_and_restore_across_restart(tmp_path):
    vector_dir = tmp_path / "vector"
    bm25_dir = tmp_path / "bm25"
    documents = [
        {
            "content": "LangGraph checkpoint persistence enables durable agent execution.",
            "source_url": "https://docs.example.com/langgraph/checkpoint",
            "source_type": "official_documentation",
            "project_name": "LangGraph",
        },
        {
            "content": "LangGraph agents can pause and resume long-running workflows.",
            "source_url": "https://docs.example.com/langgraph/durable",
            "source_type": "official_documentation",
            "project_name": "LangGraph",
        },
    ]

    first = HybridRetriever(
        embedder=FakeEmbedder(),
        vector_store=VectorStore(storage_dir=str(vector_dir)),
        bm25=BM25Retriever(storage_dir=bm25_dir),
    )
    first_stats = first.index_documents("LangGraph", documents)
    assert first_stats["vector_indexed"] == first_stats["chunks"]

    restarted = HybridRetriever(
        embedder=FakeEmbedder(),
        vector_store=VectorStore(storage_dir=str(vector_dir)),
        bm25=BM25Retriever(storage_dir=bm25_dir),
    )
    second_stats = restarted.index_documents("LangGraph", documents)
    results = restarted.search("checkpoint agent", project_name="LangGraph")

    assert restarted.vector_store.list_projects() == ["LangGraph"]
    assert second_stats["vector_indexed"] == 0
    assert second_stats["vector_removed"] == 0
    assert second_stats["vector_total"] == first_stats["chunks"]
    assert results
    assert len({result["id"] for result in results}) == len(results)


def test_embedding_identity_change_rebuilds_persisted_vectors(tmp_path):
    vector_dir = tmp_path / "vector-model-migration"
    bm25_dir = tmp_path / "bm25-model-migration"
    documents = [{
        "content": "中文问题对应 English checkpoint documentation.",
        "source_url": "https://docs.example.com/checkpoint",
        "project_name": "LangGraph",
    }]

    first_embedder = FakeEmbedder(identity="minilm-v1", dimensions=3)
    first = HybridRetriever(
        embedder=first_embedder,
        vector_store=VectorStore(
            storage_dir=str(vector_dir),
            embedding_identity=first_embedder.index_identity,
            embedding_metadata=first_embedder.index_metadata,
        ),
        bm25=BM25Retriever(storage_dir=bm25_dir),
    )
    assert first.index_documents("LangGraph", documents)["vector_indexed"] == 1
    first.vector_store.close()

    second_embedder = FakeEmbedder(identity="bge-m3-v1", dimensions=4)
    second = HybridRetriever(
        embedder=second_embedder,
        vector_store=VectorStore(
            storage_dir=str(vector_dir),
            embedding_identity=second_embedder.index_identity,
            embedding_metadata=second_embedder.index_metadata,
        ),
        bm25=BM25Retriever(storage_dir=bm25_dir),
    )
    stats = second.index_documents("LangGraph", documents)

    assert stats["vector_rebuilt_for_embedding_change"] is True
    assert stats["vector_indexed"] == 1
    assert stats["vector_total"] == 1
    collection = second.vector_store._get_existing_collection("LangGraph")
    assert collection.metadata["embedding_identity"] == "bge-m3-v1"


def test_vector_store_batches_upserts_above_client_limit(tmp_path):
    vector_store = VectorStore(storage_dir=str(tmp_path / "vector-batches"))
    vector_store._client.get_max_batch_size = lambda: 2
    chunks = [
        {
            "id": f"chunk-{index}",
            "text": f"document {index}",
            "metadata": {"chunk_id": f"chunk-{index}"},
        }
        for index in range(5)
    ]
    embeddings = [[float(index), 0.0, 1.0] for index in range(5)]

    try:
        assert vector_store.add_documents("BatchProject", chunks, embeddings) == 5
        assert vector_store.count("BatchProject") == 5
    finally:
        vector_store.close()
