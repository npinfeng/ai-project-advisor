"""Tests for the persistent Evidence and Hybrid RAG data loop."""

from project_advisor.rag.bm25_retriever import BM25Retriever
from project_advisor.rag.chunker import DocumentChunker
from project_advisor.rag.document_store import DocumentStore
from project_advisor.rag.hybrid_retriever import HybridRetriever
from project_advisor.rag.knowledge_store import persist_evidences
from project_advisor.rag.vector_store import VectorStore
from project_advisor.schemas.evidence import Evidence


class FakeEmbedder:
    """Small deterministic embedder that keeps persistence tests offline."""

    @staticmethod
    def embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(len(text)),
            float(lowered.count("checkpoint")),
            float(lowered.count("agent")),
        ]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


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
