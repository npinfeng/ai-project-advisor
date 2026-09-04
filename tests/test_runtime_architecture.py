"""Regression tests for API extraction, async RAG and correlated logging."""

import asyncio
import importlib
import json
import logging

from langchain_core.messages import AIMessage

from project_advisor.api.schemas import AdviceRequest
from project_advisor.configuration import Configuration
from project_advisor.errors import (
    AgentRunTimeoutError,
    ModelConfigurationError,
    PersistenceError,
    ProjectAdvisorError,
    StructuredOutputError,
)
from project_advisor.observability.logging import (
    bind_log_context,
    current_log_context,
    log_event,
)
from project_advisor.persistence import TaskStore
from project_advisor.rag.reranker import RelevanceScore, Reranker


def test_app_reexports_extracted_request_schema():
    app_module = importlib.import_module("project_advisor.app")
    assert app_module.AdviceRequest is AdviceRequest


def test_default_embedding_targets_multilingual_bge_m3():
    config = Configuration()

    assert config.embedding_provider == "local"
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.embedding_normalize is True
    assert config.embedding_query_instruction == ""


def test_domain_errors_keep_legacy_exception_compatibility():
    assert issubclass(ModelConfigurationError, ValueError)
    assert issubclass(StructuredOutputError, RuntimeError)
    assert issubclass(AgentRunTimeoutError, RuntimeError)
    assert issubclass(AgentRunTimeoutError, ProjectAdvisorError)


def test_task_store_normalizes_sqlite_failures(tmp_path):
    store = TaskStore(tmp_path / "missing-parent" / "tasks.sqlite3")
    try:
        asyncio.run(store.get("task-1"))
    except PersistenceError as error:
        assert "任务数据库操作失败" in str(error)
    else:
        raise AssertionError("SQLite failures must use the domain exception layer")


def test_log_context_is_isolated_between_concurrent_runs(caplog):
    logger = logging.getLogger("test.runtime.context")

    async def worker(task_id: str) -> dict[str, str]:
        with bind_log_context(task_id=task_id, candidate=f"candidate-{task_id}"):
            await asyncio.sleep(0)
            log_event(logger, logging.INFO, "test_event")
            return current_log_context()

    async def run_workers():
        return await asyncio.gather(worker("run-a"), worker("run-b"))

    with caplog.at_level(logging.INFO):
        contexts = asyncio.run(run_workers())

    assert {value["task_id"] for value in contexts} == {"run-a", "run-b"}
    assert current_log_context() == {}
    payloads = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "test.runtime.context"
    ]
    assert {value["task_id"] for value in payloads} == {"run-a", "run-b"}


def test_reranker_limits_concurrent_model_calls():
    class FakeStructuredModel:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def ainvoke(self, messages):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return {
                "parsed": RelevanceScore(score=8, reason="relevant"),
                "raw": AIMessage(
                    content="",
                    usage_metadata={
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                ),
                "parsing_error": None,
            }

    model = FakeStructuredModel()
    reranker = Reranker(max_concurrency=2, max_attempts=1)
    reranker._model = model
    documents = [
        {
            "id": f"doc-{index}",
            "text": "stateful agent documentation",
            "metadata": {"source_url": f"https://example.com/{index}"},
        }
        for index in range(6)
    ]

    results = asyncio.run(reranker.rerank("stateful agent", documents, top_k=6))

    assert len(results) == 6
    assert model.max_active == 2
    assert all(result["rerank_score"] == 8 for result in results)


def test_rag_search_is_async_and_does_not_use_asyncio_run(monkeypatch):
    rag_module = importlib.import_module("project_advisor.tools.rag_search")

    class FakePipeline:
        def search(self, *, query, project_name, top_k):
            return [{
                "id": "chunk-1",
                "project": project_name,
                "text": f"evidence for {query}",
                "score": 0.9,
                "metadata": {
                    "project_name": project_name,
                    "source_url": "https://docs.example.com/langgraph",
                    "retrieved_at": "2026-09-04T00:00:00+00:00",
                },
            }]

    class FakeRewriter:
        async def generate_multi_queries(self, query):
            return [query, f"{query} checkpoint"]

        async def rewrite(self, query):
            return query

    class FakeReranker:
        async def rerank(self, query, documents, top_k):
            return documents[:top_k]

    monkeypatch.setattr(
        rag_module,
        "_sync_from_store",
        lambda project_name, **kwargs: [{"project_name": project_name}],
    )
    monkeypatch.setattr(rag_module, "_get_pipeline", lambda config=None: FakePipeline())
    monkeypatch.setattr(rag_module, "_get_rewriter", lambda config: FakeRewriter())
    monkeypatch.setattr(rag_module, "_get_reranker", lambda config: FakeReranker())

    result = asyncio.run(rag_module.rag_search.ainvoke(
        {
            "query": "durable execution",
            "project_name": "LangGraph",
            "top_k": 2,
        },
        config={"configurable": {"search_api": "none"}},
    ))

    assert "RAG 搜索结果" in result
    assert "https://docs.example.com/langgraph" in result
