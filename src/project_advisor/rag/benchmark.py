"""Reproducible performance and retrieval-quality benchmark for Hybrid RAG.

The benchmark intentionally exercises the core retrieval path directly:
chunking, embedding, Chroma persistence, BM25 indexing, vector/BM25 search,
and RRF fusion. Its ablation suite uses deterministic rewrite/rerank proxies so
all five retrieval stages can be compared offline; production LLM latency and
quality should still be measured separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Protocol

from project_advisor.rag.bm25_retriever import BM25Retriever
from project_advisor.rag.chunker import DocumentChunker
from project_advisor.rag.embedder import Embedder
from project_advisor.rag.hybrid_retriever import HybridRetriever, score_rank_fusion
from project_advisor.rag.vector_store import VectorStore


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Fast deterministic embedding used for offline load tests and CI.

    This is not a semantic model. It makes storage and retrieval benchmarks
    repeatable without downloading a model or calling an API.
    """

    def __init__(self, dimensions: int = 384):
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[\w.-]+", text.casefold(), flags=re.UNICODE)

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


@dataclass(frozen=True)
class BenchmarkDocument:
    document_id: str
    project_name: str
    content: str
    source_url: str
    source_type: str = "benchmark"

    def as_retriever_document(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "project_name": self.project_name,
            "retrieved_at": "",
            "metadata": {"benchmark_document_id": self.document_id},
        }


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    query: str
    relevant_document_ids: tuple[str, ...]
    project_name: str | None = None
    rewritten_queries: tuple[str, ...] = ()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    output = ""
    while value:
        value, remainder = divmod(value, 36)
        output = alphabet[remainder] + output
    return output


def generate_synthetic_dataset(
    document_count: int,
    query_count: int,
    project_count: int,
) -> tuple[list[BenchmarkDocument], list[BenchmarkQuery]]:
    """Create deterministic documents with one independently known relevant item."""
    if document_count < 1 or query_count < 1 or project_count < 1:
        raise ValueError("documents, queries, and projects must all be positive")
    if query_count > document_count:
        raise ValueError("synthetic query count cannot exceed document count")

    capabilities = [
        "durable checkpoint recovery",
        "multi agent orchestration",
        "permission filtered retrieval",
        "streaming tool execution",
        "observability trace export",
        "incremental vector indexing",
        "human approval workflow",
        "self hosted deployment",
    ]
    documents: list[BenchmarkDocument] = []
    for index in range(document_count):
        project_name = f"benchmark-project-{index % project_count:03d}"
        code = f"capability-{_base36(index).rjust(8, '0')}"
        capability = capabilities[index % len(capabilities)]
        content = (
            f"Technical reference for {project_name}. The unique feature code is {code}. "
            f"This component implements {capability}. Configuration includes bounded "
            f"workers, persistent state, health diagnostics, and deterministic recovery. "
            f"Use feature code {code} when locating this exact implementation record."
        )
        documents.append(BenchmarkDocument(
            document_id=f"doc-{index:08d}",
            project_name=project_name,
            content=content,
            source_url=f"https://benchmark.invalid/{project_name}/{index}",
        ))

    queries: list[BenchmarkQuery] = []
    for index in range(query_count):
        document_index = (index * document_count) // query_count
        document = documents[document_index]
        code = f"capability-{_base36(document_index).rjust(8, '0')}"
        queries.append(BenchmarkQuery(
            query_id=f"query-{index:08d}",
            query=f"How is the exact feature {code} configured and recovered?",
            relevant_document_ids=(document.document_id,),
            project_name=document.project_name,
            rewritten_queries=(f"{code} configuration deterministic recovery",),
        ))
    return documents, queries


def load_dataset(path: Path) -> tuple[list[BenchmarkDocument], list[BenchmarkQuery]]:
    """Load an independently labelled benchmark dataset from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = [
        BenchmarkDocument(
            document_id=str(item["id"]),
            project_name=str(item.get("project_name") or "default"),
            content=str(item["content"]),
            source_url=str(item.get("source_url") or f"benchmark://{item['id']}"),
            source_type=str(item.get("source_type") or "benchmark"),
        )
        for item in payload.get("documents", [])
    ]
    queries = [
        BenchmarkQuery(
            query_id=str(item["id"]),
            query=str(item["query"]),
            relevant_document_ids=tuple(
                str(value) for value in item.get("relevant_document_ids", [])
            ),
            project_name=(
                str(item["project_name"]) if item.get("project_name") else None
            ),
            rewritten_queries=tuple(
                str(value) for value in item.get("rewritten_queries", [])
            ),
        )
        for item in payload.get("queries", [])
    ]
    if not documents or not queries:
        raise ValueError("benchmark dataset requires non-empty documents and queries")
    document_ids = {item.document_id for item in documents}
    unknown = {
        relevant
        for query in queries
        for relevant in query.relevant_document_ids
        if relevant not in document_ids
    }
    if unknown:
        raise ValueError(f"queries reference unknown document IDs: {sorted(unknown)[:5]}")
    if any(not query.relevant_document_ids for query in queries):
        raise ValueError("every benchmark query requires at least one relevant document ID")
    return documents, queries


def _result_document_id(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    return str(metadata.get("benchmark_document_id") or result.get("id") or "")


def evaluate_retrieval(
    query_results: list[tuple[BenchmarkQuery, list[dict[str, Any]], float, str | None]],
    top_k: int,
) -> dict[str, Any]:
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    latencies: list[float] = []
    errors: list[str] = []

    for query, results, latency_ms, error in query_results:
        latencies.append(latency_ms)
        if error:
            errors.append(error)
            recalls.append(0.0)
            precisions.append(0.0)
            reciprocal_ranks.append(0.0)
            ndcgs.append(0.0)
            continue
        relevant = set(query.relevant_document_ids)
        retrieved = [_result_document_id(item) for item in results[:top_k]]
        hits = [document_id in relevant for document_id in retrieved]
        hit_count = len(set(retrieved) & relevant)
        recalls.append(hit_count / len(relevant))
        precisions.append(hit_count / top_k)
        first_hit = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
        reciprocal_ranks.append(1.0 / first_hit if first_hit else 0.0)
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, hit in enumerate(hits, 1)
            if hit
        )
        ideal_hits = min(len(relevant), top_k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)

    return {
        "query_count": len(query_results),
        "top_k": top_k,
        "recall_at_k": mean(recalls),
        "precision_at_k": mean(precisions),
        "mrr": mean(reciprocal_ranks),
        "ndcg_at_k": mean(ndcgs),
        "hit_rate_at_k": mean(value > 0 for value in recalls),
        "error_count": len(errors),
        "error_rate": len(errors) / len(query_results) if query_results else 0.0,
        "latency_ms": {
            "mean": mean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies, default=0.0),
        },
        "errors": errors[:10],
    }


def _search_once(
    retriever: HybridRetriever,
    query: BenchmarkQuery,
    top_k: int,
) -> tuple[BenchmarkQuery, list[dict[str, Any]], float, str | None]:
    started = time.perf_counter_ns()
    try:
        results = retriever.search(
            query.query,
            project_name=query.project_name,
            top_k=top_k,
        )
        error = None
    except Exception as exc:  # Benchmark reports failures instead of aborting a run.
        results = []
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000
    return query, results, latency_ms, error


def _load_scenario(
    retriever: HybridRetriever,
    queries: list[BenchmarkQuery],
    top_k: int,
    concurrency: int,
    repetitions: int,
) -> dict[str, Any]:
    # Add a unique neutral suffix so Embedder's exact-text cache does not turn
    # repeated benchmark requests into unrealistically cheap cache hits.
    workload = [
        BenchmarkQuery(
            query_id=f"{query.query_id}-load-{repetition}-{index}",
            query=f"{query.query} benchmark request {repetition}-{index}",
            relevant_document_ids=query.relevant_document_ids,
            project_name=query.project_name,
        )
        for repetition in range(repetitions)
        for index, query in enumerate(queries)
    ]
    started = time.perf_counter_ns()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_search_once, retriever, query, top_k)
            for query in workload
        ]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000
    metrics = evaluate_retrieval(results, top_k)
    return {
        "concurrency": concurrency,
        "request_count": len(workload),
        "wall_time_seconds": elapsed_seconds,
        "throughput_qps": len(workload) / elapsed_seconds if elapsed_seconds else 0.0,
        "error_count": metrics["error_count"],
        "error_rate": metrics["error_rate"],
        "latency_ms": metrics["latency_ms"],
    }


def _group_documents(
    documents: Iterable[BenchmarkDocument],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        grouped.setdefault(document.project_name, []).append(
            document.as_retriever_document()
        )
    return grouped


def _quality_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: metrics[key]
        for key in (
            "recall_at_k",
            "precision_at_k",
            "mrr",
            "ndcg_at_k",
            "hit_rate_at_k",
            "error_count",
            "error_rate",
        )
    }
    summary["latency_ms"] = metrics["latency_ms"]
    return summary


def _rewrite_variants(query: BenchmarkQuery) -> list[str]:
    """Return labelled rewrites or a deterministic, domain-aware offline proxy."""
    variants = [query.query, *query.rewritten_queries]
    if not query.rewritten_queries:
        expanded = query.query.casefold()
        replacements = {
            "checkpoint": "checkpoint persistent state durable recovery",
            "rag": "retrieval augmented generation hybrid search",
            "mcp": "model context protocol tools",
            "human in the loop": "human approval interrupt resume",
            "部署": "部署 运维 成本 self hosted cloud",
            "恢复": "恢复 checkpoint durable persistence",
        }
        for needle, replacement in replacements.items():
            if needle in expanded:
                expanded = expanded.replace(needle, replacement)
        variants.append(expanded)
    return list(dict.fromkeys(value.strip() for value in variants if value.strip()))


def _merge_rewrite_results(result_lists: list[list[dict]], top_k: int) -> list[dict]:
    if not result_lists:
        return []
    return score_rank_fusion(
        result_lists,
        source_names=[f"rewrite_{index}" for index in range(len(result_lists))],
        weight_vector=[1.0] * len(result_lists),
        enable_time_decay=False,
    )[:top_k]


def _offline_rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    """Deterministic lexical reranker used only to isolate the rerank stage in CI."""
    query_tokens = set(HashEmbedder._tokens(query))

    def key(item: dict) -> tuple[float, float]:
        text_tokens = set(HashEmbedder._tokens(str(item.get("text", ""))))
        lexical = len(query_tokens & text_tokens) / max(1, len(query_tokens))
        base = float(item.get("hybrid_score", item.get("score", 0.0)) or 0.0)
        return 0.75 * lexical + 0.25 * base, base

    return sorted(results, key=key, reverse=True)[:top_k]


def run_ablation_suite(
    retriever: HybridRetriever,
    queries: list[BenchmarkQuery],
    top_k: int,
) -> dict[str, Any]:
    """Compare BM25, vector, hybrid, rewrite, and rerank stages reproducibly."""
    component = _component_quality(retriever, queries, top_k)
    hybrid_runs = [_search_once(retriever, query, top_k) for query in queries]
    rewrite_runs = []
    rerank_runs = []
    candidate_k = max(top_k * retriever.candidate_pool_factor, top_k)
    for query in queries:
        started = time.perf_counter_ns()
        error = None
        try:
            result_lists = [
                retriever.search(
                    variant,
                    project_name=query.project_name,
                    top_k=candidate_k,
                )
                for variant in _rewrite_variants(query)
            ]
            rewritten = _merge_rewrite_results(result_lists, candidate_k)
        except Exception as exc:
            rewritten = []
            error = f"{type(exc).__name__}: {exc}"
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        rewrite_runs.append((query, rewritten[:top_k], elapsed, error))

        rerank_started = time.perf_counter_ns()
        reranked = _offline_rerank(query.query, rewritten, top_k) if not error else []
        rerank_runs.append((
            query,
            reranked,
            elapsed + (time.perf_counter_ns() - rerank_started) / 1_000_000,
            error,
        ))

    variants = {
        "bm25_only": component["bm25_only"],
        "vector_only": component["vector_only"],
        "hybrid": _quality_summary(evaluate_retrieval(hybrid_runs, top_k)),
        "hybrid_rewrite": _quality_summary(evaluate_retrieval(rewrite_runs, top_k)),
        "hybrid_rewrite_rerank": _quality_summary(
            evaluate_retrieval(rerank_runs, top_k)
        ),
    }
    baseline = variants["hybrid"]
    for metrics in variants.values():
        metrics["delta_vs_hybrid"] = {
            "recall_at_k": metrics["recall_at_k"] - baseline["recall_at_k"],
            "mrr": metrics["mrr"] - baseline["mrr"],
            "ndcg_at_k": metrics["ndcg_at_k"] - baseline["ndcg_at_k"],
        }
    return {
        "method": "offline deterministic proxies; labelled rewrites preferred",
        "variants": variants,
    }


def _component_quality(
    retriever: HybridRetriever,
    queries: list[BenchmarkQuery],
    top_k: int,
) -> dict[str, Any]:
    vector_runs = []
    bm25_runs = []
    for query in queries:
        started = time.perf_counter_ns()
        try:
            query_embedding = retriever.embedder.embed(query.query)
            vector_results = retriever.vector_store.search(
                query_embedding,
                project_name=query.project_name,
                top_k=top_k,
            )
            vector_error = None
        except Exception as exc:
            vector_results = []
            vector_error = f"{type(exc).__name__}: {exc}"
        vector_runs.append((
            query,
            vector_results,
            (time.perf_counter_ns() - started) / 1_000_000,
            vector_error,
        ))

        started = time.perf_counter_ns()
        try:
            bm25_results = retriever.bm25.search(
                query.query,
                project_name=query.project_name,
                top_k=top_k,
            )
            bm25_error = None
        except Exception as exc:
            bm25_results = []
            bm25_error = f"{type(exc).__name__}: {exc}"
        bm25_runs.append((
            query,
            bm25_results,
            (time.perf_counter_ns() - started) / 1_000_000,
            bm25_error,
        ))

    return {
        "vector_only": _quality_summary(evaluate_retrieval(vector_runs, top_k)),
        "bm25_only": _quality_summary(evaluate_retrieval(bm25_runs, top_k)),
    }


def _build_embedder(provider: str, model_name: str, dimensions: int) -> EmbeddingProvider:
    if provider == "hash":
        return HashEmbedder(dimensions)
    return Embedder(model_name=model_name, provider=provider)


def _baseline_scenario(report: dict[str, Any], concurrency: int) -> dict[str, Any] | None:
    for scenario in report.get("load", {}).get("scenarios", []):
        if int(scenario.get("concurrency", -1)) == concurrency:
            return scenario
    return None


def evaluate_gates(
    report: dict[str, Any],
    *,
    gate_concurrency: int,
    min_recall_at_k: float | None,
    min_mrr: float | None,
    max_p95_ms: float | None,
    max_error_rate: float,
    baseline: dict[str, Any] | None,
    max_p95_regression_pct: float,
    max_qps_drop_pct: float,
    max_recall_drop: float,
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    quality = report["quality"]
    scenario = _baseline_scenario(report, gate_concurrency)
    if scenario is None:
        raise ValueError(f"gate concurrency {gate_concurrency} was not benchmarked")

    def add(metric: str, actual: float, expectation: str, passed: bool) -> None:
        gates.append({
            "metric": metric,
            "actual": actual,
            "expectation": expectation,
            "passed": passed,
        })

    if min_recall_at_k is not None:
        add("quality.recall_at_k", quality["recall_at_k"], f">= {min_recall_at_k}", quality["recall_at_k"] >= min_recall_at_k)
    if min_mrr is not None:
        add("quality.mrr", quality["mrr"], f">= {min_mrr}", quality["mrr"] >= min_mrr)
    if max_p95_ms is not None:
        actual = scenario["latency_ms"]["p95"]
        add(f"load.c{gate_concurrency}.p95_ms", actual, f"<= {max_p95_ms}", actual <= max_p95_ms)
    add(
        f"load.c{gate_concurrency}.error_rate",
        scenario["error_rate"],
        f"<= {max_error_rate}",
        scenario["error_rate"] <= max_error_rate,
    )

    if baseline is not None:
        baseline_quality = baseline.get("quality", {})
        baseline_recall = float(baseline_quality.get("recall_at_k", 0.0))
        add(
            "baseline.recall_drop",
            baseline_recall - quality["recall_at_k"],
            f"<= {max_recall_drop}",
            quality["recall_at_k"] >= baseline_recall - max_recall_drop,
        )
        baseline_load = _baseline_scenario(baseline, gate_concurrency)
        if baseline_load is None:
            raise ValueError(
                f"baseline has no concurrency={gate_concurrency} load scenario"
            )
        baseline_p95 = float(baseline_load["latency_ms"]["p95"])
        allowed_p95 = baseline_p95 * (1 + max_p95_regression_pct / 100)
        add(
            "baseline.p95_ms",
            scenario["latency_ms"]["p95"],
            f"<= {allowed_p95:.6f} ({max_p95_regression_pct}% regression)",
            scenario["latency_ms"]["p95"] <= allowed_p95,
        )
        baseline_qps = float(baseline_load["throughput_qps"])
        minimum_qps = baseline_qps * (1 - max_qps_drop_pct / 100)
        add(
            "baseline.throughput_qps",
            scenario["throughput_qps"],
            f">= {minimum_qps:.6f} ({max_qps_drop_pct}% drop)",
            scenario["throughput_qps"] >= minimum_qps,
        )

    return {
        "status": "pass" if all(gate["passed"] for gate in gates) else "fail",
        "gate_concurrency": gate_concurrency,
        "gates": gates,
    }


def run_benchmark(
    *,
    documents: list[BenchmarkDocument],
    queries: list[BenchmarkQuery],
    storage_dir: Path,
    embedder: EmbeddingProvider,
    embedder_name: str,
    dataset_kind: str,
    top_k: int,
    concurrency_levels: list[int],
    repetitions: int,
    warmup: int,
    chunk_size: int,
    chunk_overlap: int,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    rank_weight: float = 0.5,
    score_weight: float = 0.5,
    candidate_pool_factor: int = 4,
    min_results_per_source: int = 1,
    baseline: dict[str, Any] | None = None,
    gate_concurrency: int | None = None,
    min_recall_at_k: float | None = None,
    min_mrr: float | None = None,
    max_p95_ms: float | None = None,
    max_error_rate: float = 0.0,
    max_p95_regression_pct: float = 20.0,
    max_qps_drop_pct: float = 20.0,
    max_recall_drop: float = 0.02,
) -> dict[str, Any]:
    if not concurrency_levels or any(value < 1 for value in concurrency_levels):
        raise ValueError("concurrency levels must be positive")
    if top_k < 1 or repetitions < 1 or warmup < 0:
        raise ValueError("top-k/repetitions must be positive and warmup non-negative")

    vector_dir = storage_dir / "vector"
    bm25_dir = storage_dir / "bm25"
    retriever = HybridRetriever(
        embedder=embedder,
        chunker=DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        vector_store=VectorStore(storage_dir=str(vector_dir)),
        bm25=BM25Retriever(storage_dir=bm25_dir),
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        rank_weight=rank_weight,
        score_weight=score_weight,
        candidate_pool_factor=candidate_pool_factor,
        min_results_per_source=min_results_per_source,
    )

    grouped = _group_documents(documents)
    try:
        index_started = time.perf_counter_ns()
        index_stats = [
            {"project_name": project, **retriever.index_documents(project, items)}
            for project, items in grouped.items()
        ]
        index_seconds = (time.perf_counter_ns() - index_started) / 1_000_000_000
        chunk_count = sum(item["chunks"] for item in index_stats)

        incremental_started = time.perf_counter_ns()
        incremental_stats = [
            {"project_name": project, **retriever.index_documents(project, items)}
            for project, items in grouped.items()
        ]
        incremental_seconds = (
            time.perf_counter_ns() - incremental_started
        ) / 1_000_000_000

        # First pass measures unique-query latency and produces retrieval quality.
        cold_results = [_search_once(retriever, query, top_k) for query in queries]
        quality = evaluate_retrieval(cold_results, top_k)
        ablation = run_ablation_suite(retriever, queries, top_k)
        quality_breakdown = {
            name: ablation["variants"][name]
            for name in ("vector_only", "bm25_only", "hybrid")
        }

        for query in queries[:warmup]:
            _search_once(retriever, query, top_k)

        scenarios = [
            _load_scenario(
                retriever,
                queries,
                top_k,
                concurrency=concurrency,
                repetitions=repetitions,
            )
            for concurrency in concurrency_levels
        ]

        warnings = [
            "Rewrite/rerank ablations are deterministic offline proxies; they exclude production LLM latency/cost.",
            "Absolute latency and QPS are hardware-specific; compare runs on the same host.",
        ]
        if dataset_kind == "synthetic":
            warnings.append(
                "Synthetic labels validate the retrieval pipeline, not real-domain relevance."
            )
        if embedder_name.startswith("hash"):
            warnings.append(
                "Hash embeddings exclude production-model inference cost and semantic quality."
            )

        report: dict[str, Any] = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
            },
            "configuration": {
                "dataset_kind": dataset_kind,
                "document_count": len(documents),
                "query_count": len(queries),
                "project_count": len(grouped),
                "embedder": embedder_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "concurrency_levels": concurrency_levels,
                "repetitions": repetitions,
                "warmup_queries": min(warmup, len(queries)),
                "vector_weight": retriever.vector_weight,
                "bm25_weight": retriever.bm25_weight,
                "rank_weight": retriever.rank_weight,
                "score_weight": retriever.score_weight,
                "candidate_pool_factor": retriever.candidate_pool_factor,
                "min_results_per_source": retriever.min_results_per_source,
            },
            "indexing": {
                "chunk_count": chunk_count,
                "cold_seconds": index_seconds,
                "cold_chunks_per_second": (
                    chunk_count / index_seconds if index_seconds else 0.0
                ),
                "incremental_seconds": incremental_seconds,
                "incremental_new_vectors": sum(
                    item["vector_indexed"] for item in incremental_stats
                ),
                "storage_bytes": _directory_size(storage_dir),
                "storage_bytes_per_chunk": (
                    _directory_size(storage_dir) / chunk_count if chunk_count else 0.0
                ),
                "projects": index_stats,
            },
            "quality": quality,
            "quality_breakdown": quality_breakdown,
            "ablation": ablation,
            "load": {
                "measurement": "warm model/index with unique query text (embedding included)",
                "scenarios": scenarios,
            },
            "warnings": warnings,
        }
        report["verdict"] = evaluate_gates(
            report,
            gate_concurrency=gate_concurrency or max(concurrency_levels),
            min_recall_at_k=min_recall_at_k,
            min_mrr=min_mrr,
            max_p95_ms=max_p95_ms,
            max_error_rate=max_error_rate,
            baseline=baseline,
            max_p95_regression_pct=max_p95_regression_pct,
            max_qps_drop_pct=max_qps_drop_pct,
            max_recall_drop=max_recall_drop,
        )
        return _round_floats(report)
    finally:
        retriever.vector_store.close()


def _parse_concurrency(value: str) -> list[int]:
    levels = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not levels or any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError("concurrency must contain positive integers")
    return levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Hybrid RAG indexing, retrieval quality, latency, and throughput."
    )
    parser.add_argument("--dataset", type=Path, help="Labelled JSON dataset; otherwise generate synthetic data")
    parser.add_argument("--documents", type=int, default=1000, help="Synthetic document count")
    parser.add_argument("--queries", type=int, default=100, help="Synthetic query count")
    parser.add_argument("--projects", type=int, default=10, help="Synthetic project count")
    parser.add_argument("--provider", choices=["hash", "local", "openai"], default="hash")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--hash-dimensions", type=int, default=384)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.5)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--score-weight", type=float, default=0.5)
    parser.add_argument("--candidate-pool-factor", type=int, default=4)
    parser.add_argument("--min-results-per-source", type=int, default=1)
    parser.add_argument("--concurrency", type=_parse_concurrency, default=[1, 4, 8])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--storage-dir", type=Path, help="Parent for a retained, unique benchmark run directory")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    parser.add_argument("--baseline", type=Path, help="Previous report used for regression gates")
    parser.add_argument("--gate-concurrency", type=int, help="Concurrency scenario used by gates; defaults to highest")
    parser.add_argument("--min-recall-at-k", type=float)
    parser.add_argument("--min-mrr", type=float)
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--max-error-rate", type=float, default=0.0)
    parser.add_argument("--max-p95-regression-pct", type=float, default=20.0)
    parser.add_argument("--max-qps-drop-pct", type=float, default=20.0)
    parser.add_argument("--max-recall-drop", type=float, default=0.02)
    parser.add_argument("--fail-on-gate", action="store_true", help="Exit with status 2 when any gate fails")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dataset:
        documents, queries = load_dataset(args.dataset)
        dataset_kind = "labelled"
    else:
        documents, queries = generate_synthetic_dataset(
            args.documents, args.queries, args.projects
        )
        dataset_kind = "synthetic"

    embedder = _build_embedder(args.provider, args.model_name, args.hash_dimensions)
    embedder_name = (
        f"hash:{args.hash_dimensions}"
        if args.provider == "hash"
        else f"{args.provider}:{args.model_name}"
    )
    baseline = (
        json.loads(args.baseline.read_text(encoding="utf-8"))
        if args.baseline
        else None
    )

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.storage_dir:
        run_name = f"rag-benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        run_dir = args.storage_dir.resolve() / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="project-advisor-rag-benchmark-")
        run_dir = Path(temporary.name)

    try:
        report = run_benchmark(
            documents=documents,
            queries=queries,
            storage_dir=run_dir,
            embedder=embedder,
            embedder_name=embedder_name,
            dataset_kind=dataset_kind,
            top_k=args.top_k,
            concurrency_levels=args.concurrency,
            repetitions=args.repetitions,
            warmup=args.warmup,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            vector_weight=args.vector_weight,
            bm25_weight=args.bm25_weight,
            rank_weight=args.rank_weight,
            score_weight=args.score_weight,
            candidate_pool_factor=args.candidate_pool_factor,
            min_results_per_source=args.min_results_per_source,
            baseline=baseline,
            gate_concurrency=args.gate_concurrency,
            min_recall_at_k=args.min_recall_at_k,
            min_mrr=args.min_mrr,
            max_p95_ms=args.max_p95_ms,
            max_error_rate=args.max_error_rate,
            max_p95_regression_pct=args.max_p95_regression_pct,
            max_qps_drop_pct=args.max_qps_drop_pct,
            max_recall_drop=args.max_recall_drop,
        )
        output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        print(output, end="")
        if args.fail_on_gate and report["verdict"]["status"] == "fail":
            raise SystemExit(2)
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
