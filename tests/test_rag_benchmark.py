"""Tests for the reproducible Hybrid RAG benchmark."""

import json

from project_advisor.rag.benchmark import (
    HashEmbedder,
    evaluate_gates,
    generate_synthetic_dataset,
    load_dataset,
    run_benchmark,
)
from project_advisor.rag.hybrid_retriever import (
    _select_with_source_quotas,
    score_rank_fusion,
)


def test_synthetic_benchmark_reports_quality_load_and_incremental_index(tmp_path):
    documents, queries = generate_synthetic_dataset(24, 8, 3)
    report = run_benchmark(
        documents=documents,
        queries=queries,
        storage_dir=tmp_path,
        embedder=HashEmbedder(64),
        embedder_name="hash:64",
        dataset_kind="synthetic",
        top_k=3,
        concurrency_levels=[1, 2],
        repetitions=2,
        warmup=2,
        chunk_size=1000,
        chunk_overlap=100,
        min_recall_at_k=1.0,
        min_mrr=1.0,
    )

    assert report["indexing"]["chunk_count"] == 24
    assert report["indexing"]["incremental_new_vectors"] == 0
    assert report["indexing"]["storage_bytes"] > 0
    assert report["quality"]["recall_at_k"] == 1.0
    assert report["quality"]["mrr"] == 1.0
    assert report["quality_breakdown"]["bm25_only"]["recall_at_k"] == 1.0
    assert 0.0 <= report["quality_breakdown"]["vector_only"]["recall_at_k"] <= 1.0
    assert report["quality_breakdown"]["hybrid"]["recall_at_k"] == 1.0
    assert list(report["ablation"]["variants"]) == [
        "bm25_only",
        "vector_only",
        "hybrid",
        "hybrid_rewrite",
        "hybrid_rewrite_rerank",
    ]
    assert all(
        "delta_vs_hybrid" in metrics
        for metrics in report["ablation"]["variants"].values()
    )
    assert [item["concurrency"] for item in report["load"]["scenarios"]] == [1, 2]
    assert all(item["request_count"] == 16 for item in report["load"]["scenarios"])
    assert report["verdict"]["status"] == "pass"


def test_labelled_dataset_validation_and_baseline_regression_gate(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({
        "documents": [{
            "id": "doc-1",
            "project_name": "LangGraph",
            "content": "Durable checkpoint recovery.",
        }],
        "queries": [{
            "id": "q-1",
            "query": "checkpoint recovery",
            "project_name": "LangGraph",
            "relevant_document_ids": ["doc-1"],
        }],
    }), encoding="utf-8")
    documents, queries = load_dataset(dataset_path)

    assert documents[0].document_id == "doc-1"
    assert queries[0].relevant_document_ids == ("doc-1",)

    current = {
        "quality": {"recall_at_k": 0.95, "mrr": 0.9},
        "load": {"scenarios": [{
            "concurrency": 4,
            "throughput_qps": 90.0,
            "error_rate": 0.0,
            "latency_ms": {"p95": 120.0},
        }]},
    }
    baseline = {
        "quality": {"recall_at_k": 1.0, "mrr": 1.0},
        "load": {"scenarios": [{
            "concurrency": 4,
            "throughput_qps": 100.0,
            "error_rate": 0.0,
            "latency_ms": {"p95": 100.0},
        }]},
    }
    verdict = evaluate_gates(
        current,
        gate_concurrency=4,
        min_recall_at_k=None,
        min_mrr=None,
        max_p95_ms=None,
        max_error_rate=0.0,
        baseline=baseline,
        max_p95_regression_pct=10.0,
        max_qps_drop_pct=20.0,
        max_recall_drop=0.02,
    )

    assert verdict["status"] == "fail"
    assert {gate["metric"] for gate in verdict["gates"] if not gate["passed"]} == {
        "baseline.recall_drop",
        "baseline.p95_ms",
    }


def test_balanced_fusion_preserves_bm25_exact_hit_in_top_k():
    vector_results = [
        {"id": f"vector-{index}", "text": "semantic candidate", "score": 1 - index / 20}
        for index in range(10)
    ]
    bm25_results = [
        {"id": "exact-keyword", "text": "exact API keyword", "score": 1.0}
    ]
    fused = score_rank_fusion(
        [vector_results, bm25_results],
        source_names=["vector", "bm25"],
        weight_vector=[0.5, 0.5],
        enable_time_decay=False,
    )
    selected = _select_with_source_quotas(
        fused,
        [vector_results, bm25_results],
        top_k=5,
        min_results_per_source=1,
    )

    exact = next(item for item in selected if item["id"] == "exact-keyword")
    assert len(selected) == 5
    assert exact["fusion_sources"] == ["bm25"]
    assert exact["source_ranks"] == {"bm25": 1}


def test_score_rank_fusion_rewards_documents_found_by_both_sources():
    vector_results = [
        {"id": "shared", "text": "shared", "score": 0.8},
        {"id": "vector-only", "text": "vector", "score": 0.7},
    ]
    bm25_results = [
        {"id": "bm25-only", "text": "keyword", "score": 1.0},
        {"id": "shared", "text": "shared", "score": 0.8},
    ]

    fused = score_rank_fusion(
        [vector_results, bm25_results],
        source_names=["vector", "bm25"],
        enable_time_decay=False,
    )

    assert fused[0]["id"] == "shared"
    assert set(fused[0]["fusion_sources"]) == {"vector", "bm25"}
