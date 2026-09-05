"""Tests for the reproducible Hybrid RAG benchmark."""

import json

import pytest

from project_advisor.rag.benchmark import (
    BenchmarkQuery,
    HashEmbedder,
    evaluate_baseline_eligibility,
    evaluate_gates,
    evaluate_retrieval,
    generate_synthetic_dataset,
    load_dataset,
    load_dataset_with_metadata,
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


def test_real_dataset_loads_graded_qrels_provenance_and_slice_metrics(tmp_path):
    path = tmp_path / "real-rag.json"
    path.write_text(json.dumps({
        "metadata": {
            "name": "real-zh-v1", "kind": "real", "annotation_status": "reviewed",
            "annotation_method": "independent_human", "annotator": "reviewer-a",
            "reviewed_at": "2026-09-05T00:00:00+00:00",
            "snapshot_at": "2026-09-04T00:00:00+00:00",
            "query_source": "deidentified-production-sample",
            "corpus_version": "docs-snapshot-2026-09-04",
            "privacy_status": "reviewed", "guideline_version": "qrels-v1",
            "dataset_author": "author-a",
        },
        "documents": [
            {"id": "d3", "project_name": "A", "content": "权限过滤配置",
             "source_url": "https://docs.example.com/permissions"},
            {"id": "d1", "project_name": "A", "content": "权限概览",
             "source_url": "https://docs.example.com/overview"},
        ],
        "queries": [{
            "id": "q1", "query": "如何配置权限过滤", "language": "zh-CN",
            "category": "permissions", "project_name": "A",
            "qrels": [{"document_id": "d3", "relevance": 3},
                      {"document_id": "d1", "relevance": 1}],
        }],
    }, ensure_ascii=False), encoding="utf-8")

    documents, queries, metadata = load_dataset_with_metadata(path)
    assert metadata.kind == "real"
    assert metadata.fingerprint.startswith("sha256:")
    assert queries[0].relevance_by_document == {"d3": 3, "d1": 1}
    result = evaluate_retrieval([(
        queries[0], [
            {"id": "chunk-a", "metadata": {"benchmark_document_id": "d1"}},
            {"id": "chunk-b", "metadata": {"benchmark_document_id": "d3"}},
        ], 2.0, None,
    )], top_k=2)
    assert result["recall_at_k"] == 1.0
    assert 0 < result["ndcg_at_k"] < 1.0
    assert result["map_at_k"] == 1.0
    assert result["slices"]["language"]["zh-CN"]["query_count"] == 1
    assert result["confidence_intervals_95"]["ndcg_at_k"]["lower"] == result["ndcg_at_k"]


def test_document_level_metrics_do_not_count_duplicate_chunks_twice():
    query = BenchmarkQuery(query_id="q", query="权限", relevant_document_ids=("doc-a",),
                           relevance_grades=(("doc-a", 3),))
    results = [
        {"id": "chunk-1", "metadata": {"benchmark_document_id": "doc-a"}},
        {"id": "chunk-2", "metadata": {"benchmark_document_id": "doc-a"}},
        {"id": "chunk-3", "metadata": {"benchmark_document_id": "doc-b"}},
    ]
    metrics = evaluate_retrieval([(query, results, 1.0, None)], top_k=2)
    assert metrics["precision_at_k"] == 0.5
    assert metrics["ndcg_at_k"] == 1.0
    assert metrics["queries"][0]["retrieved_document_ids"] == ["doc-a", "doc-b"]


def test_real_baseline_eligibility_rejects_hash_and_accepts_reviewed_production_run():
    report = {
        "dataset": {
            "kind": "real", "annotation_status": "reviewed",
            "annotation_method": "independent_human", "annotator": "reviewer-a",
            "reviewed_at": "2026-09-05T00:00:00+00:00",
            "snapshot_at": "2026-09-04T00:00:00+00:00",
            "fingerprint": "sha256:abc",
            "query_source": "deidentified-production-sample",
            "corpus_version": "docs-snapshot-v1", "privacy_status": "reviewed",
            "guideline_version": "qrels-v1",
            "dataset_author": "author-a",
        },
        "configuration": {"query_count": 20, "graded_query_count": 20,
                          "embedder": "hash:384"},
        "quality_gates": {"min_recall_at_k": 0.8, "min_mrr": 0.7},
        "verdict": {"status": "pass"},
    }
    rejected = evaluate_baseline_eligibility(report)
    assert rejected["is_publishable"] is False
    assert any("Hash Embedding" in reason for reason in rejected["reasons"])
    report["configuration"]["embedder"] = "local:BAAI/bge-m3"
    assert evaluate_baseline_eligibility(report)["is_publishable"] is True


def test_regression_baseline_requires_same_dataset_top_k_and_embedder():
    scenario = {"concurrency": 4, "throughput_qps": 10.0, "error_rate": 0.0,
                "latency_ms": {"p95": 10.0}}
    current = {
        "dataset": {"fingerprint": "sha256:new"},
        "configuration": {"top_k": 5, "embedder": "local:bge-m3"},
        "quality": {"recall_at_k": 1.0, "mrr": 1.0},
        "load": {"scenarios": [scenario]},
    }
    baseline = {
        **current,
        "dataset": {"fingerprint": "sha256:old"},
    }
    verdict = evaluate_gates(current, gate_concurrency=4, min_recall_at_k=None,
        min_mrr=None, max_p95_ms=None, max_error_rate=0.0, baseline=baseline,
        max_p95_regression_pct=20, max_qps_drop_pct=20, max_recall_drop=0.02)
    assert verdict["status"] == "fail"
    assert next(gate for gate in verdict["gates"]
                if gate["metric"] == "baseline.compatibility.dataset_fingerprint")["passed"] is False


def test_real_dataset_rejects_non_integer_and_unreachable_qrels(tmp_path):
    payload = {
        "metadata": {"kind": "real"},
        "documents": [
            {"id": "a", "project_name": "A", "content": "A",
             "source_url": "https://example.com/a"},
            {"id": "b", "project_name": "B", "content": "B",
             "source_url": "https://example.com/b"},
        ],
        "queries": [{"id": "q", "query": "test", "project_name": "A",
                     "qrels": [{"document_id": "a", "relevance": 1.5}]}],
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integer"):
        load_dataset_with_metadata(path)
    payload["queries"][0]["qrels"] = [{"document_id": "b", "relevance": 3}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="outside their project"):
        load_dataset_with_metadata(path)
