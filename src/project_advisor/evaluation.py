"""Offline evaluation metrics for retrieval, citations, and end-to-end runs."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    """One reproducible evaluation case."""

    case_id: str
    relevant_documents: list[str] = Field(default_factory=list)
    retrieved_documents: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    generated_citations: list[str] = Field(default_factory=list)
    supported_citations: list[str] = Field(default_factory=list)
    task_success: bool
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)


class EvaluationReport(BaseModel):
    """Aggregated metrics kept separate from candidate project scores."""

    case_count: int
    k: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    citation_accuracy: float
    citation_coverage: float
    task_success_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    average_tokens: float
    average_cost_usd: float


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _reciprocal_rank(case: EvaluationCase) -> float:
    relevant = set(case.relevant_documents)
    for rank, document_id in enumerate(case.retrieved_documents, start=1):
        if document_id in relevant:
            return 1 / rank
    return 0.0


def _ndcg_at_k(case: EvaluationCase, k: int) -> float:
    relevant = set(case.relevant_documents)
    gains = [1 if document_id in relevant else 0 for document_id in case.retrieved_documents[:k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return _safe_ratio(dcg, ideal_dcg)


def evaluate_cases(cases: list[EvaluationCase], k: int = 5) -> EvaluationReport:
    """Calculate transparent retrieval, citation, quality, latency, and cost metrics."""
    if not cases:
        raise ValueError("评测数据至少需要一个 case。")
    if k < 1:
        raise ValueError("k 必须大于等于 1。")

    recalls = []
    precisions = []
    reciprocal_ranks = []
    ndcgs = []
    generated_citations = 0
    supported_citations = 0
    expected_citations = 0
    covered_citations = 0

    for case in cases:
        relevant = set(case.relevant_documents)
        retrieved = case.retrieved_documents[:k]
        hits = sum(document_id in relevant for document_id in retrieved)
        recalls.append(_safe_ratio(hits, len(relevant)))
        precisions.append(_safe_ratio(hits, k))
        reciprocal_ranks.append(_reciprocal_rank(case))
        ndcgs.append(_ndcg_at_k(case, k))

        generated = set(case.generated_citations)
        supported = set(case.supported_citations)
        expected = set(case.expected_citations)
        generated_citations += len(generated)
        supported_citations += len(generated & supported)
        expected_citations += len(expected)
        covered_citations += len(generated & expected)

    total_tokens = [case.input_tokens + case.output_tokens for case in cases]
    report = EvaluationReport(
        case_count=len(cases),
        k=k,
        recall_at_k=mean(recalls),
        precision_at_k=mean(precisions),
        mrr=mean(reciprocal_ranks),
        ndcg_at_k=mean(ndcgs),
        citation_accuracy=_safe_ratio(supported_citations, generated_citations),
        citation_coverage=_safe_ratio(covered_citations, expected_citations),
        task_success_rate=mean(case.task_success for case in cases),
        latency_p50_ms=_percentile([case.latency_ms for case in cases], 0.50),
        latency_p95_ms=_percentile([case.latency_ms for case in cases], 0.95),
        average_tokens=mean(total_tokens),
        average_cost_usd=mean(case.cost_usd for case in cases),
    )
    return EvaluationReport.model_validate(
        {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in report.model_dump().items()
        }
    )


def load_evaluation_file(path: Path) -> tuple[list[EvaluationCase], int]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    cases = [EvaluationCase.model_validate(item) for item in payload.get("cases", [])]
    return cases, int(payload.get("k", 5))


def evaluate_file(path: Path) -> EvaluationReport:
    cases, k = load_evaluation_file(path)
    return evaluate_cases(cases, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Project Advisor offline results.")
    parser.add_argument("--input", type=Path, required=True, help="Evaluation JSON file")
    parser.add_argument("--output", type=Path, help="Optional report JSON path")
    args = parser.parse_args()

    output = json.dumps(evaluate_file(args.input).model_dump(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
