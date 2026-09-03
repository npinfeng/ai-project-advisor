"""Offline evaluation metrics with explicit provenance and annotation status."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    """One reproducible evaluation case."""

    case_id: str
    relevant_documents: list[str] = Field(default_factory=list)
    retrieved_documents: list[str] = Field(default_factory=list)
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    generated_citations: list[str] = Field(default_factory=list)
    supported_citations: list[str] = Field(default_factory=list)
    task_success: Optional[bool] = None
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    workflow_completed: bool = True
    run_error: Optional[str] = None
    report_sha256: Optional[str] = None


class EvaluationMetadata(BaseModel):
    """Provenance that keeps demo fixtures separate from reviewed real runs."""

    dataset_name: str = "unnamed-evaluation"
    display_name: str = "未命名评测"
    dataset_kind: Literal["fixture", "synthetic", "real_run"] = "fixture"
    annotation_status: Literal["pending", "reviewed"] = "reviewed"
    annotation_method: Literal[
        "none", "fixture", "independent_human", "llm_judge"
    ] = "none"
    annotator: Optional[str] = None
    is_publishable: bool = False
    run_id: Optional[str] = None
    model: Optional[str] = None
    created_at: Optional[str] = None
    notes: Optional[str] = None
    golden_suite_name: Optional[str] = None
    golden_suite_version: Optional[str] = None
    golden_suite_sha256: Optional[str] = None
    ground_truth_status: Optional[str] = None
    ground_truth_reviewer: Optional[str] = None
    ground_truth_reviewed_at: Optional[str] = None
    candidate_suggestion_exercised: bool = False
    recovery_exercised: bool = False
    release_preflight: dict[str, Any] = Field(default_factory=dict)


class EvaluationBundle(BaseModel):
    """Evaluation cases plus the provenance required to interpret their metrics."""

    k: int = Field(default=5, ge=1)
    metadata: EvaluationMetadata = Field(default_factory=EvaluationMetadata)
    cases: list[EvaluationCase]


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
    dataset_name: str = "adhoc"
    dataset_kind: str = "fixture"
    annotation_status: str = "reviewed"
    is_publishable: bool = False
    quality_warnings: list[str] = Field(default_factory=list)


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


def normalize_evaluation_identifier(value: str) -> str:
    """Normalize URLs without changing opaque document or Evidence IDs."""
    cleaned = value.strip().rstrip(".,;:!?，。；：！？\"'")
    if not cleaned.startswith(("http://", "https://")):
        return cleaned
    parsed = urlsplit(cleaned)
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        parsed.query,
        "",
    ))


def _reciprocal_rank(case: EvaluationCase) -> float:
    relevant = {normalize_evaluation_identifier(item) for item in case.relevant_documents}
    for rank, document_id in enumerate(case.retrieved_documents, start=1):
        if normalize_evaluation_identifier(document_id) in relevant:
            return 1 / rank
    return 0.0


def _ndcg_at_k(case: EvaluationCase, k: int) -> float:
    relevant = {normalize_evaluation_identifier(item) for item in case.relevant_documents}
    seen: set[str] = set()
    gains = []
    for document_id in case.retrieved_documents[:k]:
        normalized = normalize_evaluation_identifier(document_id)
        gain = 1 if normalized in relevant and normalized not in seen else 0
        seen.add(normalized)
        gains.append(gain)
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return _safe_ratio(dcg, ideal_dcg)


def detect_evaluation_leakage(cases: list[EvaluationCase]) -> list[str]:
    """Detect common circular-label patterns that can make metrics meaningless."""
    warnings: list[str] = []
    for case in cases:
        relevant = {normalize_evaluation_identifier(item) for item in case.relevant_documents}
        retrieved = {normalize_evaluation_identifier(item) for item in case.retrieved_documents}
        expected = {normalize_evaluation_identifier(item) for item in case.expected_citations}
        generated = {normalize_evaluation_identifier(item) for item in case.generated_citations}
        supported = {normalize_evaluation_identifier(item) for item in case.supported_citations}
        if relevant and relevant == retrieved:
            warnings.append(
                f"{case.case_id}: relevant_documents 与 retrieved_documents 完全相同，需确认标注不是从运行结果复制。"
            )
        if expected and expected == generated == supported:
            warnings.append(
                f"{case.case_id}: expected/generated/supported citations 完全相同，需确认引用经过独立审核。"
            )
    return warnings


def evaluate_cases(
    cases: list[EvaluationCase],
    k: int = 5,
    metadata: EvaluationMetadata | None = None,
) -> EvaluationReport:
    """Calculate transparent retrieval, citation, quality, latency, and cost metrics."""
    if not cases:
        raise ValueError("评测数据至少需要一个 case。")
    if k < 1:
        raise ValueError("k 必须大于等于 1。")
    metadata = metadata or EvaluationMetadata(
        dataset_name="adhoc",
        display_name="临时评测",
    )
    if metadata.annotation_status != "reviewed":
        raise ValueError("评测数据仍处于 pending 状态，完成人工标注后才能计算质量指标。")
    missing_success = [case.case_id for case in cases if case.task_success is None]
    if missing_success:
        raise ValueError(
            "以下 case 尚未标注 task_success：" + "、".join(missing_success)
        )
    quality_warnings = detect_evaluation_leakage(cases)
    if metadata.is_publishable and (
        metadata.dataset_kind != "real_run"
        or metadata.annotation_method != "independent_human"
        or not metadata.annotator
    ):
        raise ValueError(
            "可发布基线必须来自 real_run，并记录独立人工审核方法和 annotator。"
        )
    if metadata.is_publishable and (
        metadata.ground_truth_status != "reviewed"
        or not metadata.golden_suite_sha256
        or not metadata.ground_truth_reviewer
        or not metadata.ground_truth_reviewed_at
        or not metadata.candidate_suggestion_exercised
        or not metadata.recovery_exercised
    ):
        raise ValueError(
            "可发布基线必须绑定已审核 Golden suite，并通过候选建议和 checkpoint 恢复验收。"
        )
    failed_runs = [
        case.case_id
        for case in cases
        if not case.workflow_completed or case.run_error or not case.report_sha256
    ]
    if metadata.is_publishable and failed_runs:
        raise ValueError(
            "以下 case 没有成功完成真实工作流：" + "、".join(failed_runs)
        )

    recalls = []
    precisions = []
    reciprocal_ranks = []
    ndcgs = []
    generated_citations = 0
    supported_citations = 0
    expected_citations = 0
    covered_citations = 0

    for case in cases:
        relevant = {normalize_evaluation_identifier(item) for item in case.relevant_documents}
        retrieved = [
            normalize_evaluation_identifier(item)
            for item in case.retrieved_documents[:k]
        ]
        hits = len(set(retrieved) & relevant)
        recalls.append(_safe_ratio(hits, len(relevant)))
        precisions.append(_safe_ratio(hits, k))
        reciprocal_ranks.append(_reciprocal_rank(case))
        ndcgs.append(_ndcg_at_k(case, k))

        generated = {normalize_evaluation_identifier(item) for item in case.generated_citations}
        supported = {normalize_evaluation_identifier(item) for item in case.supported_citations}
        expected = {normalize_evaluation_identifier(item) for item in case.expected_citations}
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
        task_success_rate=mean(bool(case.task_success) for case in cases),
        latency_p50_ms=_percentile([case.latency_ms for case in cases], 0.50),
        latency_p95_ms=_percentile([case.latency_ms for case in cases], 0.95),
        average_tokens=mean(total_tokens),
        average_cost_usd=mean(case.cost_usd for case in cases),
        dataset_name=metadata.dataset_name,
        dataset_kind=metadata.dataset_kind,
        annotation_status=metadata.annotation_status,
        is_publishable=metadata.is_publishable,
        quality_warnings=quality_warnings,
    )
    return EvaluationReport.model_validate(
        {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in report.model_dump().items()
        }
    )


def load_evaluation_bundle(path: Path) -> EvaluationBundle:
    """Load both cases and provenance, including legacy `_metadata` files."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    metadata_payload = payload.get("metadata") or payload.get("_metadata") or {}
    if not metadata_payload:
        metadata_payload = {
            "dataset_name": path.stem,
            "display_name": path.stem,
            "dataset_kind": "fixture",
            "annotation_status": "reviewed",
            "is_publishable": False,
            "notes": "Legacy evaluation file without explicit provenance.",
        }
    return EvaluationBundle.model_validate({
        "k": payload.get("k", 5),
        "metadata": metadata_payload,
        "cases": payload.get("cases", []),
    })


def load_evaluation_file(path: Path) -> tuple[list[EvaluationCase], int]:
    bundle = load_evaluation_bundle(path)
    return bundle.cases, bundle.k


def evaluate_file(path: Path, *, require_publishable: bool = False) -> EvaluationReport:
    bundle = load_evaluation_bundle(path)
    if require_publishable and not bundle.metadata.is_publishable:
        raise ValueError("该数据集不是经过审核的可发布基线。")
    return evaluate_cases(bundle.cases, k=bundle.k, metadata=bundle.metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Project Advisor offline results.")
    parser.add_argument("--input", type=Path, required=True, help="Evaluation JSON file")
    parser.add_argument("--output", type=Path, help="Optional report JSON path")
    parser.add_argument(
        "--require-publishable",
        action="store_true",
        help="Reject fixtures, synthetic data, and unreviewed runs",
    )
    args = parser.parse_args()

    output = json.dumps(
        evaluate_file(
            args.input, require_publishable=args.require_publishable
        ).model_dump(),
        ensure_ascii=False,
        indent=2,
    )
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
