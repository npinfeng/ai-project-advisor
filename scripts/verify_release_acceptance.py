"""Verify a reviewed real run against its exact Golden suite and quality gates."""

import argparse
import json
from pathlib import Path
from typing import Any

from project_advisor.evaluation import evaluate_file, load_evaluation_bundle
from project_advisor.evaluation_suite import golden_suite_sha256, load_golden_suite


def verify_release_acceptance(
    suite_path: Path,
    run_path: Path,
    *,
    min_recall: float = 0.80,
    min_citation_accuracy: float = 0.80,
    min_citation_coverage: float = 0.80,
    min_task_success: float = 0.80,
    max_p95_ms: float | None = None,
) -> dict[str, Any]:
    suite = load_golden_suite(suite_path, require_reviewed=True)
    bundle = load_evaluation_bundle(run_path)
    report = evaluate_file(run_path, require_publishable=True)
    suite_ids = {case.case_id for case in suite.cases}
    run_ids = {case.case_id for case in bundle.cases}
    checks: list[dict[str, Any]] = []

    def add(name: str, actual: Any, expected: str, passed: bool) -> None:
        checks.append({
            "name": name,
            "actual": actual,
            "expected": expected,
            "passed": passed,
        })

    expected_hash = golden_suite_sha256(suite_path)
    add(
        "golden_suite_sha256",
        bundle.metadata.golden_suite_sha256,
        expected_hash,
        bundle.metadata.golden_suite_sha256 == expected_hash,
    )
    add(
        "golden_suite_identity",
        f"{bundle.metadata.golden_suite_name}@{bundle.metadata.golden_suite_version}",
        f"{suite.suite_name}@{suite.version}",
        bundle.metadata.golden_suite_name == suite.suite_name
        and bundle.metadata.golden_suite_version == suite.version,
    )
    add("case_ids", sorted(run_ids), "exact Golden suite case set", run_ids == suite_ids)
    add("recall_at_k", report.recall_at_k, f">= {min_recall}", report.recall_at_k >= min_recall)
    add(
        "citation_accuracy",
        report.citation_accuracy,
        f">= {min_citation_accuracy}",
        report.citation_accuracy >= min_citation_accuracy,
    )
    add(
        "citation_coverage",
        report.citation_coverage,
        f">= {min_citation_coverage}",
        report.citation_coverage >= min_citation_coverage,
    )
    add(
        "task_success_rate",
        report.task_success_rate,
        f">= {min_task_success}",
        report.task_success_rate >= min_task_success,
    )
    if max_p95_ms is not None:
        add(
            "latency_p95_ms",
            report.latency_p95_ms,
            f"<= {max_p95_ms}",
            report.latency_p95_ms <= max_p95_ms,
        )
    return {
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "suite": {
            "name": suite.suite_name,
            "version": suite.version,
            "sha256": expected_hash,
            "reviewer": suite.reviewer,
            "reviewed_at": suite.reviewed_at,
        },
        "run": {
            "path": str(run_path),
            "run_id": bundle.metadata.run_id,
            "model": bundle.metadata.model,
            "annotator": bundle.metadata.annotator,
        },
        "metrics": report.model_dump(),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Project Advisor release gates.")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--min-citation-accuracy", type=float, default=0.80)
    parser.add_argument("--min-citation-coverage", type=float, default=0.80)
    parser.add_argument("--min-task-success", type=float, default=0.80)
    parser.add_argument("--max-p95-ms", type=float)
    args = parser.parse_args()
    result = verify_release_acceptance(
        args.suite,
        args.run,
        min_recall=args.min_recall,
        min_citation_accuracy=args.min_citation_accuracy,
        min_citation_coverage=args.min_citation_coverage,
        min_task_success=args.min_task_success,
        max_p95_ms=args.max_p95_ms,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
