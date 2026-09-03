"""Interactively review Golden Case labels and create a signed review artifact."""

import argparse
import json
from pathlib import Path
from typing import Any

from project_advisor.evaluation_suite import apply_golden_reviews


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "evals" / "golden_cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evals" / "golden_cases.reviewed.json"


def _yes_no(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().casefold()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("请输入 y 或 n。")


def _collect_decisions(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for case in payload.get("cases", []):
        print(f"\n=== {case['case_id']} ===")
        print(f"问题：{case['question']}")
        print("相关文档：")
        print("\n".join(f"- {url}" for url in case["relevant_documents"]))
        documents_verified = _yes_no("是否逐个打开并确认这些文档与问题相关？")
        print("期望引用：")
        print("\n".join(f"- {url}" for url in case["expected_citations"]))
        citations_verified = _yes_no("是否逐个确认这些页面能支持预期关键结论？")
        print(f"成功标准：{case['success_criteria']}")
        criteria_verified = _yes_no("成功标准是否明确、可由独立审核人判定？")
        approved = documents_verified and citations_verified and criteria_verified
        if approved:
            approved = _yes_no("确认批准该 Case 的 ground truth？")
        notes = input("审核备注（建议记录证据范围或修订原因）：").strip()
        decisions[case["case_id"]] = {
            "decision": "approved" if approved else "rejected",
            "relevant_documents_verified": documents_verified,
            "expected_citations_verified": citations_verified,
            "success_criteria_verified": criteria_verified,
            "notes": notes,
        }
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently review Golden Case ground truth."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--decisions",
        type=Path,
        help="Optional reviewer-authored JSON decisions; otherwise prompt interactively.",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    decisions = (
        json.loads(args.decisions.read_text(encoding="utf-8"))
        if args.decisions
        else _collect_decisions(payload)
    )
    reviewed = apply_golden_reviews(
        payload,
        decisions,
        reviewer=args.reviewer,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(reviewed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Golden review 已写入：{args.output}")
    print(f"状态：{reviewed['ground_truth_status']}")
    if reviewed["ground_truth_status"] != "reviewed":
        print("存在 rejected Case，发布验收仍会被阻止；请先修订 draft 后重新审核。")


if __name__ == "__main__":
    main()
