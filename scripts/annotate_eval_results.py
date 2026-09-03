"""Independently annotate a pending real-run evaluation file."""

import argparse
import json
from pathlib import Path
from typing import Any


def finalize_annotations(
    payload: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    *,
    annotator: str,
    ground_truth_confirmed: bool,
) -> dict[str, Any]:
    """Apply explicit human decisions without deriving labels from predictions."""
    metadata = payload.get("metadata", {})
    if metadata.get("dataset_kind") != "real_run":
        raise ValueError("只能审核真实运行产生的 real_run 文件。")
    if metadata.get("annotation_status") != "pending":
        raise ValueError("输入文件不是 pending 状态。")
    if not annotator.strip():
        raise ValueError("annotator 不能为空。")
    if ground_truth_confirmed:
        required_provenance = (
            "golden_suite_name",
            "golden_suite_version",
            "golden_suite_sha256",
            "ground_truth_reviewer",
            "ground_truth_reviewed_at",
        )
        missing = [key for key in required_provenance if not metadata.get(key)]
        if metadata.get("ground_truth_status") != "reviewed" or missing:
            raise ValueError(
                "不能仅凭确认选项发布：运行结果必须绑定 reviewed Golden suite；"
                f"缺少 {', '.join(missing) or 'reviewed status'}。"
            )
        if not metadata.get("candidate_suggestion_exercised"):
            raise ValueError("发布验收没有执行候选建议链路。")
        if not metadata.get("recovery_exercised"):
            raise ValueError("发布验收没有执行 checkpoint 恢复链路。")

    cases = payload.get("cases", [])
    expected_ids = {case.get("case_id") for case in cases}
    if set(decisions) != expected_ids:
        raise ValueError("每个 case 都必须提供独立审核结果。")

    reviewed_cases = []
    for case in cases:
        case_id = case["case_id"]
        decision = decisions[case_id]
        generated = set(case.get("generated_citations", []))
        supported = list(dict.fromkeys(decision.get("supported_citations", [])))
        if not set(supported).issubset(generated):
            raise ValueError(f"{case_id} 的 supported_citations 包含未生成的 URL。")
        if not isinstance(decision.get("task_success"), bool):
            raise ValueError(f"{case_id} 的 task_success 必须是布尔值。")
        if ground_truth_confirmed and (
            not case.get("workflow_completed")
            or case.get("run_error")
            or not case.get("report_sha256")
        ):
            raise ValueError(f"{case_id} 的真实工作流没有成功完成，不能发布。")
        reviewed_cases.append({
            **case,
            "supported_citations": supported,
            "task_success": decision["task_success"],
            "annotation_context": {
                **case.get("annotation_context", {}),
                "review_notes": decision.get("review_notes", ""),
            },
        })

    reviewed_metadata = {
        **metadata,
        "display_name": str(metadata.get("display_name", "")).replace(
            "（待审核）", "（人工审核）"
        ),
        "annotation_status": "reviewed",
        "annotation_method": "independent_human",
        "annotator": annotator.strip(),
        "is_publishable": ground_truth_confirmed,
        "notes": (
            f"{metadata.get('notes', '')} "
            f"ground_truth_confirmed={ground_truth_confirmed}."
        ).strip(),
    }
    return {**payload, "metadata": reviewed_metadata, "cases": reviewed_cases}


def _yes_no(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().casefold()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("请输入 y 或 n。")


def _collect_decisions(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    decisions = {}
    for case in payload.get("cases", []):
        case_id = case["case_id"]
        context = case.get("annotation_context", {})
        print(f"\n=== {case_id} ===")
        print(f"成功标准：{context.get('success_criteria', '未记录')}")
        if context.get("run_error"):
            print(f"运行错误：{context['run_error']}")
        report = context.get("generated_report") or context.get("report_preview", "")
        print(f"完整报告：\n{report}\n")

        supported = []
        for url in case.get("generated_citations", []):
            if _yes_no(f"该引用确实支持报告中的对应结论吗？\n{url}"):
                supported.append(url)
        task_success = _yes_no("该任务是否满足上述成功标准？")
        review_notes = input("审核备注（可留空）：").strip()
        decisions[case_id] = {
            "supported_citations": supported,
            "task_success": task_success,
            "review_notes": review_notes,
        }
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a pending evaluation run.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--annotator", required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    decisions = _collect_decisions(payload)
    ground_truth_confirmed = _yes_no(
        "你是否独立核对了 golden suite 中的相关文档和期望引用？"
    )
    reviewed = finalize_annotations(
        payload,
        decisions,
        annotator=args.annotator,
        ground_truth_confirmed=ground_truth_confirmed,
    )
    output_path = args.output or args.input.with_name(
        args.input.name.replace(".pending.json", ".reviewed.json")
    )
    output_path.write_text(
        json.dumps(reviewed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"审核结果已写入：{output_path}")
    if reviewed["metadata"]["is_publishable"]:
        print("该文件满足正式基线的来源与人工审核门禁。")
    else:
        print("该文件可计算调试指标，但不能标记为正式可发布基线。")


if __name__ == "__main__":
    main()
