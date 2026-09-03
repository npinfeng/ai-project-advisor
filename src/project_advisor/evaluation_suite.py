"""Golden-suite validation and independent review provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, model_validator


class GoldenCaseReview(BaseModel):
    """Explicit human verification for one pre-defined case."""

    decision: Literal["approved", "rejected"]
    relevant_documents_verified: bool
    expected_citations_verified: bool
    success_criteria_verified: bool
    notes: str = ""


class GoldenCase(BaseModel):
    case_id: str = Field(min_length=1)
    question: str = Field(min_length=10)
    candidates: list[str] = Field(min_length=1, max_length=8)
    relevant_documents: list[str] = Field(min_length=1)
    expected_citations: list[str] = Field(min_length=1)
    success_criteria: str = Field(min_length=1)
    human_review: GoldenCaseReview | None = None


class GoldenSuite(BaseModel):
    """Versioned labels that must be approved before a release run."""

    suite_name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    ground_truth_status: Literal[
        "draft_human_review_required", "reviewed", "rejected"
    ]
    review_method: Literal["independent_human"] | None = None
    reviewer: str | None = None
    reviewed_at: str | None = None
    k: int = Field(default=5, ge=1)
    cases: list[GoldenCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_review_provenance(self) -> "GoldenSuite":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Golden suite 的 case_id 必须唯一。")
        for case in self.cases:
            for value in [*case.relevant_documents, *case.expected_citations]:
                parsed = urlsplit(value)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise ValueError(f"{case.case_id} 包含无效标注 URL：{value}")
        if self.ground_truth_status == "reviewed":
            if self.review_method != "independent_human" or not (self.reviewer or "").strip():
                raise ValueError("reviewed Golden suite 必须记录独立人工 reviewer。")
            try:
                datetime.fromisoformat((self.reviewed_at or "").replace("Z", "+00:00"))
            except (TypeError, ValueError) as error:
                raise ValueError("reviewed Golden suite 必须记录有效 reviewed_at。") from error
            incomplete = [
                case.case_id
                for case in self.cases
                if case.human_review is None
                or case.human_review.decision != "approved"
                or not case.human_review.relevant_documents_verified
                or not case.human_review.expected_citations_verified
                or not case.human_review.success_criteria_verified
            ]
            if incomplete:
                raise ValueError(
                    "以下 case 尚未完成全部 ground-truth 核对：" + "、".join(incomplete)
                )
        return self


def load_golden_suite(path: Path, *, require_reviewed: bool = False) -> GoldenSuite:
    suite = GoldenSuite.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if require_reviewed and suite.ground_truth_status != "reviewed":
        raise ValueError(
            "Golden suite 尚未完成独立人工审核；发布验收拒绝使用 draft 标签。"
        )
    return suite


def golden_suite_sha256(path: Path) -> str:
    """Hash the exact reviewed artifact bound to an evaluation run."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_golden_reviews(
    payload: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    *,
    reviewer: str,
    reviewed_at: datetime | None = None,
) -> dict[str, Any]:
    """Apply independent decisions without mutating the source draft in place."""
    draft = GoldenSuite.model_validate(payload)
    if draft.ground_truth_status != "draft_human_review_required":
        raise ValueError("只能审核 draft_human_review_required Golden suite。")
    if not reviewer.strip():
        raise ValueError("reviewer 不能为空。")
    expected_ids = {case.case_id for case in draft.cases}
    if set(decisions) != expected_ids:
        raise ValueError("每个 Golden case 都必须提供独立审核结果。")

    cases = []
    for case in draft.cases:
        review = GoldenCaseReview.model_validate(decisions[case.case_id])
        cases.append({**case.model_dump(exclude={"human_review"}), "human_review": review.model_dump()})
    status = (
        "reviewed"
        if all(item["human_review"]["decision"] == "approved" for item in cases)
        else "rejected"
    )
    reviewed_at = reviewed_at or datetime.now(timezone.utc)
    result = {
        **draft.model_dump(exclude={"cases"}),
        "ground_truth_status": status,
        "review_method": "independent_human",
        "reviewer": reviewer.strip(),
        "reviewed_at": reviewed_at.isoformat(),
        "cases": cases,
    }
    return GoldenSuite.model_validate(result).model_dump()
