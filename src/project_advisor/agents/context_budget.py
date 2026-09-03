"""Deterministic evidence selection for bounded Reviewer prompts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from project_advisor.schemas.evidence import Evidence
from project_advisor.tools.citations import get_evidence_quality_score


_CONFIDENCE_PRIORITY = {"high": 3, "medium": 2, "low": 1}


def _fingerprint(content: str) -> str:
    normalized = " ".join(content.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sort_key(evidence: Evidence, now: datetime) -> tuple[float, int, str]:
    return (
        get_evidence_quality_score(evidence, now),
        _CONFIDENCE_PRIORITY.get(evidence.confidence.casefold(), 0),
        evidence.source_date or evidence.retrieved_at,
    )


def build_evidence_payload(
    evidences: list[Evidence],
    *,
    max_chars: int,
    max_chars_per_evidence: int,
    now: datetime | None = None,
) -> tuple[list[dict], dict]:
    """Prioritize, de-duplicate, and trim evidence to a hard serialized budget."""
    if max_chars < 2 or max_chars_per_evidence < 1:
        raise ValueError("Reviewer context budgets must be positive")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    ordered = sorted(evidences, key=lambda item: _sort_key(item, now), reverse=True)
    unique: list[Evidence] = []
    seen_content: set[str] = set()
    duplicate_count = 0
    for evidence in ordered:
        fingerprint = _fingerprint(evidence.content or "")
        if fingerprint in seen_content:
            duplicate_count += 1
            continue
        seen_content.add(fingerprint)
        unique.append(evidence)

    payload: list[dict] = []
    truncated_count = 0
    dropped_for_budget = 0
    for evidence in unique:
        full_content = evidence.content or ""
        content = full_content[:max_chars_per_evidence]
        truncated = len(content) < len(full_content)
        entry = {**evidence.model_dump(exclude={"content"}), "content": content}
        if truncated:
            entry["content_truncated"] = True
            entry["original_content_chars"] = len(full_content)

        candidate = [*payload, entry]
        serialized_chars = len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")))
        if serialized_chars <= max_chars:
            payload = candidate
            truncated_count += int(truncated)
            continue

        # Use any remaining room for a partial final record, while still counting
        # the JSON envelope and metadata against the hard budget.
        entry["content"] = ""
        base_chars = len(json.dumps([*payload, entry], ensure_ascii=False, separators=(",", ":")))
        remaining = max_chars - base_chars
        if remaining > 0:
            entry["content"] = content[:remaining]
            entry["content_truncated"] = True
            entry["original_content_chars"] = len(full_content)
            candidate = [*payload, entry]
            while len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))) > max_chars:
                entry["content"] = entry["content"][:-1]
            if entry["content"]:
                payload = candidate
                truncated_count += 1
                continue
        dropped_for_budget += 1

    serialized_chars = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    dropped_for_budget = len(unique) - len(payload)
    diagnostics = {
        "input_count": len(evidences),
        "selected_count": len(payload),
        "duplicate_count": duplicate_count,
        "dropped_for_budget": max(0, dropped_for_budget),
        "truncated_count": truncated_count,
        "selected_chars": serialized_chars,
        "max_chars": max_chars,
        "over_budget": dropped_for_budget > 0 or truncated_count > 0,
        "compressed": (
            duplicate_count > 0
            or dropped_for_budget > 0
            or truncated_count > 0
        ),
        "selection_order": "source_authority_freshness_confidence",
    }
    return payload, diagnostics
