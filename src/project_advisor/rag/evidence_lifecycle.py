"""Evidence lifecycle policy, URL validation, and version resolution."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Literal
from urllib.parse import urlsplit, urlunsplit

from project_advisor.schemas.evidence import Evidence


LifecycleStatus = Literal["active", "stale", "expired", "invalid"]


@dataclass(frozen=True)
class EvidenceLifecyclePolicy:
    """Retention thresholds used by status, indexing, and maintenance."""

    stale_after_days: int = 180
    expire_after_days: int = 365

    def __post_init__(self) -> None:
        if self.stale_after_days < 0:
            raise ValueError("stale_after_days must be non-negative")
        if self.expire_after_days <= self.stale_after_days:
            raise ValueError("expire_after_days must be greater than stale_after_days")


def canonical_source_url(value: str) -> str:
    """Normalize a source URL for version grouping without changing its meaning."""
    try:
        parsed = urlsplit((value or "").strip())
    except ValueError:
        return ""
    scheme = parsed.scheme.casefold()
    hostname = (parsed.hostname or "").casefold()
    if scheme in {"http", "https"} and hostname:
        try:
            parsed_port = parsed.port
        except ValueError:
            return ""
        port = f":{parsed_port}" if parsed_port else ""
        path = parsed.path.rstrip("/") or "/"
        return urlunsplit((scheme, f"{hostname}{port}", path, parsed.query, ""))
    if scheme == "tool" and (parsed.netloc or parsed.path):
        return urlunsplit((scheme, parsed.netloc.casefold(), parsed.path, parsed.query, ""))
    return ""


def is_valid_evidence_url(value: str) -> bool:
    """Accept public-style HTTP(S) citations and internal ``tool://`` provenance."""
    try:
        return bool(canonical_source_url(value))
    except (ValueError, TypeError):
        return False


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def evidence_reference_time(evidence: Evidence) -> datetime | None:
    """Prefer the source's own date; fall back to collection time for retention."""
    return _parse_datetime(evidence.source_date) or _parse_datetime(evidence.retrieved_at)


def classify_evidence(
    evidence: Evidence,
    policy: EvidenceLifecyclePolicy | None = None,
    *,
    now: datetime | None = None,
) -> tuple[LifecycleStatus, int | None]:
    """Classify one record and return ``(status, age_days)``."""
    policy = policy or EvidenceLifecyclePolicy()
    if not is_valid_evidence_url(evidence.source_url):
        return "invalid", None
    reference = evidence_reference_time(evidence)
    if reference is None:
        return "stale", None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age_days = max(0, int((now - reference).total_seconds() // 86400))
    if age_days > policy.expire_after_days:
        return "expired", age_days
    if age_days > policy.stale_after_days:
        return "stale", age_days
    return "active", age_days


def _content_fingerprint(evidence: Evidence) -> str:
    normalized = " ".join((evidence.content or "").casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _version_sort_key(evidence: Evidence) -> tuple[datetime, datetime, str, str]:
    timestamp = evidence_reference_time(evidence) or datetime.min.replace(
        tzinfo=timezone.utc
    )
    retrieved = _parse_datetime(evidence.retrieved_at) or datetime.min.replace(
        tzinfo=timezone.utc
    )
    natural_version = re.sub(
        r"\d+",
        lambda match: match.group(0).zfill(20),
        evidence.version_info or "",
    )
    return timestamp, retrieved, natural_version, evidence.evidence_id


def resolve_current_evidences(
    evidences: Iterable[Evidence],
) -> tuple[list[Evidence], list[dict]]:
    """Keep the newest record for each project/source and report content conflicts."""
    groups: dict[tuple[str, str, str], list[Evidence]] = {}
    passthrough: list[Evidence] = []
    for evidence in evidences:
        canonical = canonical_source_url(evidence.source_url)
        if not canonical:
            passthrough.append(evidence)
            continue
        key = (
            evidence.project_name.casefold().strip(),
            canonical,
            evidence.relevance.casefold().strip(),
        )
        groups.setdefault(key, []).append(evidence)

    selected = list(passthrough)
    conflicts: list[dict] = []
    for (_, canonical, relevance), group in groups.items():
        ordered = sorted(group, key=_version_sort_key, reverse=True)
        winner = ordered[0]
        selected.append(winner)
        fingerprints = {_content_fingerprint(item) for item in group}
        versions = {item.version_info for item in group if item.version_info}
        if len(group) > 1 and (len(fingerprints) > 1 or len(versions) > 1):
            conflicts.append({
                "type": "version_conflict",
                "project": winner.project_name,
                "url": canonical,
                "topic": relevance,
                "selected_evidence_id": winner.evidence_id,
                "superseded_evidence_ids": [item.evidence_id for item in ordered[1:]],
                "versions": sorted(versions),
                "detail": (
                    f"来源 {canonical} 存在 {len(group)} 个内容/版本记录；"
                    f"检索仅使用最新证据 {winner.evidence_id}。"
                ),
            })
    return sorted(selected, key=_version_sort_key, reverse=True), conflicts


def lifecycle_snapshot(
    evidences: Iterable[Evidence],
    policy: EvidenceLifecyclePolicy | None = None,
    *,
    now: datetime | None = None,
) -> dict:
    """Return lifecycle counts plus deterministic current-version diagnostics."""
    values = list(evidences)
    counts = {"active": 0, "stale": 0, "expired": 0, "invalid": 0}
    ages: list[int] = []
    for evidence in values:
        status, age_days = classify_evidence(evidence, policy, now=now)
        counts[status] += 1
        if age_days is not None:
            ages.append(age_days)
    valid = [
        evidence
        for evidence in values
        if classify_evidence(evidence, policy, now=now)[0] != "invalid"
    ]
    indexable = [
        evidence
        for evidence in valid
        if classify_evidence(evidence, policy, now=now)[0] in {"active", "stale"}
    ]
    current, conflicts = resolve_current_evidences(valid)
    indexable_current, _ = resolve_current_evidences(indexable)
    return {
        "total": len(values),
        "counts": counts,
        "current_version_count": len(current),
        "indexable_current_count": len(indexable_current),
        "superseded_count": max(0, len(valid) - len(current)),
        "oldest_age_days": max(ages, default=None),
        "newest_age_days": min(ages, default=None),
        "version_conflicts": conflicts,
    }
