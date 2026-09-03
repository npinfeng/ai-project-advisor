"""Reviewer context-budget and Evidence lifecycle regression tests."""

from datetime import datetime, timezone

import pytest

from project_advisor.agents.context_budget import build_evidence_payload
from project_advisor.rag.document_store import DocumentStore
from project_advisor.rag.evidence_lifecycle import (
    EvidenceLifecyclePolicy,
    classify_evidence,
    resolve_current_evidences,
)
from project_advisor.rag.knowledge_store import persist_evidences
from project_advisor.schemas.evidence import Evidence
from project_advisor.tools.citations import validate_citation


NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def evidence(
    *,
    url: str,
    content: str,
    source_type: str = "official_documentation",
    source_date: str = "2026-09-01T00:00:00+00:00",
    version: str | None = None,
) -> Evidence:
    return Evidence(
        source_url=url,
        source_type=source_type,
        project_name="Example",
        content=content,
        relevance="feature_match",
        confidence="high",
        retrieved_at="2026-09-03T00:00:00+00:00",
        source_date=source_date,
        version_info=version,
    )


def test_context_budget_prioritizes_sources_deduplicates_and_stays_bounded():
    official = evidence(
        url="https://docs.example.com/official",
        content="same authoritative statement " * 20,
    )
    duplicate_blog = evidence(
        url="https://blog.example.com/copy",
        content=official.content,
        source_type="blog",
    )
    community = evidence(
        url="https://forum.example.com/thread",
        content="lower priority evidence " * 50,
        source_type="community",
    )

    payload, diagnostics = build_evidence_payload(
        [community, duplicate_blog, official],
        max_chars=650,
        max_chars_per_evidence=300,
        now=NOW,
    )

    assert payload[0]["evidence_id"] == official.evidence_id
    assert diagnostics["duplicate_count"] == 1
    assert diagnostics["selected_chars"] <= diagnostics["max_chars"]
    assert diagnostics["over_budget"] is True


def test_lifecycle_classification_and_apply_cleanup(tmp_path):
    policy = EvidenceLifecyclePolicy(stale_after_days=30, expire_after_days=60)
    active = evidence(url="https://docs.example.com/active", content="active")
    stale = evidence(
        url="https://docs.example.com/stale",
        content="stale",
        source_date="2026-07-20T00:00:00+00:00",
    )
    expired = evidence(
        url="https://docs.example.com/expired",
        content="expired",
        source_date="2026-01-01T00:00:00+00:00",
    )
    invalid = evidence(url="not a url", content="invalid")

    assert classify_evidence(active, policy, now=NOW)[0] == "active"
    assert classify_evidence(stale, policy, now=NOW)[0] == "stale"
    assert classify_evidence(expired, policy, now=NOW)[0] == "expired"
    assert classify_evidence(invalid, policy, now=NOW)[0] == "invalid"
    assert validate_citation(invalid, "claim")["is_valid"] is False

    store = DocumentStore(storage_dir=str(tmp_path))
    store.add_batch([active, stale, expired, invalid])
    # Invalid URLs are rejected on ingestion, while expired records remain auditable
    # until an explicit maintenance apply.
    assert store.get_stats()["total_documents"] == 3
    preview = store.maintain(policy=policy, dry_run=True, now=NOW)
    assert preview["removed_count"] == 1
    assert store.get_stats()["total_documents"] == 3
    applied = store.maintain(policy=policy, dry_run=False, now=NOW)
    assert applied["removed_evidence_ids"] == [expired.evidence_id]
    assert store.get_stats()["total_documents"] == 2


def test_content_change_keeps_history_but_only_newest_version_is_current(tmp_path):
    old = evidence(
        url="https://docs.example.com/versioned",
        content="old behavior",
        source_date="2026-08-01T00:00:00+00:00",
        version="v1",
    )
    new = evidence(
        url="https://docs.example.com/versioned",
        content="new behavior",
        source_date="2026-09-01T00:00:00+00:00",
        version="v2",
    )
    store = DocumentStore(storage_dir=str(tmp_path))
    assert store.add_batch([old, new]) == 2

    current, conflicts = resolve_current_evidences(store.get_by_project("Example"))
    assert [item.evidence_id for item in current] == [new.evidence_id]
    assert conflicts[0]["selected_evidence_id"] == new.evidence_id
    assert conflicts[0]["superseded_evidence_ids"] == [old.evidence_id]
    assert store.get_current_by_project("Example")[0].evidence_id == new.evidence_id


def test_persistence_reports_invalid_urls(tmp_path):
    result = persist_evidences(
        [evidence(url="bad", content="bad")], storage_dir=tmp_path
    )
    assert result["valid"] == 0
    assert result["rejected_invalid_url"] == 1
    with pytest.raises(ValueError, match="无效 Evidence URL"):
        DocumentStore(storage_dir=str(tmp_path)).add(evidence(url="bad", content="bad"))
