"""Persistence helpers that turn collected tool evidence into reusable knowledge."""

from pathlib import Path
from threading import Lock

from project_advisor.rag.document_store import DocumentStore
from project_advisor.rag.evidence_lifecycle import is_valid_evidence_url
from project_advisor.schemas.evidence import Evidence


_STORE_LOCK = Lock()


def persist_evidences(
    values: list,
    *,
    storage_dir: str | Path = "./data/documents",
) -> dict:
    """Validate, de-duplicate and persist evidence without breaking the workflow."""
    evidences: list[Evidence] = []
    rejected_invalid_url = 0
    seen: set[str] = set()
    for value in values:
        try:
            evidence = value if isinstance(value, Evidence) else Evidence.model_validate(value)
        except (TypeError, ValueError):
            continue
        if not is_valid_evidence_url(evidence.source_url):
            rejected_invalid_url += 1
            continue
        if evidence.evidence_id in seen:
            continue
        seen.add(evidence.evidence_id)
        evidences.append(evidence)

    with _STORE_LOCK:
        store = DocumentStore(storage_dir=str(storage_dir))
        stored = store.add_batch(evidences)
        total_documents = store.get_stats()["total_documents"]
    return {
        "received": len(values),
        "valid": len(evidences),
        "rejected_invalid_url": rejected_invalid_url,
        "stored": stored,
        "total_documents": total_documents,
        "projects": sorted({evidence.project_name for evidence in evidences}),
    }
