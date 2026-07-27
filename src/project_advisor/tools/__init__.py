"""Project Advisor tools exposed without eagerly importing RAG dependencies."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "fetch_webpage": ("project_advisor.tools.document_collector", "fetch_webpage"),
    "web_fetch_tool": ("project_advisor.tools.document_collector", "web_fetch_tool"),
    "batch_fetch_tool": ("project_advisor.tools.document_collector", "batch_fetch_tool"),
    "collect_documents": (
        "project_advisor.tools.document_collector",
        "collect_documents",
    ),
    "rag_search": ("project_advisor.tools.rag_search", "rag_search"),
    "rag_ingest": ("project_advisor.tools.rag_search", "rag_ingest"),
    "rag_rebuild": ("project_advisor.tools.rag_search", "rag_rebuild"),
    "rag_status": ("project_advisor.tools.rag_search", "rag_status"),
    "model_info": ("project_advisor.tools.model_registry", "model_info"),
    "check_local_feasibility": (
        "project_advisor.tools.model_registry",
        "check_local_feasibility",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load each tool only when its package-level attribute is requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
