"""Separate model-facing observations from machine-readable provenance artifacts."""

from datetime import datetime, timezone
from functools import wraps
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from project_advisor.schemas.evidence import Evidence


class SourceDocument(BaseModel):
    source_url: str
    content: str
    source_type: str = "web_search"
    evidence_kind: Literal["primary", "search_snippet", "discovery", "inference", "unverified"] = "primary"
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_date: str | None = None
    version_info: str | None = None
    locator: str = ""
    truncated: bool = False


class SourceArtifact(BaseModel):
    schema_version: int = 1
    documents: list[SourceDocument] = Field(default_factory=list)
    reused_evidences: list[Evidence] = Field(default_factory=list)


def sourced(content: str, documents=(), *, reused_evidences=()):
    return content, SourceArtifact(
        documents=list(documents), reused_evidences=list(reused_evidences)
    ).model_dump()


def source_tool(*, description: str):
    """Keep direct .ainvoke callers compatible while retaining artifacts at runtime."""
    def decorate(function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            result = await function(*args, **kwargs)
            return sourced(result) if isinstance(result, str) else result
        return tool(description=description, response_format="content_and_artifact")(wrapped)
    return decorate
