"""Validated request contracts for the Project Advisor HTTP API."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AdviceRequest(BaseModel):
    """Input accepted by the streaming advisor endpoint."""

    question: str = Field(min_length=10, max_length=5000)
    candidates: list[str] = Field(default_factory=list, max_length=8)
    allow_clarification: bool = False
    confirmed_plan: dict[str, Any] | None = None
    confirmed_candidates: bool = False

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        return value.strip()

    @field_validator("candidates")
    @classmethod
    def normalize_candidates(cls, value: list[str]) -> list[str]:
        normalized = []
        for candidate in value:
            candidate_name = candidate.strip()
            if len(candidate_name) > 120:
                raise ValueError("单个候选项目名称不能超过 120 个字符。")
            if candidate_name and candidate_name not in normalized:
                normalized.append(candidate_name)
        return normalized

    @field_validator("confirmed_plan")
    @classmethod
    def limit_confirmed_plan(cls, value: dict[str, Any] | None):
        if value is not None and len(json.dumps(value, ensure_ascii=False)) > 100_000:
            raise ValueError("确认计划不能超过 100000 个字符。")
        return value


class CandidateSuggestionRequest(BaseModel):
    """Input for the candidate preview stage."""

    question: str = Field(min_length=10, max_length=5000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        return value.strip()


class TaskResumeRequest(BaseModel):
    """Optional answer used to resume an interrupted task."""

    response: str | list[str] | dict[str, Any] | None = None
