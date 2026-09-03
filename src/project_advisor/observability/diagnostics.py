"""Provider-neutral token, latency, citation and cost diagnostics."""

from __future__ import annotations

import re
import time
from typing import Any, Mapping

from project_advisor.configuration import Configuration
from project_advisor.mcp_client import get_mcp_diagnostics


URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}]+")


def count_citation_urls(report: str) -> int:
    return len({url.rstrip(".,;:!?，。；：！？\"'") for url in URL_PATTERN.findall(report)})


def usage_values(usage: Mapping[str, Any]) -> tuple[int, int]:
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    return int(input_tokens or 0), int(output_tokens or 0)


def collect_token_usage(
    value: Any,
    seen_objects: set[int] | None = None,
) -> tuple[int, int]:
    """Collect LangChain/OpenAI usage metadata from one graph update."""
    seen_objects = seen_objects if seen_objects is not None else set()
    object_id = id(value)
    if object_id in seen_objects:
        return 0, 0
    seen_objects.add(object_id)

    if isinstance(value, Mapping):
        direct_usage = value.get("usage_metadata") or value.get("token_usage")
        if isinstance(direct_usage, Mapping):
            return usage_values(direct_usage)
        totals = [collect_token_usage(child, seen_objects) for child in value.values()]
        return sum(item[0] for item in totals), sum(item[1] for item in totals)

    if isinstance(value, (list, tuple, set)):
        totals = [collect_token_usage(child, seen_objects) for child in value]
        return sum(item[0] for item in totals), sum(item[1] for item in totals)

    usage_metadata = getattr(value, "usage_metadata", None)
    if isinstance(usage_metadata, Mapping):
        return usage_values(usage_metadata)
    response_metadata = getattr(value, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if isinstance(token_usage, Mapping):
            return usage_values(token_usage)
    return 0, 0


def build_runtime_diagnostics(
    *,
    started_at: float,
    stage_durations_ms: dict[str, int],
    candidate_count: int,
    report: str,
    input_tokens: int,
    output_tokens: int,
    tool_execution: dict[str, int],
    config: Configuration,
    context_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    estimated_cost = (
        input_tokens * config.input_price_per_million
        + output_tokens * config.output_price_per_million
    ) / 1_000_000
    return {
        "total_duration_ms": round((time.perf_counter() - started_at) * 1000),
        "stage_durations_ms": stage_durations_ms,
        "candidate_count": candidate_count,
        "citation_url_count": count_citation_urls(report),
        "mcp": get_mcp_diagnostics(config),
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "collected": (input_tokens + output_tokens) > 0,
        },
        "tool_execution": dict(tool_execution),
        "context_budget": dict(context_budget or {}),
        "estimated_cost_usd": round(estimated_cost, 6),
        "cost_configured": (
            config.input_price_per_million > 0
            or config.output_price_per_million > 0
        ),
        "budget": {
            "max_run_tokens": config.max_run_tokens,
            "max_run_cost_usd": config.max_run_cost_usd,
            "token_limit_enabled": config.max_run_tokens > 0,
            "cost_limit_enabled": config.max_run_cost_usd > 0,
            "cost_limit_enforceable": (
                config.max_run_cost_usd > 0
                and (
                    config.input_price_per_million > 0
                    or config.output_price_per_million > 0
                )
            ),
        },
    }
