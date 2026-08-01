"""Context-local token accounting for model calls nested inside tools."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping


_current_usage: ContextVar[dict[str, int] | None] = ContextVar(
    "project_advisor_current_usage",
    default=None,
)


def message_token_usage(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, Mapping):
        response_metadata = getattr(message, "response_metadata", None)
        if isinstance(response_metadata, Mapping):
            usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    if not isinstance(usage, Mapping):
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(
            usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        ),
        "output_tokens": int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        ),
    }


def record_message_usage(message: Any) -> dict[str, int]:
    usage = message_token_usage(message)
    accumulator = _current_usage.get()
    if accumulator is not None:
        accumulator["input_tokens"] += usage["input_tokens"]
        accumulator["output_tokens"] += usage["output_tokens"]
    return usage


@contextmanager
def usage_scope() -> Iterator[dict[str, int]]:
    accumulator = {"input_tokens": 0, "output_tokens": 0}
    token = _current_usage.set(accumulator)
    try:
        yield accumulator
    finally:
        _current_usage.reset(token)


def add_usage(*values: Mapping[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": sum(int(value.get("input_tokens", 0) or 0) for value in values),
        "output_tokens": sum(int(value.get("output_tokens", 0) or 0) for value in values),
    }
