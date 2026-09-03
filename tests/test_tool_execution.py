"""Regression tests for the bounded Research Agent tool execution policy."""

import asyncio
import importlib

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from project_advisor.configuration import Configuration
from project_advisor.tools.execution import execute_tool
from project_advisor.utils import invoke_structured_with_retry


def test_tool_execution_retries_transient_failure_and_records_success():
    calls = 0

    @tool
    async def flaky_lookup(query: str) -> str:
        """Return a lookup result after one transient outage."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("HTTP 503 temporarily unavailable")
        return f"result:{query}"

    result = asyncio.run(execute_tool(
        flaky_lookup,
        {"query": "LangGraph"},
        {"configurable": {
            "thread_id": "run-1",
            "tool_max_retries": 1,
            "tool_retry_backoff_seconds": 0,
            "tool_timeout_seconds": 1,
        }},
        call_id="call-1",
        project_name="LangGraph",
        research_topic="可靠性",
    ))

    assert result.observation == "result:LangGraph"
    assert result.record.status == "succeeded"
    assert result.record.agent_run_id == "run-1"
    assert result.record.retry_count == 1
    assert calls == 2


def test_tool_execution_timeout_is_bounded_and_observable():
    @tool
    async def slow_lookup(query: str) -> str:
        """Simulate a tool that never finishes within its budget."""
        await asyncio.sleep(0.1)
        return query

    result = asyncio.run(execute_tool(
        slow_lookup,
        {"query": "LangGraph"},
        {"configurable": {
            "tool_max_retries": 0,
            "tool_timeout_seconds": 0.01,
        }},
        call_id="call-timeout",
        project_name="LangGraph",
        research_topic="可靠性",
    ))

    assert result.record.status == "timed_out"
    assert result.record.error_type in {"TimeoutError", "CancelledError"}
    assert result.evidences == []
    assert "工具执行失败" in result.observation


def test_tool_execution_rejects_invalid_arguments_without_invocation():
    calls = 0

    @tool
    async def typed_lookup(limit: int) -> str:
        """Return the validated limit."""
        nonlocal calls
        calls += 1
        return str(limit)

    result = asyncio.run(execute_tool(
        typed_lookup,
        {"limit": "not-an-integer"},
        {"configurable": {"tool_max_retries": 3}},
        call_id="call-invalid",
        project_name="LangGraph",
        research_topic="可靠性",
    ))

    assert result.record.status == "invalid_arguments"
    assert result.record.retry_count == 0
    assert calls == 0


def test_tool_reported_error_is_not_misclassified_as_success():
    @tool
    async def missing_repository(github_url: str) -> str:
        """Return the same terminal observation as the GitHub tool."""
        return "仓库不存在：example/missing"

    result = asyncio.run(execute_tool(
        missing_repository,
        {"github_url": "https://github.com/example/missing"},
        {"configurable": {"tool_max_retries": 2}},
        call_id="call-missing",
        project_name="Missing",
        research_topic="仓库状态",
    ))

    assert result.record.status == "failed"
    assert result.record.retry_count == 0
    assert result.evidences == []


def test_structured_output_uses_one_total_attempt_budget():
    class FlakyStructuredModel:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("HTTP 503 temporarily unavailable")
            if self.calls == 2:
                return {"parsed": None, "raw": "bad", "parsing_error": ValueError("bad JSON")}
            return {"parsed": {"ok": True}, "raw": "good", "parsing_error": None}

    model = FlakyStructuredModel()
    parsed, raw_responses = asyncio.run(invoke_structured_with_retry(
        model,
        ["prompt"],
        max_attempts=3,
        backoff_seconds=0,
    ))

    assert parsed == {"ok": True}
    assert raw_responses == ["bad", "good"]
    assert model.calls == 3


def test_structured_output_does_not_retry_non_transient_error():
    class InvalidRequestModel:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            raise ValueError("invalid API key")

    model = InvalidRequestModel()
    try:
        asyncio.run(invoke_structured_with_retry(
            model,
            ["prompt"],
            max_attempts=3,
            backoff_seconds=0,
        ))
    except ValueError as error:
        assert "API key" in str(error)
    else:
        raise AssertionError("non-transient failures must propagate")
    assert model.calls == 1


def test_researcher_caps_parallel_tool_fanout(monkeypatch):
    analyst_module = importlib.import_module(
        "project_advisor.agents.repository_analyst"
    )
    calls = 0

    @tool
    async def bounded_lookup(query: str) -> str:
        """Count executions for fan-out protection."""
        nonlocal calls
        calls += 1
        return query

    async def fake_tools(config):
        return [bounded_lookup]

    monkeypatch.setattr(analyst_module, "get_repository_tools", fake_tools)
    tool_calls = [
        {
            "name": "bounded_lookup",
            "args": {"query": str(index)},
            "id": f"call-{index}",
            "type": "tool_call",
        }
        for index in range(3)
    ]
    output = asyncio.run(analyst_module.analyst_tools(
        {
            "researcher_messages": [AIMessage(content="", tool_calls=tool_calls)],
            "project_name": "LangGraph",
            "research_topic": "fan-out",
            "tool_call_iterations": 1,
        },
        {"configurable": {
            "max_tool_calls_per_step": 2,
            "tool_max_retries": 0,
        }},
    ))

    assert calls == 2
    assert len(output["researcher_messages"]) == 3
    assert [record["status"] for record in output["tool_executions"]] == [
        "succeeded",
        "succeeded",
        "rejected",
    ]


def test_configuration_rejects_misconfigured_scoring_weights():
    try:
        Configuration(weight_feature_match=0.5)
    except ValueError as error:
        assert "权重之和必须为 1.0" in str(error)
    else:
        raise AssertionError("invalid score weights must not silently change ranking")
