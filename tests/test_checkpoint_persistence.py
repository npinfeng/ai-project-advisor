"""Persistent checkpoint and task-history regression tests."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from typing import TypedDict

from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from project_advisor.persistence import TaskStore
from project_advisor.schemas.evidence import CandidateRecommendation, Requirements
from project_advisor.state import ResearchPlan


class ApprovalState(TypedDict):
    prompt: str
    answer: str


def test_sqlite_checkpoint_survives_connection_restart(tmp_path):
    async def run():
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        database = tmp_path / "checkpoints.sqlite3"
        builder = StateGraph(ApprovalState)

        def wait_for_answer(state: ApprovalState):
            answer = interrupt({"kind": "clarification", "question": state["prompt"]})
            return {"answer": str(answer)}

        builder.add_node("wait_for_answer", wait_for_answer)
        builder.add_edge(START, "wait_for_answer")
        builder.add_edge("wait_for_answer", END)
        config = {"configurable": {"thread_id": "persistent-thread"}}

        async with aiosqlite.connect(database) as connection:
            saver = AsyncSqliteSaver(connection)
            first_graph = builder.compile(checkpointer=saver)
            first = await first_graph.ainvoke(
                {"prompt": "请补充部署方式", "answer": ""}, config=config
            )
            assert first["__interrupt__"][0].value["kind"] == "clarification"

        async with aiosqlite.connect(database) as connection:
            saver = AsyncSqliteSaver(connection)
            resumed_graph = builder.compile(checkpointer=saver)
            resumed = await resumed_graph.ainvoke(
                Command(resume="私有化部署"), config=config
            )
            assert resumed["answer"] == "私有化部署"

    asyncio.run(run())


def test_task_store_survives_restart(tmp_path):
    async def run():
        database = tmp_path / "tasks.sqlite3"
        first = TaskStore(database)
        await first.setup()
        await first.create(
            task_id="task-1",
            question="比较 LangGraph 和 CrewAI 的持久化能力。",
            candidates=["LangGraph", "CrewAI"],
            allow_clarification=True,
            confirmed_plan=None,
            confirmed_candidates=False,
        )
        await first.update(
            "task-1",
            status="waiting_input",
            pending_interrupt={"kind": "clarification", "question": "部署方式？"},
        )

        reopened = TaskStore(database)
        await reopened.setup()
        record = await reopened.get("task-1")
        assert record is not None
        assert record["status"] == "waiting_input"
        assert record["pending_interrupt"]["question"] == "部署方式？"
        assert (await reopened.list())[0]["task_id"] == "task-1"

    asyncio.run(run())


def test_task_store_recovers_orphaned_running_tasks(tmp_path):
    async def run():
        store = TaskStore(tmp_path / "tasks.sqlite3")
        await store.setup()
        await store.create(
            task_id="orphaned",
            question="比较两个 Agent 框架的生产可靠性。",
            candidates=["LangGraph", "CrewAI"],
            allow_clarification=False,
            confirmed_plan=None,
            confirmed_candidates=True,
        )
        await store.update("orphaned", status="running")

        assert await store.recover_incomplete() == 1
        recovered = await store.get("orphaned")
        assert recovered is not None
        assert recovered["status"] == "paused"
        assert "可恢复" in recovered["error"]
        assert await store.recover_incomplete() == 0

    asyncio.run(run())


def test_task_api_persists_completed_report(monkeypatch, tmp_path):
    app_module = importlib.import_module("project_advisor.app")
    store = TaskStore(tmp_path / "tasks.sqlite3")
    asyncio.run(store.setup())
    monkeypatch.setattr(app_module.app.state, "task_store", store, raising=False)

    class CompletedGraph:
        async def astream(self, *args, **kwargs):
            yield {"clarify_requirements": {"next": "plan_evaluation"}}
            yield {"generate_report": {"final_report": "# 持久化报告"}}

    monkeypatch.setattr(app_module, "graph", CompletedGraph())
    plan = ResearchPlan(
        research_brief="测试任务持久化。",
        requirements=Requirements(language="Python"),
        candidates=[CandidateRecommendation(name="LangGraph", reason="测试")],
        evaluation_focus=["feature_match"],
    )
    client = TestClient(app_module.app)
    with client.stream(
        "POST",
        "/api/advice/stream",
        json={
            "question": "请比较适合 Python 的 Agent 框架。",
            "candidates": ["LangGraph"],
            "confirmed_plan": plan.model_dump(),
            "confirmed_candidates": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    started = next(
        line[6:].strip()
        for line in body.splitlines()
        if line.startswith("data:") and '"task_id"' in line
    )
    task_id = json.loads(started)["task_id"]
    task_response = client.get(f"/api/tasks/{task_id}")

    assert response.status_code == 200
    assert task_response.status_code == 200
    assert task_response.json()["status"] == "completed"
    assert task_response.json()["report"] == "# 持久化报告"


def test_task_api_interrupts_and_resumes_same_task(monkeypatch, tmp_path):
    app_module = importlib.import_module("project_advisor.app")
    store = TaskStore(tmp_path / "tasks.sqlite3")
    asyncio.run(store.setup())
    monkeypatch.setattr(app_module.app.state, "task_store", store, raising=False)

    class InterruptingGraph:
        async def astream(self, graph_input, *args, **kwargs):
            if isinstance(graph_input, Command):
                assert graph_input.resume == {"answer": "私有化部署"}
                yield {"generate_report": {"final_report": "# 恢复后的报告"}}
                return
            yield {
                "__interrupt__": [
                    SimpleNamespace(value={
                        "kind": "clarification",
                        "question": "请确认部署方式。",
                    })
                ]
            }

    monkeypatch.setattr(app_module, "graph", InterruptingGraph())
    client = TestClient(app_module.app)
    with client.stream(
        "POST",
        "/api/advice/stream",
        json={
            "question": "请选择适合内网部署的 Agent 框架。",
            "allow_clarification": True,
        },
    ) as response:
        interrupted_body = "".join(response.iter_text())

    started = next(
        line[6:].strip()
        for line in interrupted_body.splitlines()
        if line.startswith("data:") and '"task_id"' in line
    )
    task_id = json.loads(started)["task_id"]
    waiting = client.get(f"/api/tasks/{task_id}").json()
    assert waiting["status"] == "waiting_input"
    assert waiting["pending_interrupt"]["kind"] == "clarification"

    with client.stream(
        "POST",
        f"/api/tasks/{task_id}/resume",
        json={"response": {"answer": "私有化部署"}},
    ) as response:
        resumed_body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "恢复后的报告" in resumed_body
    completed = client.get(f"/api/tasks/{task_id}").json()
    assert completed["status"] == "completed"
    assert completed["report"] == "# 恢复后的报告"


def test_app_lifespan_initializes_sqlite_persistence(monkeypatch, tmp_path):
    app_module = importlib.import_module("project_advisor.app")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite3"))
    monkeypatch.setenv("TASK_DB_PATH", str(tmp_path / "tasks.sqlite3"))
    monkeypatch.setattr(app_module, "graph", None)

    with TestClient(app_module.app) as client:
        payload = client.get("/api/health").json()
        assert payload["persistence"]["status"] == "ready"
        assert payload["persistence"]["checkpoint_enabled"] is True

    assert (tmp_path / "checkpoints.sqlite3").exists()
    assert (tmp_path / "tasks.sqlite3").exists()
