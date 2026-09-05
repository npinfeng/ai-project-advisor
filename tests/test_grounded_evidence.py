"""Provenance round trips and grounded requirement gates, with no network calls."""

import asyncio
import base64
import importlib
import json
from datetime import datetime, timezone

import httpx
import pytest
from langchain_core.messages import AIMessage

from project_advisor.agents import requirement_verifier as verifier
from project_advisor.agents import reviewer
from project_advisor.agents.reviewer import _bind_scores_to_evidence, generate_report
from project_advisor.graph import evidence_coverage
from project_advisor.rag.document_store import DocumentStore
from project_advisor.rag.knowledge_store import persist_evidences
from project_advisor.schemas.evidence import (
    CandidateRecommendation, DimensionRationale, Evidence, EvidenceCitation, ProjectScore,
    RequirementAssessment, RequirementVerdict, ReviewResult,
)
from project_advisor.tools import document_collector, github, search
from project_advisor.tools.evidence_factory import build_evidences_from_tool_result
from project_advisor.tools.execution import execute_tool
from project_advisor.tools.source_result import SourceDocument, sourced, source_tool


def evidence(content="Acme supports persistent checkpoints.", project="Acme", **kwargs):
    return Evidence(source_url=kwargs.pop("source_url", "https://docs.example.org/checkpoints"),
        project_name=project, content=content, source_type="official_documentation",
        evidence_kind=kwargs.pop("evidence_kind", "primary"), relevance="checkpoint",
        retrieved_at=datetime.now(timezone.utc).isoformat(), **kwargs)


def verdict(source, status="built_in", requirement="checkpoint", **kwargs):
    return RequirementVerdict(project_name=source.project_name, requirement=requirement,
        status=status, citations=[EvidenceCitation(evidence_id=source.evidence_id, quote=source.content)], **kwargs)


def run_tool(tool, args, project="Acme"):
    return asyncio.run(execute_tool(tool, args, {"configurable": {"tool_max_retries": 0}},
        call_id="test-call", project_name=project, research_topic="checkpoint"))


def test_artifact_round_trip_keeps_sources_separate_and_legacy_text_unverified():
    @source_tool(description="Two independent fetched sources.")
    async def lookup(query: str) -> str:
        return sourced("Readable summary with unrelated https://bad.example.org/link", [
            SourceDocument(source_url="https://example.org/a", content="Only document A.", source_date="2026-01-01"),
            SourceDocument(source_url="https://example.org/b", content="Only document B."),
        ])

    result = run_tool(lookup, {"query": "checkpoint"})
    assert result.record.status == "succeeded"
    assert [e.content for e in result.evidences] == ["Only document A.", "Only document B."]
    assert [e.source_url for e in result.evidences] == ["https://example.org/a", "https://example.org/b"]
    assert result.evidences[0].source_date == "2026-01-01"
    assert result.evidences[0].content_hash != result.evidences[1].content_hash
    legacy = build_evidences_from_tool_result(tool_name="batch_fetch_tool", args={},
        result="https://example.org/a A; https://example.org/b B", project_name="Acme", research_topic="x")
    assert len(legacy) == 1
    assert legacy[0].source_url == "tool://batch_fetch_tool"
    assert legacy[0].evidence_kind == "unverified"


def test_github_file_uses_file_url_version_and_body(monkeypatch):
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url, **kwargs):
            assert kwargs["params"] == {"ref": "v2.0"}
            return httpx.Response(200, json={"type": "file", "encoding": "base64",
                "content": base64.b64encode(b"[project]\nname = 'Acme'").decode(),
                "html_url": "https://github.com/acme/demo/blob/v2.0/pyproject.toml"})
    monkeypatch.setattr(github.httpx, "AsyncClient", Client)
    result = run_tool(github.github_get_file, {"github_url": "acme/demo", "path": "pyproject.toml", "ref": "v2.0"})
    assert result.record.status == "succeeded"
    source = result.evidences[0]
    assert source.source_url.endswith("/blob/v2.0/pyproject.toml")
    assert source.version_info == "v2.0"
    assert source.locator == "pyproject.toml"
    assert source.content == "[project]\nname = 'Acme'"
    assert source.evidence_kind == "primary"


def test_web_fetch_redirect_and_batch_preserve_each_full_passage(monkeypatch):
    async def fetch(url):
        return {"status": 200, "url": url + "/final", "title": "Documentation",
            "content": ("Document A. " if url.endswith("/a") else "Document B. ") * 1000,
            "source_date": "2026-02-03"}
    monkeypatch.setattr(document_collector, "fetch_webpage", fetch)
    single = run_tool(document_collector.web_fetch_tool, {"url": "https://example.org/a"})
    assert single.evidences[0].source_url == "https://example.org/a/final"
    batch = run_tool(document_collector.batch_fetch_tool, {"urls": ["https://example.org/a", "https://example.org/b"]})
    assert len(batch.evidences) == 2
    assert len(batch.evidences[0].content) > 500
    assert "Document B" not in batch.evidences[0].content
    assert batch.evidences[0].source_date == "2026-02-03"
    assert batch.evidences[0].truncated is True
    assert batch.evidences[0].content == ("Document A. " * 1000)[:10000]
    assert single.evidences[0].truncated is True


def test_search_distinguishes_snippets_and_discovery_from_fetched_text(monkeypatch):
    async def search_results(*args, **kwargs):
        return [{"results": [
            {"url": "https://example.org/a", "content": "Snippet A"},
            {"url": "https://example.org/b", "content": "Snippet B", "raw_content": "Full B"},
        ]}]
    monkeypatch.setattr(search, "tavily_search_async", search_results)
    result = run_tool(search.tavily_search_tool, {"queries": ["Acme"]})
    assert [e.evidence_kind for e in result.evidences] == ["search_snippet", "primary"]
    assert result.evidences[1].content == "Full B"

    async def fetch(url):
        return {"status": 200, "url": url, "links": [{"url": url + "/api", "text": "API"}]}
    monkeypatch.setattr(document_collector, "fetch_webpage", fetch)
    result = run_tool(document_collector.web_discover_links, {"url": "https://example.org"})
    assert result.evidences[0].evidence_kind == "discovery"


def test_rag_round_trip_does_not_mint_new_identity_date_or_project(monkeypatch, tmp_path):
    rag = importlib.import_module("project_advisor.tools.rag_search")
    source = evidence(project="Original", source_date="2025-12-01")
    persist_evidences([source], storage_dir=tmp_path)
    class Rewriter:
        async def generate_multi_queries(self, query):
            return [query, query + " docs"]
    class Reranker:
        async def rerank(self, query, candidates, top_k):
            return candidates[:top_k]
    class Pipeline:
        def search(self, **kwargs):
            return [{"id": "chunk1", "text": source.content, "score": 1,
                     "metadata": {"evidence_id": source.evidence_id, "source_url": source.source_url}}]
    monkeypatch.setattr(rag, "_sync_from_store", lambda *args, **kwargs: [{"project_name": "Original"}])
    monkeypatch.setattr(rag, "_get_pipeline", lambda *args: Pipeline())
    monkeypatch.setattr(rag, "_get_rewriter", lambda *args: Rewriter())
    monkeypatch.setattr(rag, "_get_reranker", lambda *args: Reranker())
    store_module = importlib.import_module("project_advisor.rag.document_store")
    monkeypatch.setattr(store_module, "DocumentStore", lambda: DocumentStore(str(tmp_path)))
    result = run_tool(rag.rag_search, {"query": "checkpoint"}, project="Different caller")
    assert result.record.status == "succeeded"
    assert result.evidences[0].model_dump() == source.model_dump()
    assert persist_evidences(result.evidences, storage_dir=tmp_path)["stored"] == 0


@pytest.mark.parametrize("kind", ["unverified", "discovery", "search_snippet", "inference"])
def test_non_primary_sources_cannot_pass_hard_constraints(kind):
    source = evidence(evidence_kind=kind)
    checked = verifier.validate_verdicts("Acme", ["checkpoint"], [verdict(source)], [source])
    assert checked[0].status == "unknown"
    assert verifier.eligibility(checked) == "conditional"


def test_citation_validation_blocks_fabrication_cross_candidate_and_stripped_negation():
    source = evidence("Acme does not support persistent checkpoints.")
    valid = EvidenceCitation(evidence_id=source.evidence_id, quote=source.content, source_url="https://forged.example.org")
    checked = verifier.validate_citations([valid], {source.evidence_id: source}, "Acme")
    assert checked[0].source_url == source.source_url
    assert source.content[checked[0].start_char:checked[0].end_char] == source.content
    for quote in ["Acme supports persistent checkpoints.", "support persistent checkpoints."]:
        citation = valid.model_copy(update={"quote": quote})
        assert verifier.validate_citations([citation], {source.evidence_id: source}, "Acme") == []
    assert verifier.validate_citations([valid], {source.evidence_id: source}, "Other") == []


def test_absent_or_keyword_only_verdicts_trigger_supplement_but_not_unsupported():
    source = evidence("This document discusses checkpoint terminology.")
    state = {"candidates": ["Acme"], "requirements": {"required_features": ["checkpoint", "custom unknown requirement"]},
             "evidences": [source], "candidate_recommendations": [], "supplemental_round_used": False}
    first = evidence_coverage(state)
    assert first["next"] == "supplemental_research"
    assert any("custom unknown requirement" in gap.reason for gap in first["evidence_gaps"])
    final = evidence_coverage({**state, "supplemental_round_used": True})
    assert final["next"] == "review_and_score"
    assert verifier.validate_verdicts("Acme", ["checkpoint"], [], [source])[0].status == "unknown"


def test_versions_conflicts_and_missing_citations_fail_closed():
    yes = evidence(version_info="v2")
    no = evidence("Acme does not support persistent checkpoints in v1.", version_info="v1",
                  source_url="https://docs.example.org/v1")
    checked = verifier.validate_verdicts("Acme", ["checkpoint"],
        [verdict(yes, applicable_version="v2"), verdict(no, "unsupported", applicable_version="v1")], [yes, no])
    assert checked[0].status == "conflicting"
    assert verifier.validate_verdicts("Acme", ["checkpoint"], checked, [yes, no])[0].status == "conflicting"
    assert verifier.eligibility(checked) == "conditional"
    fabricated_version = verifier.validate_verdicts("Acme", ["checkpoint"], [verdict(yes, applicable_version="v99")], [yes])
    assert fabricated_version[0].status == "unknown"
    missing = verdict(yes).model_copy(update={"citations": []})
    assert verifier.validate_verdicts("Acme", ["checkpoint"], [missing], [yes])[0].status == "unknown"


def test_semantic_node_usage_and_unknown_fallback(monkeypatch):
    source = evidence("Acme does not support persistent checkpoints.")
    class Model:
        def with_structured_output(self, *args, **kwargs):
            return self
        async def ainvoke(self, messages):
            return {"parsed": RequirementAssessment(verdicts=[verdict(source, "unsupported")]),
                    "raw": AIMessage(content="", usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20})}
    monkeypatch.setattr(verifier, "create_chat_model", lambda *args, **kwargs: Model())
    state = {"candidates": ["Acme"], "requirements": {"required_features": ["checkpoint", "rag"]}, "evidences": [source]}
    result = asyncio.run(verifier.verify_requirements(state, {}))
    assert [v.status for v in result["requirement_verdicts"]] == ["unsupported", "unknown"]
    assert result["token_usage"] == {"input_tokens": 12, "output_tokens": 8}
    def fail(*args, **kwargs):
        raise ValueError("Model unavailable")
    monkeypatch.setattr(verifier, "create_chat_model", fail)
    result = asyncio.run(verifier.verify_requirements(state, {}))
    assert all(v.status == "unknown" for v in result["requirement_verdicts"])


def test_dimension_binding_only_attaches_cited_sources_and_caps_unsupported_scores():
    source = evidence()
    irrelevant = evidence("Acme has documentation about other things.", source_url="https://example.org/other")
    score = ProjectScore(project_name="Acme", feature_match=9, extensibility=9,
        evidence_ids=[irrelevant.evidence_id], dimension_rationales=[DimensionRationale(
            dimension="feature_match", reason="Direct checkpoint evidence.",
            citations=[EvidenceCitation(evidence_id=source.evidence_id, quote=source.content)])])
    scores, _ = _bind_scores_to_evidence([score], ["Acme"], [source, irrelevant], ["checkpoint"], [verdict(source)])
    assert scores[0].feature_match == 9
    assert scores[0].extensibility == 5
    assert scores[0].evidence_ids == [source.evidence_id]
    assert scores[0].source_urls == [source.source_url]
    assert scores[0].eligibility == "eligible"


def test_high_score_cannot_override_hard_constraint_and_unknown_is_conditional():
    no = evidence("Acme does not support persistent checkpoints.")
    yes = evidence("Beta supports persistent checkpoints.", project="Beta", source_url="https://example.org/beta")
    scores = [ProjectScore(project_name="Acme", weighted_total=9.9, evidence_confidence="high", eligibility="eligible"),
              ProjectScore(project_name="Beta", weighted_total=7, evidence_confidence="high")]
    state = {"candidates": ["Acme", "Beta"], "requirements": {"required_features": ["checkpoint"]},
        "scores": scores, "evidences": [no, yes], "requirement_verdicts": [verdict(no, "unsupported"), verdict(yes)]}
    report = asyncio.run(generate_report(state, {}))["final_report"]
    assert "**当前首选**：Beta" in report
    assert scores[0].eligibility == "excluded"
    assert "Acme does not support" in report
    report = asyncio.run(generate_report({**state, "requirement_verdicts": []}, {}))["final_report"]
    assert "**当前首选**：Beta" not in report
    assert all(s.eligibility == "conditional" for s in scores)


def test_mixed_valid_and_fabricated_citations_invalidate_entire_rationale():
    source = evidence()
    citations = [EvidenceCitation(evidence_id=source.evidence_id, quote=source.content),
                 EvidenceCitation(evidence_id="fabricated", quote="This source does not exist.")]
    score = ProjectScore(project_name="Acme", feature_match=10, dimension_rationales=[
        DimensionRationale(dimension="feature_match", reason="Mixed sources", citations=citations)])
    bound, _ = _bind_scores_to_evidence([score], ["Acme"], [source])
    assert bound[0].feature_match == 5
    assert bound[0].dimension_rationales == []
    assert bound[0].evidence_ids == []


def test_original_id_ignores_research_topic_but_distinguishes_version():
    source = evidence(version_info="v1")
    another_topic = Evidence.model_validate({**source.model_dump(), "evidence_id": "", "relevance": "another topic"})
    another_version = Evidence.model_validate({**source.model_dump(), "evidence_id": "", "version_info": "v2"})
    assert another_topic.evidence_id == source.evidence_id
    assert another_version.evidence_id != source.evidence_id


def test_verification_review_report_chain_preserves_gates_and_rejects_freeform_claims(monkeypatch):
    sources = {
        "Acme": evidence("Acme does not support persistent checkpoints."),
        "Beta": evidence("Beta supports persistent checkpoints.", project="Beta",
                         source_url="https://docs.example.org/beta"),
    }
    calls = []

    class Model:
        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            return self

        async def ainvoke(self, messages):
            prompt = messages[-1].content
            if self.schema is RequirementAssessment:
                name = json.loads(prompt)["candidate"]
                result = RequirementAssessment(verdicts=[verdict(
                    sources[name], "unsupported" if name == "Acme" else "built_in")])
            else:
                name = "Acme" if "<候选项目>Acme</候选项目>" in prompt else "Beta"
                source = sources[name]
                own_score = ProjectScore(project_name=name, feature_match=10 if name == "Acme" else 7,
                    dimension_rationales=[DimensionRationale(dimension="feature_match",
                        reason="依据引用评估 checkpoint。", citations=[
                            EvidenceCitation(evidence_id=source.evidence_id, quote=source.content)])])
                result = ReviewResult(analysis="UNVERIFIED_FREEFORM_RECOMMENDATION", scores=[
                    own_score, ProjectScore(project_name="Injected candidate", feature_match=10)])
            calls.append((self.schema.__name__, name))
            return {"parsed": result, "raw": AIMessage(content="", usage_metadata={
                "input_tokens": 12, "output_tokens": 8, "total_tokens": 20})}

    monkeypatch.setattr(verifier, "create_chat_model", lambda *a, **k: Model())
    monkeypatch.setattr(reviewer, "create_chat_model", lambda *a, **k: Model())
    state = {"candidates": list(sources), "requirements": {"required_features": ["checkpoint"]},
             "evidences": list(sources.values()), "candidate_recommendations": [
                 CandidateRecommendation(name=name, reason="test") for name in sources]}
    checked = asyncio.run(verifier.verify_requirements(state, {}))
    # Exercise serialization as used by persisted state, not just Python objects.
    state["requirement_verdicts"] = [json.loads(v.model_dump_json()) for v in checked["requirement_verdicts"]]
    coverage = evidence_coverage(state)
    assert coverage["next"] == "review_and_score"
    reviewed = asyncio.run(reviewer.review_and_score(state, {}))
    assert [s.project_name for s in reviewed["scores"]] == ["Acme", "Beta"]
    assert [s.eligibility for s in reviewed["scores"]] == ["excluded", "eligible"]
    assert reviewed["token_usage"] == {"input_tokens": 24, "output_tokens": 16}
    report = asyncio.run(generate_report({**state, **reviewed}, {}))["final_report"]
    assert "**当前首选**：Beta" in report
    assert "UNVERIFIED_FREEFORM_RECOMMENDATION" not in report
    assert "Injected candidate" not in report
    assert len(calls) == 4  # two bounded verifications and two independent reviews
