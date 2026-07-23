"""Reviewer Agent — 证据汇总、项目评分、冲突检测和报告生成。

Reviewer 是技术选型流程的最后一个核心 Agent，负责：
1. 汇总所有 Agent 返回的证据
2. 对每个候选项目进行多维度评分
3. 检查证据充分性和来源可信度
4. 识别不同来源之间的矛盾
5. 生成结构化的技术选型报告
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from project_advisor.configuration import Configuration
from project_advisor.prompts import reviewer_system_prompt
from project_advisor.schemas.evidence import Evidence, ProjectScore, ReviewResult
from project_advisor.state import AgentState
from project_advisor.tools.citations import detect_conflicts
from project_advisor.tools.scoring import (
    compare_projects,
    create_criteria_from_config,
    format_score_table,
)
from project_advisor.utils import (
    create_chat_model,
    get_message_token_usage,
    get_today_str,
)


def _normalize_evidences(values: list) -> list[Evidence]:
    """Validate and de-duplicate evidence carried through graph state."""
    evidences: list[Evidence] = []
    seen: set[str] = set()
    for value in values:
        try:
            evidence = value if isinstance(value, Evidence) else Evidence.model_validate(value)
        except (TypeError, ValueError):
            continue
        if evidence.evidence_id not in seen:
            seen.add(evidence.evidence_id)
            evidences.append(evidence)
    return evidences


def _evidence_for_project(project_name: str, evidences: list[Evidence]) -> list[Evidence]:
    target = project_name.casefold().strip()
    return [
        evidence
        for evidence in evidences
        if target == evidence.project_name.casefold().strip()
        or target in evidence.project_name.casefold()
    ]


def _bind_scores_to_evidence(
    scores: list[ProjectScore],
    candidates: list[str],
    evidences: list[Evidence],
) -> tuple[list[ProjectScore], list[str]]:
    """Force one score per confirmed candidate and only attach validated references."""
    score_map = {score.project_name.casefold(): score for score in scores}
    bound_scores: list[ProjectScore] = []
    gaps: list[str] = []

    for candidate in candidates:
        score = score_map.get(candidate.casefold())
        if score is None:
            score = ProjectScore(
                project_name=candidate,
                justification="Reviewer 未返回该候选项目的结构化评分。",
            )
            gaps.append(f"{candidate} 缺少结构化评分。")

        matched = _evidence_for_project(candidate, evidences)
        evidence_ids = [evidence.evidence_id for evidence in matched]
        source_urls = list(dict.fromkeys(evidence.source_url for evidence in matched))
        source_types = {evidence.source_type for evidence in matched}
        if len(matched) >= 3 and len(source_types) >= 2:
            confidence = "high"
        elif len(matched) >= 2:
            confidence = "medium"
        elif matched:
            confidence = "low"
        else:
            confidence = "insufficient"
            gaps.append(f"{candidate} 没有可追溯到该项目的结构化证据。")

        bound_scores.append(
            score.model_copy(
                update={
                    "project_name": candidate,
                    "evidence_ids": evidence_ids,
                    "source_urls": source_urls,
                    "evidence_confidence": confidence,
                }
            )
        )
    return bound_scores, gaps


async def review_and_score(state: AgentState, config: RunnableConfig):
    """Run one tool-less structured review over canonical Evidence only."""
    candidates = state.get("candidates", [])
    research_brief = state.get("research_brief", "")
    evidences = _normalize_evidences(state.get("evidences", []))
    evidence_payload = [
        {
            **evidence.model_dump(exclude={"content"}),
            "content": evidence.content[:1000],
        }
        for evidence in evidences[:60]
    ]
    workflow_gaps = []
    for value in state.get("evidence_gaps", []):
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if isinstance(value, dict):
            workflow_gaps.append(
                f"{value.get('project_name', '未知项目')}/"
                f"{value.get('track', 'unknown')}: {value.get('reason', '')}"
            )

    configurable = Configuration.from_runnable_config(config)

    # 使用 Reviewer 模型进行评分分析
    reviewer_prompt = reviewer_system_prompt.format(date=get_today_str())
    review_prompt = f"""基于以下研究简报和结构化证据，分析每个候选项目并给出评分建议。

<研究简报>
{research_brief}
</研究简报>

<候选项目>
{', '.join(candidates) if candidates else '待确定'}
</候选项目>

<结构化证据>
{json.dumps(evidence_payload, ensure_ascii=False, indent=2) if evidence_payload else '暂无结构化证据'}
</结构化证据>

<工作流检测到的证据缺口>
{json.dumps(workflow_gaps, ensure_ascii=False) if workflow_gaps else '无'}
</工作流检测到的证据缺口>

请对每个候选项目分析其优势、劣势和风险。提供评分建议（1-10 分/每维度）。
评分只能依据上面的结构化证据，不得编造 evidence_id 或来源 URL。
证据不足时应保守评分，并明确指出证据缺口和来源冲突。"""

    reviewer = (
        create_chat_model(
            configurable.final_report_model,
            max_tokens=configurable.final_report_model_max_tokens,
        )
        .with_structured_output(ReviewResult, method="function_calling")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    response = await reviewer.ainvoke([
        SystemMessage(content=reviewer_prompt),
        HumanMessage(content=review_prompt),
    ])

    # 构建评估权重
    weights = {
        "weight_feature_match": configurable.weight_feature_match,
        "weight_engineering_reliability": configurable.weight_engineering_reliability,
        "weight_community": configurable.weight_community,
        "weight_documentation": configurable.weight_documentation,
        "weight_learning_cost": configurable.weight_learning_cost,
        "weight_extensibility": configurable.weight_extensibility,
        "weight_deployment_cost": configurable.weight_deployment_cost,
    }
    criteria = create_criteria_from_config(weights)
    bound_scores, binding_gaps = _bind_scores_to_evidence(
        response.scores, candidates, evidences
    )
    ranked_scores = compare_projects(bound_scores, criteria)
    all_gaps = list(dict.fromkeys([
        *workflow_gaps,
        *response.evidence_gaps,
        *binding_gaps,
    ]))

    return {
        "evaluation_criteria": criteria,
        "scores": ranked_scores,
        "review_analysis": response.analysis,
        "review_evidence_gaps": all_gaps,
        "token_usage": get_message_token_usage(response),
        "next": "generate_report",
    }


async def generate_report(state: AgentState, config: RunnableConfig):
    """Render the reviewed state as Markdown without another model call."""
    candidates = state.get("candidates", [])
    research_brief = state.get("research_brief", "")
    scores = state.get("scores", [])
    evidences = _normalize_evidences(state.get("evidences", []))
    score_table = format_score_table(scores) if scores else "暂无结构化评分。"
    analysis = state.get("review_analysis", "暂无 Reviewer 分析。")
    gaps = state.get("review_evidence_gaps", [])
    eligible_scores = [
        score
        for score in scores
        if score.evidence_confidence != "insufficient"
    ]
    winner = (
        eligible_scores[0].project_name
        if eligible_scores
        else "证据不足，暂不推荐"
    )
    sections = [
        "# 技术选型评估报告",
        "## 1. 需求与范围",
        research_brief or "未提供研究简报。",
        "## 2. 候选项目",
        "\n".join(f"- {candidate}" for candidate in candidates) or "- 未确定",
        "## 3. Reviewer 分析",
        analysis,
        "## 4. 程序化加权评分",
        score_table,
        "## 5. 推荐结果",
        f"- **当前首选**：{winner}",
        "- 排名由配置权重确定性计算；Reviewer 只提供结构化维度分。",
    ]

    score_map = {score.project_name.casefold(): score for score in scores}
    details = []
    for candidate in candidates:
        score = score_map.get(candidate.casefold())
        if score is None:
            details.append(f"### {candidate}\n- 缺少结构化评分。")
            continue
        details.append(
            f"### {candidate}\n"
            f"- 功能匹配度：{score.feature_match}/10\n"
            f"- 工程可靠性：{score.engineering_reliability}/10\n"
            f"- 社区与维护：{score.community_and_maintenance}/10\n"
            f"- 文档质量：{score.documentation_quality}/10\n"
            f"- 学习成本：{score.learning_cost}/10\n"
            f"- 扩展能力：{score.extensibility}/10\n"
            f"- 部署成本：{score.deployment_cost}/10\n"
            f"- **加权总分：{score.weighted_total}/10**\n"
            f"- 证据置信度：{score.evidence_confidence}\n"
            f"- 评分依据：{score.justification}\n"
            f"- 证据 ID：{', '.join(score.evidence_ids) or '无'}"
        )
    sections.extend(["## 6. 候选项目详情", "\n\n".join(details) or "暂无详情。"])
    sections.extend([
        "## 7. 证据缺口",
        "\n".join(f"- {gap}" for gap in gaps) or "- 未检测到明确缺口。",
    ])

    if evidences:
        conflicts = detect_conflicts(evidences)
        if conflicts:
            conflict_text = "\n".join([
                f"- **{c['type']}**：{c.get('detail', '')}" for c in conflicts
            ])
        else:
            conflict_text = "- 未检测到来源冲突。"
        sections.extend(["## 8. 来源冲突检测", conflict_text])

    source_lines = [
        f"- `{evidence.evidence_id}` [{evidence.project_name}] "
        f"{evidence.source_url}（{evidence.source_type}，{evidence.retrieved_at}）"
        for evidence in evidences
    ]
    sections.extend([
        "## 9. 信息来源",
        "\n".join(source_lines) or "- 没有可追溯来源。",
    ])
    report_content = "\n\n".join(sections)

    return {
        "final_report": report_content,
        "messages": [SystemMessage(content=report_content)],
    }
