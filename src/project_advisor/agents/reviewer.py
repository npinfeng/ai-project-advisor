"""Reviewer Agent — 证据汇总、项目评分、冲突检测和报告生成。

Reviewer 是技术选型流程的最后一个核心 Agent，负责：
1. 汇总所有 Agent 返回的证据
2. 对每个候选项目进行多维度评分
3. 检查证据充分性和来源可信度
4. 识别不同来源之间的矛盾
5. 生成结构化的技术选型报告
"""

from langchain_core.messages import HumanMessage, SystemMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig

from project_advisor.configuration import Configuration
from project_advisor.prompts import final_report_template, reviewer_system_prompt
from project_advisor.schemas.evidence import ReviewResult
from project_advisor.state import AgentState
from project_advisor.tools.citations import detect_conflicts
from project_advisor.tools.scoring import (
    compare_projects,
    create_criteria_from_config,
    format_score_table,
)
from project_advisor.utils import create_chat_model, get_today_str


async def review_and_score(state: AgentState, config: RunnableConfig):
    """汇总研究发现并对候选项目进行评分。

    此节点：
    1. 汇总 Supervisor 收集的所有研究笔记
    2. 使用 Reviewer LLM 对每个项目进行多维度评分
    3. 计算加权总分并排序
    """
    notes = state.get("notes", [])
    findings = "\n\n".join(notes) if notes else "暂无研究发现。"
    candidates = state.get("candidates", [])
    research_brief = state.get("research_brief", "")

    configurable = Configuration.from_runnable_config(config)

    # 使用 Reviewer 模型进行评分分析
    reviewer_prompt = reviewer_system_prompt.format(date=get_today_str())
    review_prompt = f"""基于以下研究简报和发现，分析每个候选项目并给出评分建议。

<研究简报>
{research_brief}
</研究简报>

<候选项目>
{', '.join(candidates) if candidates else '待确定'}
</候选项目>

<研究发现>
{findings}
</研究发现>

请对每个候选项目分析其优势、劣势和风险。提供评分建议（1-10 分/每维度）。
同时指出证据不足的地方和来源冲突之处。"""

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
    ranked_scores = compare_projects(response.scores, criteria)
    score_table = format_score_table(ranked_scores)
    evidence_gaps = "\n".join(f"- {gap}" for gap in response.evidence_gaps)
    review_note = f"=== Reviewer 分析 ===\n{response.analysis}\n\n{score_table}"
    if evidence_gaps:
        review_note += f"\n\n=== 证据缺口 ===\n{evidence_gaps}"

    return {
        "evaluation_criteria": criteria,
        "scores": ranked_scores,
        "notes": [review_note],
        "next": "generate_report",
    }


async def generate_report(state: AgentState, config: RunnableConfig):
    """生成最终技术选型报告。

    使用 Reviewer 分析结果 + 所有研究发现，
    按照技术选型报告模板生成结构化的 Markdown 报告。
    """
    configurable = Configuration.from_runnable_config(config)
    notes = state.get("notes", [])
    findings = "\n\n".join(notes) if notes else "暂无研究发现。"
    candidates = state.get("candidates", [])
    research_brief = state.get("research_brief", "")
    scores = state.get("scores", [])
    score_table = format_score_table(scores) if scores else "暂无结构化评分。"
    findings = f"{findings}\n\n=== 程序计算的结构化评分 ===\n{score_table}"

    # 从笔记数据构建候选项目信息
    candidates_data = _extract_candidates_from_notes(notes, candidates)

    report_model = create_chat_model(
        configurable.final_report_model,
        max_tokens=configurable.final_report_model_max_tokens,
    )

    # 构建报告参数
    if candidates:
        candidate_names = " | ".join(candidates)
        separator = "|".join(["---------"] * (len(candidates) + 1))

        score_map = {score.project_name.lower(): score for score in scores}
        project_sections = []
        for name in candidates:
            score = score_map.get(name.lower())
            if score:
                project_sections.append(
                    f"### 5.{candidates.index(name) + 1} {name}\n"
                    f"- **功能匹配度**：{score.feature_match}/10\n"
                    f"- **工程可靠性**：{score.engineering_reliability}/10\n"
                    f"- **社区与维护状态**：{score.community_and_maintenance}/10\n"
                    f"- **文档与示例质量**：{score.documentation_quality}/10\n"
                    f"- **学习成本**：{score.learning_cost}/10\n"
                    f"- **扩展能力**：{score.extensibility}/10\n"
                    f"- **部署和运行成本**：{score.deployment_cost}/10\n"
                    f"- **加权总分**：{score.weighted_total}/10\n"
                    f"- **评分依据**：{score.justification}\n"
                )
            else:
                project_sections.append(
                    f"### 5.{candidates.index(name) + 1} {name}\n"
                    "该项目缺少足够证据，暂未生成结构化评分。\n"
                )
    else:
        candidate_names = "项目A | 项目B | 项目C"
        separator = "|---------|---------|---------|"
        project_sections = ["（待确定候选项目）"]

    project_sections_str = "\n".join(project_sections) if isinstance(project_sections, list) else project_sections

    report_prompt = final_report_template.format(
        research_brief=research_brief,
        findings=findings,
        candidates_data=candidates_data,
        candidate_names=candidate_names,
        separator=separator,
        project_sections=project_sections_str,
    )

    try:
        final_report = await report_model.ainvoke([
            HumanMessage(content=report_prompt),
        ])
        report_content = str(final_report.content)
    except Exception as e:
        report_content = f"报告生成失败：{str(e)}\n\n=== 原始研究发现 ===\n\n{findings}"

    # 检查冲突
    from project_advisor.schemas.evidence import Evidence
    evidences = state.get("evidences", [])
    if isinstance(evidences, list) and evidences:
        conflicts = detect_conflicts([
            e for e in evidences if isinstance(e, Evidence)
        ])
        if conflicts:
            conflict_text = "\n".join([
                f"- **{c['type']}**：{c.get('detail', '')}" for c in conflicts
            ])
            report_content += f"\n\n## 12. 来源冲突检测\n\n{conflict_text}"

    return {
        "final_report": report_content,
        "messages": [SystemMessage(content=report_content)],
    }


def _extract_candidates_from_notes(notes: list[str], candidates: list[str]) -> str:
    """从研究笔记中提取候选项目相关数据。"""
    if not notes:
        return "暂无候选项目数据。"

    relevant = []
    for note in notes:
        for candidate in candidates:
            if candidate.lower() in note.lower():
                # 截取相关片段
                idx = note.lower().find(candidate.lower())
                start = max(0, idx - 100)
                end = min(len(note), idx + 500)
                relevant.append(f"[关于 {candidate}] ...{note[start:end]}...")
                break

    if relevant:
        return "\n\n".join(relevant)
    return "\n\n".join(notes[:3])  # 如果找不到匹配，返回前几条笔记
