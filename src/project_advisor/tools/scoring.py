"""评分引擎 — 项目评分的计算、比较和格式化。"""

from project_advisor.schemas.evidence import (
    EvaluationCriteria,
    ProjectScore,
    calculate_weighted_score,
)


def create_default_criteria() -> EvaluationCriteria:
    """创建默认评估权重。"""
    return EvaluationCriteria()


def create_criteria_from_config(config_weights: dict[str, float]) -> EvaluationCriteria:
    """从配置字典创建评估权重。"""
    return EvaluationCriteria(
        feature_match=config_weights.get("weight_feature_match", 0.30),
        engineering_reliability=config_weights.get("weight_engineering_reliability", 0.20),
        community_and_maintenance=config_weights.get("weight_community", 0.15),
        documentation_quality=config_weights.get("weight_documentation", 0.10),
        learning_cost=config_weights.get("weight_learning_cost", 0.10),
        extensibility=config_weights.get("weight_extensibility", 0.10),
        deployment_cost=config_weights.get("weight_deployment_cost", 0.05),
    )


def compare_projects(
    scores: list[ProjectScore],
    criteria: EvaluationCriteria,
) -> list[ProjectScore]:
    """计算所有项目的加权总分并按分数降序排列。

    Args:
        scores: 项目评分列表
        criteria: 评估权重

    Returns:
        按加权总分降序排列的评分列表
    """
    for score in scores:
        score.weighted_total = calculate_weighted_score(score, criteria)
    return sorted(scores, key=lambda s: s.weighted_total, reverse=True)


def format_score_table(
    scores: list[ProjectScore],
) -> str:
    """将评分列表格式化为 Markdown 表格。

    Args:
        scores: 已排序的项目评分列表

    Returns:
        Markdown 格式的评分对比表
    """
    header = (
        "| 项目 | 功能匹配 | 工程可靠性 | 社区维护 | 文档质量 | "
        "学习成本 | 扩展能力 | 部署成本 | **加权总分** |\n"
        "|------|---------|----------|---------|---------|"
        "--------|--------|--------|------------|"
    )

    rows = []
    for s in scores:
        rows.append(
            f"| {s.project_name} | {s.feature_match} | {s.engineering_reliability} | "
            f"{s.community_and_maintenance} | {s.documentation_quality} | "
            f"{s.learning_cost} | {s.extensibility} | {s.deployment_cost} | "
            f"**{s.weighted_total}** |"
        )

    return header + "\n" + "\n".join(rows)
