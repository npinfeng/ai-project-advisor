"""引用验证 — 检查来源新鲜度、格式、权威性和一致性。"""

import math
from datetime import datetime, timedelta, timezone

from project_advisor.schemas.evidence import Evidence

# 来源类型权威性权重（供置信度计算和 Reviewer 参考）
SOURCE_AUTHORITY_WEIGHTS: dict[str, float] = {
    "official_documentation": 1.0,
    "release_note": 1.0,
    "github": 0.9,
    "mcp": 0.8,
    "blog": 0.6,
    "local_rag": 0.6,
    "web_search": 0.4,
    "community": 0.4,
}

# 未识别来源类型的默认权重
DEFAULT_SOURCE_AUTHORITY = 0.3

# 新鲜度半衰期（天）
FRESHNESS_HALF_LIFE_DAYS = 90


def get_source_authority(source_type: str) -> float:
    """根据来源类型返回权威性权重。

    Args:
        source_type: 来源类型标识

    Returns:
        0.0~1.0 的权威性权重
    """
    if not source_type:
        return DEFAULT_SOURCE_AUTHORITY
    normalized = source_type.strip().lower()
    for key, weight in SOURCE_AUTHORITY_WEIGHTS.items():
        if key in normalized or normalized in key:
            return weight
    return DEFAULT_SOURCE_AUTHORITY


def get_evidence_quality_score(
    evidence: Evidence,
    current_date: datetime | None = None,
) -> float:
    """计算单条证据的综合质量分数（权威性 × 新鲜度）。

    公式：quality = authority_weight × freshness_decay

    Args:
        evidence: 证据对象
        current_date: 参考日期（默认当前 UTC 时间）

    Returns:
        0.0~1.0 的质量分数
    """
    if current_date is None:
        current_date = datetime.now(timezone.utc)

    # 权威性得分
    authority = get_source_authority(evidence.source_type)

    # 新鲜度必须优先使用来源自身的发布日期/更新时间。抓取时间只能说明
    # "我们何时看过"，不能证明内容是新的。来源日期缺失时采用保守中性值。
    freshness = 0.7
    freshness_date = evidence.source_date
    if freshness_date:
        try:
            source_dt = datetime.fromisoformat(freshness_date.replace("Z", "+00:00"))
            if source_dt.tzinfo is None:
                source_dt = source_dt.replace(tzinfo=timezone.utc)
            age_days = (current_date - source_dt).total_seconds() / 86400.0
            if age_days > 0:
                freshness = math.pow(0.5, age_days / FRESHNESS_HALF_LIFE_DAYS)
        except (ValueError, TypeError):
            freshness = 0.5

    return round(authority * freshness, 4)


def compute_evidence_confidence(
    evidences: list[Evidence],
    current_date: datetime | None = None,
) -> tuple[str, float]:
    """基于证据质量和多样性计算置信度（替代纯数量逻辑）。

    算法：
    1. 计算每条证据的质量分数
    2. 按质量分加权求和，而非简单计数
    3. 多样性加分：不同来源类型越多，置信度越高
    4. 综合得出 high/medium/low/insufficient

    Args:
        evidences: 与某个候选项目相关的证据列表
        current_date: 参考日期

    Returns:
        (confidence_label, confidence_score) 元组
    """
    if not evidences:
        return "insufficient", 0.0

    if current_date is None:
        current_date = datetime.now(timezone.utc)

    # 计算质量加权总分
    quality_scores = []
    source_types: set[str] = set()
    for evidence in evidences:
        score = get_evidence_quality_score(evidence, current_date)
        quality_scores.append(score)
        if evidence.source_type:
            source_types.add(evidence.source_type.strip().lower())

    total_quality = sum(quality_scores)
    avg_quality = total_quality / len(quality_scores)
    diversity_bonus = min(len(source_types) * 0.15, 0.45)

    # 综合置信度分数（0.0~1.0 范围）
    confidence_score = min(avg_quality + diversity_bonus, 1.0)

    # 根据分数映射到标签
    if confidence_score >= 0.7:
        label = "high"
    elif confidence_score >= 0.4:
        label = "medium"
    elif confidence_score >= 0.15:
        label = "low"
    else:
        label = "insufficient"

    return label, round(confidence_score, 4)


def check_source_freshness(
    evidence: Evidence,
    max_age_days: int = 180,
) -> dict:
    """检查证据来源的新鲜度。

    Args:
        evidence: 证据对象
        max_age_days: 最大允许天数，超过则标记为过时

    Returns:
        包含新鲜度评估结果的字典
    """
    try:
        if not evidence.source_date:
            return {
                "is_fresh": None,
                "age_days": None,
                "warning": "来源未提供发布日期或更新时间，不能用抓取时间推断新鲜度。",
            }
        source_date = datetime.fromisoformat(
            evidence.source_date.replace("Z", "+00:00")
        )
        now = datetime.now(source_date.tzinfo) if source_date.tzinfo else datetime.now()
        age = now - source_date

        if age > timedelta(days=max_age_days):
            return {
                "is_fresh": False,
                "age_days": age.days,
                "warning": f"证据已过 {age.days} 天，超过 {max_age_days} 天阈值。",
            }
        return {
            "is_fresh": True,
            "age_days": age.days,
            "warning": None,
        }
    except (ValueError, TypeError):
        return {
            "is_fresh": True,
            "age_days": None,
            "warning": "无法解析检索日期，跳过新鲜度检查。",
        }


def validate_citation(
    evidence: Evidence,
    claim: str,
) -> dict:
    """基础引用验证 — 检查引用是否包含必要字段。

    Args:
        evidence: 证据对象
        claim: 使用该证据支撑的声明

    Returns:
        验证结果字典
    """
    issues = []

    if not evidence.source_url:
        issues.append("缺少来源 URL")
    if not evidence.content:
        issues.append("证据内容为空")
    if not evidence.retrieved_at:
        issues.append("缺少检索时间")
    if not evidence.project_name:
        issues.append("缺少项目名称")

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "claim": claim[:200],
        "source_url": evidence.source_url,
    }


def detect_conflicts(
    evidences: list[Evidence],
) -> list[dict]:
    """检测不同证据之间的矛盾。

    简化实现：按项目名和相关性分组，检查同一话题下是否有矛盾声明。

    Args:
        evidences: 证据列表

    Returns:
        冲突列表
    """
    conflicts = []
    # 按 (project_name, relevance) 分组
    groups: dict[tuple[str, str], list[Evidence]] = {}
    for e in evidences:
        key = (e.project_name, e.relevance)
        groups.setdefault(key, []).append(e)

    for (project, relevance), group in groups.items():
        # 检查版本信息不一致
        versions = {e.version_info for e in group if e.version_info}
        if len(versions) > 1:
            conflicts.append({
                "type": "version_mismatch",
                "project": project,
                "topic": relevance,
                "versions": list(versions),
                "detail": f"项目 {project} 在 {relevance} 方面存在多个版本引用：{versions}",
            })

        # 检查同一来源的新旧证据
        urls = {}
        for e in group:
            if e.source_url in urls:
                conflicts.append({
                    "type": "duplicate_source",
                    "project": project,
                    "url": e.source_url,
                    "detail": f"来源 {e.source_url} 被多次引用（项目 {project}）。",
                })
            urls[e.source_url] = e

    return conflicts
