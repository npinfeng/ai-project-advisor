"""引用验证 — 检查来源新鲜度、格式和一致性。"""

from datetime import datetime, timedelta

from project_advisor.schemas.evidence import Evidence


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
        retrieved_date = datetime.fromisoformat(evidence.retrieved_at)
        age = datetime.now() - retrieved_date

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
