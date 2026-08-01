"""技术选型与项目评估的数据模型定义。

所有结构化输出和证据模型集中在此模块，供 Agent、工具和 Graph 共同使用。
"""

import hashlib
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ===== 需求模型 =====

class Requirement(BaseModel):
    """单条技术需求约束。"""

    category: str = Field(
        description="Category of the requirement (e.g., language, feature, deployment, team_level).",
    )
    value: str = Field(
        description="The specific value or constraint for this category.",
    )
    priority: str = Field(
        default="required",
        description="Priority level: 'required', 'preferred', or 'optional'.",
    )


class Requirements(BaseModel):
    """用户技术需求的结构化表示。"""

    language: Optional[str] = Field(
        default=None,
        description="Preferred programming language (e.g., Python, TypeScript).",
    )
    required_features: list[str] = Field(
        default_factory=list,
        description="Must-have capabilities (e.g., multi_agent, mcp, checkpoint, human_in_the_loop).",
    )
    preferred_features: list[str] = Field(
        default_factory=list,
        description="Nice-to-have capabilities.",
    )
    deployment: Optional[str] = Field(
        default=None,
        description="Deployment preference: 'self_hosted', 'cloud', 'hybrid'.",
    )
    team_level: Optional[str] = Field(
        default=None,
        description="Team experience level: 'beginner', 'intermediate', 'advanced'.",
    )
    budget_constraints: Optional[str] = Field(
        default=None,
        description="Any budget or resource constraints mentioned by the user.",
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any other important context or constraints.",
    )


# ===== 候选项目模型 =====

class CandidateProject(BaseModel):
    """候选开源项目的统一信息模型。"""

    name: str = Field(description="Project name.")
    github_url: str = Field(description="GitHub repository URL.")
    official_site: Optional[str] = Field(
        default=None, description="Official website URL."
    )
    description: Optional[str] = Field(
        default=None, description="Project description from README or website."
    )
    primary_language: Optional[str] = Field(
        default=None, description="Primary programming language."
    )
    license_type: Optional[str] = Field(
        default=None, description="Open source license type."
    )

    # GitHub 指标
    stars: Optional[int] = Field(default=None, description="GitHub star count.")
    contributors: Optional[int] = Field(
        default=None, description="Number of contributors."
    )
    last_updated: Optional[str] = Field(
        default=None, description="Date of most recent commit."
    )
    release_count: Optional[int] = Field(
        default=None, description="Number of releases."
    )
    latest_release: Optional[str] = Field(
        default=None, description="Latest release version and date."
    )
    open_issues: Optional[int] = Field(
        default=None, description="Number of open issues."
    )
    issue_resolution_time: Optional[str] = Field(
        default=None, description="Average issue resolution time."
    )

    # 维护状态
    is_archived: bool = Field(default=False, description="Whether the repo is archived.")
    is_maintained: Optional[bool] = Field(
        default=None, description="Whether the project appears actively maintained."
    )


class CandidateRecommendation(BaseModel):
    """Planner 推荐的候选项目及其入选理由。"""

    name: str = Field(
        min_length=1,
        max_length=120,
        description="Candidate project name.",
    )
    github_url: Optional[str] = Field(
        default=None,
        description="Canonical GitHub repository URL when known.",
    )
    reason: str = Field(
        min_length=1,
        max_length=2000,
        description="Why this candidate matches the user's requirements.",
    )


# ===== 证据模型 =====

class Evidence(BaseModel):
    """统一证据对象 — 所有研究发现的标准表示。"""

    evidence_id: str = Field(
        default="",
        description="Stable evidence identifier derived from source and content.",
    )
    source_url: str = Field(description="URL of the evidence source.")
    source_type: str = Field(
        description="Source type: 'github', 'official_documentation', 'blog', 'community', 'release_note', 'issue'.",
    )
    project_name: str = Field(description="Which candidate project this evidence relates to.")
    content: str = Field(description="The evidence content or summary.")
    relevance: str = Field(
        description="Which evaluation dimension or requirement this addresses."
    )
    confidence: str = Field(
        default="high",
        description="Confidence level: 'high', 'medium', 'low'.",
    )
    retrieved_at: str = Field(
        description="ISO datetime when this evidence was collected."
    )
    source_date: Optional[str] = Field(
        default=None,
        description=(
            "Publication or last-updated date reported by the source. "
            "Use this, rather than retrieval time, for freshness decisions."
        ),
    )
    version_info: Optional[str] = Field(
        default=None, description="Document or software version at time of retrieval."
    )

    @model_validator(mode="after")
    def populate_evidence_id(self) -> "Evidence":
        """Generate a deterministic ID while preserving explicitly supplied IDs."""
        if not self.evidence_id:
            payload = "\x1f".join(
                [self.source_url, self.project_name, self.relevance, self.content]
            )
            self.evidence_id = f"ev_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"
        return self


# ===== 评估模型 =====

class EvaluationCriteria(BaseModel):
    """评估维度和权重配置。"""

    feature_match: float = Field(
        default=0.30, description="Weight for feature match score."
    )
    engineering_reliability: float = Field(
        default=0.20, description="Weight for engineering reliability score."
    )
    community_and_maintenance: float = Field(
        default=0.15, description="Weight for community and maintenance score."
    )
    documentation_quality: float = Field(
        default=0.10, description="Weight for documentation quality score."
    )
    learning_cost: float = Field(
        default=0.10, description="Weight for learning cost score."
    )
    extensibility: float = Field(
        default=0.10, description="Weight for extensibility score."
    )
    deployment_cost: float = Field(
        default=0.05, description="Weight for deployment and operational cost score."
    )


class ProjectScore(BaseModel):
    """单个候选项目在各维度上的评分。"""

    project_name: str
    feature_match: float = Field(default=0.0, ge=0.0, le=10.0)
    engineering_reliability: float = Field(default=0.0, ge=0.0, le=10.0)
    community_and_maintenance: float = Field(default=0.0, ge=0.0, le=10.0)
    documentation_quality: float = Field(default=0.0, ge=0.0, le=10.0)
    learning_cost: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Learning ease: 10 means easiest/lowest learning cost.",
    )
    extensibility: float = Field(default=0.0, ge=0.0, le=10.0)
    deployment_cost: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Deployment economy: 10 means simplest/lowest operating cost.",
    )
    weighted_total: float = Field(default=0.0, description="Final weighted total score.")
    justification: str = Field(
        default="", description="Brief justification for the scores."
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Validated evidence IDs supporting this project score.",
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description="Validated source URLs supporting this project score.",
    )
    evidence_confidence: str = Field(
        default="low",
        description="Evidence confidence: high, medium, low, or insufficient.",
    )


class ReviewResult(BaseModel):
    """Reviewer 的结构化输出。"""

    analysis: str = Field(description="Overall comparison, strengths, risks, and recommendation rationale.")
    scores: list[ProjectScore] = Field(description="One evidence-based score object for each candidate project.")
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="Claims or dimensions that lack enough reliable evidence.",
    )


def calculate_weighted_score(
    score: ProjectScore, criteria: EvaluationCriteria
) -> float:
    """根据评估权重计算加权总分。"""
    weighted = (
        score.feature_match * criteria.feature_match
        + score.engineering_reliability * criteria.engineering_reliability
        + score.community_and_maintenance * criteria.community_and_maintenance
        + score.documentation_quality * criteria.documentation_quality
        + score.learning_cost * criteria.learning_cost
        + score.extensibility * criteria.extensibility
        + score.deployment_cost * criteria.deployment_cost
    )
    return round(weighted, 2)
