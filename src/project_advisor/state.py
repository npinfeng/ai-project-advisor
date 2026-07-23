"""深度研究 Agent 的图状态定义和数据结构。"""

import operator
from typing import Annotated, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from project_advisor.schemas.evidence import (
    CandidateRecommendation,
    CandidateProject,
    EvaluationCriteria,
    Evidence,
    ProjectScore,
    Requirements,
)


class ResearchComplete(BaseModel):
    """调用此工具表示研究已完成。"""


class ResearchPlan(BaseModel):
    """Planner 生成的研究计划。"""

    research_brief: str = Field(
        description="Detailed research brief describing what needs to be investigated.",
    )
    requirements: Requirements = Field(
        description="Structured requirements extracted from the user request.",
    )
    candidates: list[CandidateRecommendation] = Field(
        min_length=1,
        max_length=8,
        description="Candidate projects with repository URLs and recommendation reasons.",
    )
    evaluation_focus: list[str] = Field(
        description="Key evaluation dimensions to focus on during research.",
    )


class ResearchTask(BaseModel):
    """A deterministic work item dispatched to exactly one specialist role."""

    task_id: str
    project_name: str
    github_url: Optional[str] = None
    track: Literal["repository", "documentation"]
    dimensions: list[str] = Field(default_factory=list)
    research_topic: str
    round: int = Field(default=0, ge=0, le=1)


class EvidenceGap(BaseModel):
    """A machine-detectable evidence coverage gap."""

    project_name: str
    track: Literal["repository", "documentation"]
    reason: str


# ===== 状态定义 =====


def override_reducer(current_value, new_value):
    """Reducer 函数，允许覆盖状态中的值。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


def token_usage_reducer(current_value, new_value):
    """Add token counters emitted by bounded model nodes."""
    current_value = current_value or {}
    new_value = new_value or {}
    return {
        "input_tokens": int(current_value.get("input_tokens", 0) or 0)
        + int(new_value.get("input_tokens", 0) or 0),
        "output_tokens": int(current_value.get("output_tokens", 0) or 0)
        + int(new_value.get("output_tokens", 0) or 0),
    }


class AgentInputState(MessagesState):
    """输入状态仅包含 'messages'。"""

    confirmed_plan: Optional[ResearchPlan] = None
    confirmed_candidates: list[str] = []


class AgentState(MessagesState):
    """主 Agent 状态 — 贯穿技术选型全流程。"""

    # 流程控制
    next: str = ""

    # 需求解析阶段
    requirements: Optional[Requirements] = None
    candidates: list[str] = []
    candidate_recommendations: list[CandidateRecommendation] = []
    evaluation_focus: list[str] = []
    confirmed_plan: Optional[ResearchPlan] = None
    confirmed_candidates: list[str] = []

    # 研究阶段
    research_brief: Optional[str] = None
    research_tasks: list[ResearchTask] = []
    research_round: int = 0
    supplemental_round_used: bool = False
    evidence_gaps: list[EvidenceGap] = []
    token_usage: Annotated[dict, token_usage_reducer] = {}
    raw_notes: Annotated[list[str], override_reducer] = []
    evidences: Annotated[list, operator.add] = []
    knowledge_stats: dict = {}

    # 评分与报告
    evaluation_criteria: Optional[EvaluationCriteria] = None
    scores: Annotated[list, operator.add] = []
    review_analysis: str = ""
    review_evidence_gaps: list[str] = []
    final_report: str = ""


class ResearcherState(dict):
    """个体研究员子图状态 — 单个项目或主题的深度研究。"""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    project_name: str = ""
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    evidences: Annotated[list[Evidence], operator.add] = []
    token_usage: Annotated[dict, token_usage_reducer] = {}
    next: str = ""


class ResearcherOutputState(BaseModel):
    """个体研究员的输出 — 压缩后的研究发现。"""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    evidences: list[Evidence] = []
    token_usage: dict = {}
