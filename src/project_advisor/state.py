"""深度研究 Agent 的图状态定义和数据结构。"""

import operator
from typing import Annotated, Optional

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


# ===== 结构化输出：Agent 工具调用 =====

class ConductResearch(BaseModel):
    """调用此工具对特定主题或项目进行研究。"""

    research_topic: str = Field(
        description="The topic to research. Should describe what to investigate and which candidate project to focus on, in detail.",
    )
    project_name: str = Field(
        default="",
        description="The candidate project this research task primarily concerns.",
    )


class ResearchComplete(BaseModel):
    """调用此工具表示研究已完成。"""


class ClarifyWithUser(BaseModel):
    """用于用户澄清请求的模型。"""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the technology requirements or scope",
    )
    verification: str = Field(
        description="Verification message confirming research will start after clarification.",
    )


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


# ===== 状态定义 =====


def override_reducer(current_value, new_value):
    """Reducer 函数，允许覆盖状态中的值。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


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
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str] = None
    raw_notes: Annotated[list[str], override_reducer] = []
    evidences: Annotated[list, operator.add] = []
    knowledge_stats: dict = {}

    # 评分与报告
    evaluation_criteria: Optional[EvaluationCriteria] = None
    scores: Annotated[list, operator.add] = []
    final_report: str = ""


class SupervisorState(dict):
    """研究主管子图状态 — 管理并行研究任务的分配和汇总。"""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    candidates: list[str]
    notes: Annotated[list[str], override_reducer] = []
    raw_notes: Annotated[list[str], override_reducer] = []
    evidences: Annotated[list[Evidence], operator.add] = []
    knowledge_stats: dict = {}
    research_iterations: int = 0
    next: str = "supervisor"


class ResearcherState(dict):
    """个体研究员子图状态 — 单个项目或主题的深度研究。"""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    project_name: str = ""
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    evidences: Annotated[list[Evidence], operator.add] = []
    next: str = ""


class ResearcherOutputState(BaseModel):
    """个体研究员的输出 — 压缩后的研究发现。"""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    evidences: list[Evidence] = []
