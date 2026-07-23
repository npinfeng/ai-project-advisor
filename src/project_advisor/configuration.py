"""配置管理 — 模型、API、评分权重等所有可调参数。"""

import os
from enum import Enum
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """可用的搜索 API 提供者。"""

    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    NONE = "none"


class Configuration(BaseModel):
    """项目 Advisor 的主配置类。"""

    # ===== 通用配置 =====
    max_structured_output_retries: int = Field(
        default=3,
        description="结构化输出调用最大重试次数",
    )
    allow_clarification: bool = Field(
        default=True,
        description="是否在开始研究前向用户追问以澄清需求",
    )
    max_concurrent_research_units: int = Field(
        default=5,
        description="最大并发研究单元数",
    )
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        description="搜索 API 选择",
    )

    # ===== 研究迭代配置 =====
    max_researcher_iterations: int = Field(
        default=6,
        description="研究主管最大迭代次数",
    )
    max_react_tool_calls: int = Field(
        default=10,
        description="单个研究员最大工具调用次数",
    )

    # ===== 模型配置 =====
    research_model: str = Field(
        default="deepseek:deepseek-chat",
        description="研究模型（Agent 决策和工具调用）",
    )
    research_model_max_tokens: int = Field(
        default=10000,
        description="研究模型最大输出 token 数",
    )
    compression_model: str = Field(
        default="deepseek:deepseek-chat",
        description="压缩模型（汇总研究发现）",
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        description="压缩模型最大输出 token 数",
    )
    final_report_model: str = Field(
        default="deepseek:deepseek-chat",
        description="最终报告模型",
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        description="最终报告模型最大输出 token 数",
    )

    # ===== GitHub 配置 =====
    github_token: Optional[str] = Field(
        default=None,
        description="GitHub Personal Access Token（提升 API 频率限制）",
    )

    # ===== MCP 配置 =====
    enable_local_mcp: bool = Field(
        default=True,
        description="是否启用项目内置的 stdio MCP Server",
    )
    mcp_servers_json: str = Field(
        default="",
        description="额外 MCP Server 的 JSON 连接配置",
    )
    mcp_required: bool = Field(
        default=False,
        description="MCP 连接失败时是否中断工作流",
    )
    mcp_connect_timeout_seconds: float = Field(
        default=20.0,
        gt=0,
        description="MCP 工具发现超时时间（秒）",
    )

    # ===== 运行诊断配置 =====
    input_price_per_million: float = Field(
        default=0.0,
        ge=0.0,
        description="诊断面板使用的输入 Token 单价（USD / 1M Token）",
    )
    output_price_per_million: float = Field(
        default=0.0,
        ge=0.0,
        description="诊断面板使用的输出 Token 单价（USD / 1M Token）",
    )

    # ===== 评分权重配置 =====
    weight_feature_match: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_engineering_reliability: float = Field(default=0.20, ge=0.0, le=1.0)
    weight_community: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_documentation: float = Field(default=0.10, ge=0.0, le=1.0)
    weight_learning_cost: float = Field(default=0.10, ge=0.0, le=1.0)
    weight_extensibility: float = Field(default=0.10, ge=0.0, le=1.0)
    weight_deployment_cost: float = Field(default=0.05, ge=0.0, le=1.0)

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 RunnableConfig 创建 Configuration 实例。"""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
