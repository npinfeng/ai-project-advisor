"""配置管理 — 模型、API、评分权重等所有可调参数。"""

import os
from enum import Enum
from typing import Any, Literal, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, model_validator


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
        ge=1,
        le=16,
        description="最大并发研究单元数",
    )
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        description="搜索 API 选择",
    )

    # ===== 研究迭代配置 =====
    max_react_tool_calls: int = Field(
        default=10,
        ge=1,
        le=30,
        description="单个研究员最大工具调用次数",
    )
    max_tool_calls_per_step: int = Field(
        default=5,
        ge=1,
        le=20,
        description="单轮模型响应允许并行执行的 Tool 调用数",
    )

    # ===== 超时与重试 =====
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="单次 LLM HTTP 请求超时（秒）",
    )
    tool_timeout_seconds: float = Field(
        default=45.0,
        gt=0,
        description="单次 Tool 调用硬超时（秒）",
    )
    tool_max_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Tool 遇到瞬时错误后的最大重试次数",
    )
    tool_retry_backoff_seconds: float = Field(
        default=0.5,
        ge=0,
        le=30,
        description="Tool 重试的指数退避基数（秒）",
    )
    agent_run_timeout_seconds: float = Field(
        default=600.0,
        gt=0,
        description="单次 Agent Run 的端到端硬超时（秒）",
    )

    # ===== Reviewer 上下文预算 =====
    reviewer_context_max_chars: int = Field(
        default=30000,
        ge=2000,
        description="全部候选项目可发送给 Reviewer 的证据字符总预算",
    )
    reviewer_evidence_max_chars: int = Field(
        default=3000,
        ge=200,
        description="单条 Evidence 可发送给 Reviewer 的最大字符数",
    )

    # ===== Evidence 生命周期 =====
    evidence_stale_after_days: int = Field(
        default=180,
        ge=0,
        description="Evidence 标记为 stale 前允许的最大年龄（天）",
    )
    evidence_expire_after_days: int = Field(
        default=365,
        ge=1,
        description="Evidence 停止索引并允许清理前的最大年龄（天）",
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
    final_report_model: str = Field(
        default="deepseek:deepseek-chat",
        description="受限 Reviewer 使用的结构化输出模型",
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        description="最终报告模型最大输出 token 数",
    )
    embedding_provider: Literal["local", "openai"] = Field(
        default="local",
        description="Embedding 提供方：本地 Sentence Transformers 或 OpenAI 兼容接口",
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        min_length=1,
        description="RAG 使用的中英混合 Embedding 模型",
    )
    embedding_normalize: bool = Field(
        default=True,
        description="是否对文档和 Query Embedding 做 L2 归一化",
    )
    embedding_query_instruction: str = Field(
        default="",
        description="仅添加到检索 Query 的模型指令；BGE-M3 应保持为空",
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
    repository_mcp_tool_allowlist: str = Field(
        default="",
        description="Repository Agent 可使用的 MCP 工具名，逗号分隔；默认全部拒绝",
    )
    documentation_mcp_tool_allowlist: str = Field(
        default="",
        description="Documentation Agent 可使用的 MCP 工具名，逗号分隔；默认全部拒绝",
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

    # ===== Web 安全与资源保护 =====
    advisor_api_key: Optional[str] = Field(
        default=None,
        description="可选的 Web API 访问密钥；为空时仅建议绑定回环地址使用",
    )
    api_rate_limit_per_minute: int = Field(
        default=20,
        ge=1,
        description="单个客户端每分钟最多发起的高成本 API 请求数",
    )
    max_concurrent_evaluations: int = Field(
        default=2,
        ge=1,
        description="单进程允许同时执行的深度评估任务数",
    )
    max_run_tokens: int = Field(
        default=0,
        ge=0,
        description="单次评估 Token 硬上限；0 表示不启用",
    )
    max_run_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="单次评估已观测成本硬上限；0 表示不启用",
    )
    checkpoint_db_path: str = Field(
        default="data/checkpoints.sqlite3",
        description="LangGraph SQLite checkpoint 数据库路径",
    )
    task_db_path: str = Field(
        default="data/tasks.sqlite3",
        description="任务状态、报告与恢复元数据数据库路径",
    )

    # ===== 评分权重配置 =====
    weight_feature_match: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_engineering_reliability: float = Field(default=0.20, ge=0.0, le=1.0)
    weight_community: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_documentation: float = Field(default=0.10, ge=0.0, le=1.0)
    weight_learning_cost: float = Field(default=0.10, ge=0.0, le=1.0)
    weight_extensibility: float = Field(default=0.10, ge=0.0, le=1.0)
    weight_deployment_cost: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_scoring_weights(self) -> "Configuration":
        """Reject silently misconfigured rankings instead of normalizing them."""
        total = sum((
            self.weight_feature_match,
            self.weight_engineering_reliability,
            self.weight_community,
            self.weight_documentation,
            self.weight_learning_cost,
            self.weight_extensibility,
            self.weight_deployment_cost,
        ))
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"评分权重之和必须为 1.0，当前为 {total:.6f}。")
        if self.evidence_expire_after_days <= self.evidence_stale_after_days:
            raise ValueError("Evidence 过期天数必须大于陈旧天数。")
        return self

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
