"""Async query rewriting with bounded structured-output fallback.

策略：
1. Query Rewrite：将用户问题改写为更适合检索的查询
2. Multi-Query：从功能、工程、社区等角度生成多个子查询
3. Context Compression：从检索结果中只保留与查询相关的段落
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from project_advisor.observability.logging import log_event
from project_advisor.usage_tracking import record_message_usage
from project_advisor.utils import invoke_structured_with_retry

logger = logging.getLogger(__name__)


class RewrittenQuery(BaseModel):
    """改写后的查询。"""
    rewritten: str = Field(description="The rewritten query optimized for retrieval.")


class MultiQueries(BaseModel):
    """多角度查询列表。"""
    queries: list[str] = Field(
        description="List of query variations from different evaluation perspectives."
    )


class QueryRewriter:
    """查询改写器。

    使用 LLM 将用户问题改写为更精确的检索查询，
    支持生成多角度子查询以覆盖不同评估维度。
    """

    def __init__(
        self,
        model_name: str = "deepseek:deepseek-chat",
        *,
        timeout_seconds: float = 60.0,
        max_attempts: int = 2,
    ):
        """初始化查询改写器。

        Args:
            model_name: 用于查询改写的模型
        """
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self._rewrite_model = None
        self._multi_query_model = None

    def _get_rewrite_model(self):
        if self._rewrite_model is None:
            from project_advisor.utils import create_chat_model
            self._rewrite_model = create_chat_model(
                self.model_name,
                timeout_seconds=self.timeout_seconds,
            ).with_structured_output(
                RewrittenQuery,
                method="function_calling",
                include_raw=True,
            )
        return self._rewrite_model

    def _get_multi_query_model(self):
        if self._multi_query_model is None:
            from project_advisor.utils import create_chat_model
            self._multi_query_model = create_chat_model(
                self.model_name,
                timeout_seconds=self.timeout_seconds,
            ).with_structured_output(
                MultiQueries,
                method="function_calling",
                include_raw=True,
            )
        return self._multi_query_model

    async def rewrite(self, query: str) -> str:
        """将用户问题改写为更适合检索的查询。

        优化方向：
        - 去除口语化表达
        - 补充技术关键词
        - 将模糊问题转化为精确查询

        Args:
            query: 原始用户查询

        Returns:
            改写后的查询
        """
        model = self._get_rewrite_model()
        prompt = f"""请将以下技术选型问题改写为更精确的检索查询。

原始问题：{query}

改写要求：
1. 补充相关的技术关键词（如框架名、API 名、技术术语）
2. 去除口语化表达（如"我想知道"、"帮我查一下"）
3. 保留所有技术约束条件
4. 如果问题涉及版本，补充当前年份 2026

只返回改写后的查询，不要解释。"""

        try:
            response, _ = await invoke_structured_with_retry(
                model,
                [
                    SystemMessage(content="你是信息检索专家。将用户问题改写为高效的搜索查询。"),
                    HumanMessage(content=prompt),
                ],
                max_attempts=self.max_attempts,
                on_raw=record_message_usage,
            )
            return response.rewritten
        except Exception as error:
            log_event(
                logger,
                logging.WARNING,
                "query_rewrite_fallback",
                error_type=type(error).__name__,
            )
            return query

    async def generate_multi_queries(
        self,
        query: str,
        candidates: list[str] | None = None,
    ) -> list[str]:
        """生成多角度子查询，覆盖不同评估维度。

        从功能、工程、社区、文档、部署等角度分别生成查询。

        Args:
            query: 原始用户查询
            candidates: 候选项目列表（可选）

        Returns:
            子查询列表（含原始查询）
        """
        model = self._get_multi_query_model()

        candidates_str = ", ".join(candidates) if candidates else "相关开源项目"
        prompt = f"""针对以下技术选型问题，从不同评估维度生成多个具体搜索查询。

原始问题：{query}
候选项目：{candidates_str}

请从以下角度各生成一个查询：
1. **功能匹配**：项目是否支持特定功能特性？
2. **工程可靠性**：项目的 Release 频率、Issue 处理、代码质量？
3. **社区活跃度**：社区规模、维护者响应、最新动态？
4. **文档质量**：官方文档完整性、教程、API 参考？
5. **架构扩展**：插件系统、自定义能力、集成生态？

每个查询应该具体、可搜索、包含项目名称。
只返回查询列表。"""

        try:
            response, _ = await invoke_structured_with_retry(
                model,
                [
                    SystemMessage(content="你是技术搜索专家。从多角度生成精确的搜索查询以覆盖技术选型的所有维度。"),
                    HumanMessage(content=prompt),
                ],
                max_attempts=self.max_attempts,
                on_raw=record_message_usage,
            )
            # 原始查询也保留
            all_queries = [query] + response.queries
            return all_queries[:6]  # 最多 6 个查询
        except Exception as error:
            log_event(
                logger,
                logging.WARNING,
                "multi_query_fallback",
                error_type=type(error).__name__,
            )
            return [query]
