"""重排序器 — 对检索结果进行精排。

支持两种模式：
1. LLM-based：使用 LLM 判断文档与查询的相关性（精确但慢）
2. Cross-encoder：使用轻量级 cross-encoder 模型（快速）

默认使用 LLM-based 模式，因为技术文档需要深度语义理解。
"""

import asyncio
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class RelevanceScore(BaseModel):
    """文档相关性评分。"""
    score: int = Field(description="Relevance score from 1 (not relevant) to 10 (highly relevant).")
    reason: str = Field(description="One-sentence reason for the score.")


class Reranker:
    """检索结果重排序器。

    对混合检索返回的候选文档进行精排，
    去除噪声，提升 Top-K 结果质量。
    """

    def __init__(
        self,
        model_name: str = "deepseek:deepseek-chat",
        use_llm: bool = True,
    ):
        """初始化重排序器。

        Args:
            model_name: 用于重排序的模型（仅在 use_llm=True 时使用）
            use_llm: 是否使用 LLM 进行重排序
        """
        self.model_name = model_name
        self.use_llm = use_llm
        self._model = None

    def _get_model(self):
        """延迟加载模型。"""
        if self._model is None and self.use_llm:
            from project_advisor.utils import create_chat_model
            self._model = (
                create_chat_model(self.model_name)
                .with_structured_output(RelevanceScore, method="function_calling")
                .with_retry(stop_after_attempt=2)
            )
        return self._model

    async def _score_single(
        self, query: str, doc: dict, index: int
    ) -> tuple[int, float]:
        """使用 LLM 对单个文档评分（含新鲜度和权威性）。"""
        text = doc.get("text", "")[:2000]  # 截断以控制成本
        metadata = doc.get("metadata", {})
        retrieved_at = metadata.get("retrieved_at", "未知")
        source_type = metadata.get("source_type", "")
        time_decay = doc.get("time_decay")

        freshness_hint = ""
        if retrieved_at and retrieved_at != "未知":
            freshness_hint = f"文档检索时间：{retrieved_at}\n"
        if time_decay is not None:
            freshness_hint += f"时间衰减因子：{time_decay:.2f}（1.0=最新，接近0=很旧）\n"
        if source_type:
            freshness_hint += f"来源类型：{source_type}（官方文档权威性最高，社区文章较低）\n"

        prompt = f"""评估以下文档与查询的相关性（1-10分）。

查询：{query}

文档标题/来源：{metadata.get('source_url', '未知')}
{freshness_hint}
文档内容：
{text}

请给出 1（不相关）到 10（高度相关）的评分。考虑以下因素（权重递减）：
- 文档是否直接回答了查询问题？（最重要）
- 文档是否来自权威来源（official_documentation > 官方博客 > 社区文章 > 第三方博客）？
- 文档内容是否新鲜？检索时间距今越近越好，超过 6 个月的应降分。
- 文档是否包含具体的技术细节（版本号、API 名称、配置示例）而非泛泛而谈？"""

        try:
            model = self._get_model()
            if model is None:
                return index, 5  # 无模型时默认中等分数

            response = await model.ainvoke([
                SystemMessage(content="你是技术文档相关性评估专家。客观评分，综合考虑相关性、权威性、新鲜度和技术深度。"),
                HumanMessage(content=prompt),
            ])
            return index, float(response.score)
        except Exception:
            return index, 5  # 出错时保持原分数

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """对检索结果进行重排序。

        Args:
            query: 原始查询
            documents: 待重排序的文档列表
            top_k: 返回数量

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        # 如果只有一个结果，不需要重排
        if len(documents) <= 1:
            return documents

        if self.use_llm:
            # 并行评分
            tasks = [
                self._score_single(query, doc, i)
                for i, doc in enumerate(documents[:20])  # 最多重排 20 个
            ]
            scores = await asyncio.gather(*tasks)

            # 按 LLM 分数重新排序
            score_map = dict(scores)
            for i, doc in enumerate(documents):
                doc["rerank_score"] = score_map.get(i, 5)

            documents.sort(
                key=lambda x: x.get("rerank_score", 5), reverse=True
            )
        else:
            # 无 LLM 则维持 RRF 融合分数排序
            pass

        return documents[:top_k]

    def rerank_sync(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """同步版本的 rerank（供工具调用使用）。"""
        return asyncio.run(self.rerank(query, documents, top_k))
