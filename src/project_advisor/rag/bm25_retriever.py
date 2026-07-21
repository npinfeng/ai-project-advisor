"""BM25 关键词检索器 — 基于精确术语匹配的搜索。

优势：
- 对类名、API 名、版本号等精确术语敏感
- 与向量检索互补：向量擅长语义，BM25 擅长精确匹配
- 轻量级，不需要 GPU
"""

import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 关键词检索。

    对文档进行分词后构建 BM25 索引，
    支持项目级过滤和 Top-K 检索。
    """

    def __init__(self):
        """初始化 BM25 检索器。"""
        self._indexes: dict[str, dict] = {}  # project_name → {corpus, bm25, metadata}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词：小写化 + 按非字母数字字符分割。

        针对代码和技术文档优化：
        - 保留 CamelCase 和 snake_case 的完整形式
        - 保留版本号（如 v2.0、1.5.3）
        """
        text_lower = text.lower()
        # 保留点号连接的版本号
        tokens = re.findall(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", text_lower)
        return [t for t in tokens if len(t) > 1]

    def index(
        self,
        project_name: str,
        chunks: list[dict],
    ):
        """为项目构建 BM25 索引。

        Args:
            project_name: 项目名称
            chunks: chunk 列表，每个包含 text 和 metadata
        """
        if not chunks:
            return

        corpus = [chunk["text"] for chunk in chunks]
        tokenized = [self._tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized)

        self._indexes[project_name] = {
            "corpus": corpus,
            "tokenized": tokenized,
            "bm25": bm25,
            "metadata": [chunk.get("metadata", {}) for chunk in chunks],
        }

    def search(
        self,
        query: str,
        project_name: Optional[str] = None,
        top_k: int = 10,
    ) -> list[dict]:
        """BM25 关键词检索。

        Args:
            query: 搜索查询
            project_name: 限定项目
            top_k: 返回数量

        Returns:
            搜索结果列表
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        all_results = []

        projects = (
            [project_name] if project_name else list(self._indexes.keys())
        )

        for proj in projects:
            if proj not in self._indexes:
                continue

            index_data = self._indexes[proj]
            bm25 = index_data["bm25"]
            scores = bm25.get_scores(query_tokens)

            # 归一化分数到 0-1 范围
            max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
            normalized = scores / max_score

            for i, score in enumerate(normalized):
                if score > 0:
                    all_results.append({
                        "id": f"bm25_{proj}_{i}",
                        "text": index_data["corpus"][i],
                        "metadata": index_data["metadata"][i],
                        "score": float(score),
                        "project": proj,
                    })

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def clear_project(self, project_name: str):
        """清除项目的 BM25 索引。"""
        self._indexes.pop(project_name, None)

    def count(self) -> int:
        """统计已索引的文档数量。"""
        return sum(
            len(idx["corpus"]) for idx in self._indexes.values()
        )
