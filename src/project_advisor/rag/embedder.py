"""嵌入器 — 将文本转换为向量表示。

支持：
- 本地 Sentence Transformers 模型（离线、免费）
- OpenAI Embeddings（高质量、付费）
- 缓存已计算的嵌入
"""

import hashlib
from typing import Optional

import numpy as np


class Embedder:
    """文本嵌入器。

    优先使用本地模型避免 API 成本，
    也支持 OpenAI 等云端嵌入服务。
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        provider: str = "local",
    ):
        """初始化嵌入器。

        Args:
            model_name: 嵌入模型名称
            provider: 'local'（Sentence Transformers）或 'openai'
        """
        self.model_name = model_name
        self.provider = provider
        self._model = None
        self._cache: dict[str, list[float]] = {}

    def _get_local_model(self):
        """延迟加载本地 Sentence Transformers 模型。"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "请安装 sentence-transformers：pip install sentence-transformers"
                )
        return self._model

    def _get_openai_embedding(self, texts: list[str]) -> list[list[float]]:
        """使用 OpenAI API 生成嵌入。"""
        import os
        import httpx

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置，请检查 .env 文件。")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        response = httpx.post(
            f"{base_url.rstrip('/')}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model_name, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json().get("data") or []
        return [item["embedding"] for item in data]

    def embed(self, text: str) -> list[float]:
        """对单个文本进行嵌入。使用缓存避免重复计算。

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.provider == "openai":
            vectors = self._get_openai_embedding([text])
            result = vectors[0]
            self._cache[cache_key] = result
            return result
        else:
            model = self._get_local_model()
            result = model.encode(text).tolist()
            self._cache[cache_key] = result
            return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本。

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        results = []
        uncached_texts = []
        uncached_indices = []

        # 先查缓存
        for i, text in enumerate(texts):
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            if cache_key in self._cache:
                results.append((i, self._cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 批量计算未缓存文本的嵌入
        if uncached_texts:
            if self.provider == "openai":
                vectors = self._get_openai_embedding(uncached_texts)
            else:
                model = self._get_local_model()
                vectors = model.encode(uncached_texts).tolist()

            for idx, text, vec in zip(uncached_indices, uncached_texts, vectors):
                cache_key = hashlib.sha256(text.encode()).hexdigest()
                self._cache[cache_key] = vec
                results.append((idx, vec))

        # 按原始顺序返回
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
