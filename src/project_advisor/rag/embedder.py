"""嵌入器 — 将文本转换为适合中英混合检索的向量表示。

支持：
- 本地 Sentence Transformers 模型（离线、免费）
- OpenAI Embeddings（高质量、付费）
- 缓存已计算的嵌入
"""

import hashlib
import json

import numpy as np


class Embedder:
    """文本嵌入器。

    优先使用本地模型避免 API 成本，
    也支持 OpenAI 等云端嵌入服务。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        provider: str = "local",
        normalize_embeddings: bool = True,
        query_instruction: str = "",
    ):
        """初始化嵌入器。

        Args:
            model_name: 嵌入模型名称
            provider: 'local'（Sentence Transformers）或 'openai'
            normalize_embeddings: 是否将输出向量归一化
            query_instruction: 仅添加到检索 Query 的指令前缀；BGE-M3 留空
        """
        if provider not in {"local", "openai"}:
            raise ValueError("Embedding provider 必须是 'local' 或 'openai'。")
        self.model_name = model_name
        self.provider = provider
        self.normalize_embeddings = normalize_embeddings
        self.query_instruction = query_instruction
        self._model = None
        self._cache: dict[str, list[float]] = {}

    @property
    def index_identity(self) -> str:
        """Return a stable identity for persisted vectors produced by this embedder."""
        payload = json.dumps(
            {
                "schema": 1,
                "provider": self.provider,
                "model": self.model_name,
                "normalize_embeddings": self.normalize_embeddings,
                "query_instruction": self.query_instruction,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    @property
    def index_metadata(self) -> dict[str, str | bool | int]:
        """Metadata persisted with a vector collection for migration checks."""
        return {
            "embedding_schema": 1,
            "embedding_identity": self.index_identity,
            "embedding_provider": self.provider,
            "embedding_model": self.model_name,
            "embedding_normalized": self.normalize_embeddings,
        }

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
        vectors = [item["embedding"] for item in data]
        return self._normalize_vectors(vectors)

    def _normalize_vectors(self, vectors) -> list[list[float]]:
        """Convert model output to lists and optionally apply L2 normalization."""
        array = np.asarray(vectors, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if self.normalize_embeddings and array.size:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            array = array / norms
        return array.tolist()

    def _prepare_query(self, text: str) -> str:
        return f"{self.query_instruction}{text}" if self.query_instruction else text

    @staticmethod
    def _cache_key(kind: str, text: str) -> str:
        return hashlib.sha256(f"{kind}\x1f{text}".encode("utf-8")).hexdigest()

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        model = self._get_local_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
        )
        return self._normalize_vectors(vectors)

    def _embed_texts(self, texts: list[str], *, kind: str) -> list[list[float]]:
        """Embed query or document text with mode-specific caching."""
        results: list[tuple[int, list[float]]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for index, text in enumerate(texts):
            cache_key = self._cache_key(kind, text)
            if cache_key in self._cache:
                results.append((index, self._cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(index)

        if uncached_texts:
            vectors = (
                self._get_openai_embedding(uncached_texts)
                if self.provider == "openai"
                else self._embed_local(uncached_texts)
            )
            if len(vectors) != len(uncached_texts):
                raise RuntimeError("Embedding 服务返回的向量数量与输入数量不一致。")
            for index, text, vector in zip(
                uncached_indices, uncached_texts, vectors
            ):
                self._cache[self._cache_key(kind, text)] = vector
                results.append((index, vector))

        results.sort(key=lambda item: item[0])
        return [vector for _, vector in results]

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query, applying its optional instruction only here."""
        prepared = self._prepare_query(text)
        return self._embed_texts([prepared], kind="query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages without a query instruction."""
        return self._embed_texts(texts, kind="document")

    def embed(self, text: str) -> list[float]:
        """向后兼容的 Query 嵌入入口。使用缓存避免重复计算。

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        return self.embed_query(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """向后兼容的文档批量嵌入入口。

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        return self.embed_documents(texts)
