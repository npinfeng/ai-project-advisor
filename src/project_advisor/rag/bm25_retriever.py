"""BM25 关键词检索器 — 基于精确术语匹配的搜索。

优势：
- 对类名、API 名、版本号等精确术语敏感
- 与向量检索互补：向量擅长语义，BM25 擅长精确匹配
- 轻量级，不需要 GPU
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25L

from project_advisor.rag.text_analysis import lexical_tokens


class BM25Retriever:
    """BM25 关键词检索。

    对文档进行分词后构建 BM25 索引，
    支持项目级过滤和 Top-K 检索。
    """

    def __init__(self, storage_dir: str | Path | None = "./data/bm25"):
        """初始化 BM25 检索器，并恢复磁盘上的项目索引。"""
        self._indexes: dict[str, dict] = {}  # project_name → {corpus, bm25, metadata}
        self.storage_dir = Path(storage_dir) if storage_dir is not None else None
        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_all()

    def _project_file(self, project_name: str) -> Path:
        if self.storage_dir is None:
            raise RuntimeError("BM25 persistence is disabled.")
        digest = hashlib.sha256(project_name.encode("utf-8")).hexdigest()[:16]
        return self.storage_dir / f"{digest}.json"

    def _build_index(self, project_name: str, chunks: list[dict]) -> None:
        corpus = [chunk["text"] for chunk in chunks]
        tokenized = [self._tokenize(doc) for doc in corpus]
        self._indexes[project_name] = {
            "corpus": corpus,
            "tokenized": tokenized,
            # BM25Okapi produces a negative score for an exact hit in a
            # single-document corpus. BM25L remains well-defined for the small
            # per-project indexes used by this application.
            "bm25": BM25L(tokenized),
            "metadata": [chunk.get("metadata", {}) for chunk in chunks],
            "ids": [
                chunk.get("id")
                or chunk.get("metadata", {}).get("chunk_id")
                or f"bm25_{project_name}_{index}"
                for index, chunk in enumerate(chunks)
            ],
        }

    def _save_index(self, project_name: str) -> None:
        if self.storage_dir is None:
            return
        data = self._indexes[project_name]
        payload = {
            "project_name": project_name,
            "chunks": [
                {"id": doc_id, "text": text, "metadata": metadata}
                for doc_id, text, metadata in zip(
                    data["ids"], data["corpus"], data["metadata"]
                )
            ],
        }
        path = self._project_file(project_name)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_path, path)

    def _load_all(self) -> None:
        if self.storage_dir is None:
            return
        for path in self.storage_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                project_name = str(payload["project_name"])
                chunks = payload.get("chunks", [])
                if chunks:
                    self._build_index(project_name, chunks)
            except (OSError, ValueError, KeyError, TypeError):
                continue

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize Chinese phrases and Latin/API/version terms consistently."""
        return lexical_tokens(text)

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
            self.clear_project(project_name)
            return
        self._build_index(project_name, chunks)
        self._save_index(project_name)

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

        raw_results = []

        projects = (
            [project_name] if project_name else list(self._indexes.keys())
        )

        for proj in projects:
            if proj not in self._indexes:
                continue

            index_data = self._indexes[proj]
            bm25 = index_data["bm25"]
            scores = bm25.get_scores(query_tokens)

            # BM25L uses a positive delta baseline, so explicitly reject
            # documents with no lexical overlap before normalizing scores.
            query_set = set(query_tokens)
            matching = [
                i for i, tokens in enumerate(index_data["tokenized"])
                if query_set.intersection(tokens)
            ]
            for i in matching:
                raw_score = float(scores[i])
                if raw_score > 0:
                    overlap = len(query_set.intersection(index_data["tokenized"][i])) / len(query_set)
                    raw_results.append({
                        "id": index_data["ids"][i],
                        "text": index_data["corpus"][i],
                        "metadata": index_data["metadata"][i],
                        "score": raw_score,
                        "lexical_overlap": overlap,
                        "project": proj,
                    })

        # Normalize once across the complete search scope. Per-project
        # normalization incorrectly made every project's best result score 1.0.
        max_score = max((item["score"] for item in raw_results), default=0.0)
        all_results = [
            {**item, "score": item["score"] / max_score}
            for item in raw_results
        ] if max_score > 0 else []
        all_results.sort(
            key=lambda x: (-x["score"], -x["lexical_overlap"], str(x["id"]))
        )
        return all_results[:top_k]

    def clear_project(self, project_name: str):
        """清除项目的 BM25 索引。"""
        self._indexes.pop(project_name, None)
        if self.storage_dir is not None:
            path = self._project_file(project_name)
            if path.exists():
                path.unlink()

    def count(self, project_name: Optional[str] = None) -> int:
        """统计已索引的文档数量。"""
        if project_name is not None:
            return len(self._indexes.get(project_name, {}).get("corpus", []))
        return sum(
            len(idx["corpus"]) for idx in self._indexes.values()
        )

    def list_projects(self) -> list[str]:
        """列出已经恢复或构建 BM25 索引的项目。"""
        return list(self._indexes)

    def document_ids(self, project_name: str) -> set[str]:
        """Return stable chunk IDs in a project's keyword index."""
        return set(self._indexes.get(project_name, {}).get("ids", []))
