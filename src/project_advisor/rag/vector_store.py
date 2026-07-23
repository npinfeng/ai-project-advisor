"""Chroma 向量存储 — 文档向量的存储和相似度检索。

封装 ChromaDB，提供：
- 文档 + 向量批量插入
- 相似度检索（返回 top-K）
- 按元数据过滤
- 与 DocumentStore 同步
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


class VectorStore:
    """基于 ChromaDB 的向量存储。

    每个项目使用独立的 Collection，方便管理和过滤。
    """

    def __init__(
        self,
        storage_dir: str = "./data/vector_store",
        collection_prefix: str = "project_advisor",
    ):
        """初始化向量存储。

        Args:
            storage_dir: 持久化目录
            collection_prefix: Collection 名称前缀
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.collection_prefix = collection_prefix

        self._client = chromadb.PersistentClient(
            path=str(self.storage_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collections: dict[str, chromadb.Collection] = {}
        self._discover_collections()

    def _discover_collections(self) -> None:
        """Restore persisted project collections after a process restart."""
        try:
            collections = self._client.list_collections()
        except Exception:
            return
        for item in collections:
            try:
                collection = (
                    item
                    if hasattr(item, "count")
                    else self._client.get_collection(name=str(item))
                )
                metadata = getattr(collection, "metadata", None) or {}
                project_name = metadata.get("project")
                if project_name:
                    self._collections[str(project_name)] = collection
            except Exception:
                continue

    def _get_collection_name(self, project_name: str) -> str:
        """获取项目的 Collection 名称。"""
        safe_name = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in project_name
        )
        return f"{self.collection_prefix}_{safe_name}"

    def _get_or_create_collection(
        self, project_name: str
    ) -> chromadb.Collection:
        """获取或创建项目的 Collection。"""
        if project_name in self._collections:
            return self._collections[project_name]

        collection_name = self._get_collection_name(project_name)
        try:
            collection = self._client.get_collection(name=collection_name)
        except Exception:
            collection = self._client.create_collection(
                name=collection_name,
                metadata={"project": project_name},
            )

        self._collections[project_name] = collection
        return collection

    def _get_existing_collection(
        self, project_name: str
    ) -> Optional[chromadb.Collection]:
        if project_name in self._collections:
            return self._collections[project_name]
        try:
            collection = self._client.get_collection(
                name=self._get_collection_name(project_name)
            )
        except Exception:
            return None
        self._collections[project_name] = collection
        return collection

    @staticmethod
    def _metadata_for_chroma(metadata: dict) -> dict:
        normalized = {}
        for key, value in metadata.items():
            if value is None:
                normalized[str(key)] = ""
            elif isinstance(value, (str, int, float, bool)):
                normalized[str(key)] = value
            else:
                normalized[str(key)] = json.dumps(
                    value, ensure_ascii=False, default=str
                )
        return normalized

    def add_documents(
        self,
        project_name: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """批量添加文档 chunk 和对应的嵌入向量。

        Args:
            project_name: 项目名称
            chunks: chunk 列表，每个包含 text 和 metadata
            embeddings: 对应的嵌入向量列表

        Returns:
            添加的文档数量
        """
        if not chunks or not embeddings:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量必须一致。")

        collection = self._get_or_create_collection(project_name)

        ids = []
        documents = []
        metadatas = []
        embeds = []

        for chunk, embedding in zip(chunks, embeddings):
            metadata = chunk.get("metadata", {})
            doc_id = chunk.get("id") or metadata.get("chunk_id")
            if not doc_id:
                identity = "\x1f".join([
                    project_name,
                    str(metadata.get("source_url", "")),
                    chunk["text"],
                ])
                doc_id = f"chunk_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"

            ids.append(doc_id)
            documents.append(chunk["text"])
            metadatas.append(self._metadata_for_chroma(metadata))
            embeds.append(embedding)

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeds,
        )
        return len(ids)

    def search(
        self,
        query_embedding: list[float],
        project_name: Optional[str] = None,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """向量相似度检索。

        Args:
            query_embedding: 查询的嵌入向量
            project_name: 限定搜索范围（可选，不指定则搜索全部）
            top_k: 返回数量
            filter_metadata: 元数据过滤条件

        Returns:
            搜索结果列表，每个包含 id、text、metadata、score
        """
        results = []

        projects = (
            [project_name] if project_name else list(self._collections.keys())
        )

        for proj in projects:
            collection = self._get_existing_collection(proj)
            if collection is None:
                continue
            collection_count = collection.count()
            if collection_count < 1:
                continue

            where_filter = filter_metadata or None
            query_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection_count),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            if query_results.get("ids") and query_results["ids"][0]:
                for j in range(len(query_results["ids"][0])):
                    results.append({
                        "id": query_results["ids"][0][j],
                        "text": query_results["documents"][0][j],
                        "metadata": query_results["metadatas"][0][j],
                        "score": 1.0 - query_results["distances"][0][j],
                        "project": proj,
                    })

        # 按分数降序排列
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_project(self, project_name: str):
        """删除项目的所有向量数据。"""
        collection_name = self._get_collection_name(project_name)
        try:
            self._client.delete_collection(name=collection_name)
        except Exception:
            pass
        self._collections.pop(project_name, None)

    def count(self, project_name: Optional[str] = None) -> int:
        """统计文档数量。"""
        if project_name:
            collection = self._get_existing_collection(project_name)
            return collection.count() if collection is not None else 0
        else:
            return sum(
                self.count(proj) for proj in self._collections
            )

    def list_projects(self) -> list[str]:
        """列出所有已索引的项目。"""
        return list(self._collections.keys())

    def document_ids(self, project_name: str) -> set[str]:
        """Return all persisted stable chunk IDs for one project."""
        collection = self._get_existing_collection(project_name)
        if collection is None or collection.count() < 1:
            return set()
        result = collection.get(include=[])
        return set(result.get("ids", []))

    def delete_documents(self, project_name: str, document_ids: set[str]) -> int:
        """Delete selected stale chunks without dropping the project collection."""
        if not document_ids:
            return 0
        collection = self._get_existing_collection(project_name)
        if collection is None:
            return 0
        collection.delete(ids=sorted(document_ids))
        return len(document_ids)
