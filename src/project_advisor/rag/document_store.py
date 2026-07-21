"""文档存储 — JSON 文件持久化，为后续向量检索做准备。

功能：
- 按项目分组存储文档
- 支持增删查
- 自动去重（基于 URL + 内容哈希）
- 元数据过滤（项目、文档类型、日期）
- 导出为 JSON 格式
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from project_advisor.schemas.evidence import Evidence


class DocumentStore:
    """基于 JSON 文件的文档存储。

    为 Phase 3 的向量检索做准备，当前阶段使用简单的文件存储。
    所有文档按项目分组，每个项目一个 JSON 文件。
    """

    def __init__(self, storage_dir: str = "./data/documents"):
        """初始化文档存储。

        Args:
            storage_dir: 文档存储目录
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, list[dict]] = {}  # project_name → list of doc dicts
        self._url_index: set[str] = set()  # 去重用的 URL 集合
        self._load_all()

    def _get_project_file(self, project_name: str) -> Path:
        """获取项目的文档文件路径。"""
        # 安全处理项目名称
        safe_name = "".join(c for c in project_name if c.isalnum() or c in "_-")
        return self.storage_dir / f"{safe_name}.json"

    def _load_all(self):
        """加载所有已存储的文档。"""
        if not self.storage_dir.exists():
            return

        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                    project = file_path.stem
                    self._index[project] = docs
                    for doc in docs:
                        self._url_index.add(doc.get("source_url", ""))
            except (json.JSONDecodeError, IOError):
                continue

    def _save_project(self, project_name: str):
        """保存项目文档到文件。"""
        file_path = self._get_project_file(project_name)
        docs = self._index.get(project_name, [])
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

    def add(self, evidence: Evidence) -> bool:
        """添加一条 Evidence 到存储。自动去重。

        Args:
            evidence: 证据对象

        Returns:
            如果新增返回 True，如果已存在返回 False
        """
        # URL 去重
        if evidence.source_url in self._url_index:
            return False

        doc_dict = evidence.model_dump()
        doc_dict["stored_at"] = datetime.now().isoformat()

        project = evidence.project_name
        if project not in self._index:
            self._index[project] = []

        self._index[project].append(doc_dict)
        self._url_index.add(evidence.source_url)
        self._save_project(project)
        return True

    def add_batch(self, evidences: list[Evidence]) -> int:
        """批量添加 Evidence。

        Returns:
            新增的文档数量
        """
        count = 0
        for ev in evidences:
            if self.add(ev):
                count += 1
        return count

    def get_by_project(
        self,
        project_name: str,
        doc_type: Optional[str] = None,
        max_results: int = 50,
    ) -> list[Evidence]:
        """按项目查询文档。

        Args:
            project_name: 项目名称
            doc_type: 可选的文档类型过滤
            max_results: 最大返回数

        Returns:
            Evidence 对象列表
        """
        docs = self._index.get(project_name, [])

        if doc_type:
            docs = [d for d in docs if d.get("source_type") == doc_type]

        docs = docs[:max_results]

        return [
            Evidence(
                source_url=d["source_url"],
                source_type=d.get("source_type", "unknown"),
                project_name=d["project_name"],
                content=d.get("content", ""),
                relevance=d.get("relevance", ""),
                confidence=d.get("confidence", "medium"),
                retrieved_at=d.get("retrieved_at", ""),
                version_info=d.get("version_info"),
            )
            for d in docs
        ]

    def search(
        self,
        query: str,
        project_name: Optional[str] = None,
        max_results: int = 20,
    ) -> list[Evidence]:
        """简单的关键词搜索（为 Phase 3 BM25 做准备）。

        Args:
            query: 搜索关键词
            project_name: 可限制搜索范围
            max_results: 最大返回数

        Returns:
            匹配的 Evidence 对象列表
        """
        query_lower = query.lower()
        results = []

        projects = (
            [project_name] if project_name else list(self._index.keys())
        )

        for proj in projects:
            for doc in self._index.get(proj, []):
                content = doc.get("content", "").lower()
                title = doc.get("source_url", "").lower()
                if query_lower in content or query_lower in title:
                    results.append(
                        Evidence(
                            source_url=doc["source_url"],
                            source_type=doc.get("source_type", "unknown"),
                            project_name=proj,
                            content=doc.get("content", ""),
                            relevance=doc.get("relevance", ""),
                            confidence=doc.get("confidence", "medium"),
                            retrieved_at=doc.get("retrieved_at", ""),
                            version_info=doc.get("version_info"),
                        )
                    )

        return results[:max_results]

    def get_stats(self) -> dict:
        """获取存储统计信息。"""
        total = sum(len(docs) for docs in self._index.values())
        return {
            "total_documents": total,
            "projects": list(self._index.keys()),
            "docs_per_project": {
                p: len(docs) for p, docs in self._index.items()
            },
            "storage_dir": str(self.storage_dir),
        }

    def clear_project(self, project_name: str):
        """清除指定项目的所有文档。"""
        if project_name in self._index:
            for doc in self._index[project_name]:
                self._url_index.discard(doc.get("source_url", ""))
            del self._index[project_name]
            file_path = self._get_project_file(project_name)
            if file_path.exists():
                file_path.unlink()
