"""文档存储 — JSON 文件持久化，为后续向量检索做准备。

功能：
- 按项目分组存储文档
- 支持增删查
- 自动去重（基于 URL + 内容哈希）
- 元数据过滤（项目、文档类型、日期）
- 导出为 JSON 格式
"""

import json
import hashlib
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from project_advisor.schemas.evidence import Evidence
from project_advisor.rag.evidence_lifecycle import (
    EvidenceLifecyclePolicy,
    classify_evidence,
    is_valid_evidence_url,
    lifecycle_snapshot,
    resolve_current_evidences,
)


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
        self._evidence_index: set[str] = set()
        self._project_files: dict[str, set[Path]] = {}
        self._load_all()

    def _get_project_file(self, project_name: str) -> Path:
        """获取项目的文档文件路径。"""
        safe_name = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in project_name
        ).strip("_") or "project"
        digest = hashlib.sha256(project_name.encode("utf-8")).hexdigest()[:10]
        return self.storage_dir / f"{safe_name[:60]}_{digest}.json"

    def _load_all(self):
        """加载所有已存储的文档。"""
        if not self.storage_dir.exists():
            return

        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                    for doc in docs:
                        try:
                            evidence = Evidence.model_validate(doc)
                        except (TypeError, ValueError):
                            continue
                        project = evidence.project_name
                        if evidence.evidence_id in self._evidence_index:
                            continue
                        normalized = {
                            **evidence.model_dump(),
                            "stored_at": doc.get("stored_at", evidence.retrieved_at),
                        }
                        self._index.setdefault(project, []).append(normalized)
                        self._project_files.setdefault(project, set()).add(file_path)
                        self._evidence_index.add(evidence.evidence_id)
            except (json.JSONDecodeError, IOError):
                continue

    def _save_project(self, project_name: str):
        """保存项目文档到文件。"""
        file_path = self._get_project_file(project_name)
        docs = sorted(
            self._index.get(project_name, []),
            key=lambda item: item.get("retrieved_at", item.get("stored_at", "")),
            reverse=True,
        )
        temporary_path = file_path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp"
        )
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        os.replace(temporary_path, file_path)
        for legacy_path in self._project_files.get(project_name, set()) - {file_path}:
            if legacy_path.exists():
                legacy_path.unlink()
        self._project_files[project_name] = {file_path}

    def add(self, evidence: Evidence) -> bool:
        """添加一条 Evidence 到存储。自动去重。

        Args:
            evidence: 证据对象

        Returns:
            如果新增返回 True，如果已存在返回 False
        """
        if not is_valid_evidence_url(evidence.source_url):
            raise ValueError(f"无效 Evidence URL：{evidence.source_url!r}")
        if evidence.evidence_id in self._evidence_index:
            return False

        doc_dict = evidence.model_dump()
        doc_dict["stored_at"] = datetime.now().isoformat()

        project = evidence.project_name
        if project not in self._index:
            self._index[project] = []

        self._index[project].append(doc_dict)
        self._evidence_index.add(evidence.evidence_id)
        self._save_project(project)
        return True

    def add_batch(self, evidences: list[Evidence]) -> int:
        """批量添加 Evidence。

        Returns:
            新增的文档数量
        """
        count = 0
        changed_projects: set[str] = set()
        for ev in evidences:
            if not is_valid_evidence_url(ev.source_url):
                continue
            if ev.evidence_id in self._evidence_index:
                continue
            doc_dict = ev.model_dump()
            doc_dict["stored_at"] = datetime.now().isoformat()
            self._index.setdefault(ev.project_name, []).append(doc_dict)
            self._evidence_index.add(ev.evidence_id)
            changed_projects.add(ev.project_name)
            count += 1
        for project_name in changed_projects:
            self._save_project(project_name)
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

        return [Evidence.model_validate(d) for d in docs]

    def get_current_by_project(
        self,
        project_name: str,
        *,
        policy: EvidenceLifecyclePolicy | None = None,
        max_results: int = 50,
    ) -> list[Evidence]:
        """Return indexable records: valid, unexpired, and newest per source URL."""
        policy = policy or EvidenceLifecyclePolicy()
        evidences = self.get_by_project(project_name, max_results=100_000)
        eligible = [
            evidence
            for evidence in evidences
            if classify_evidence(evidence, policy)[0] in {"active", "stale"}
        ]
        current, _ = resolve_current_evidences(eligible)
        return current[:max_results]

    def lifecycle_status(
        self,
        project_name: str = "",
        *,
        policy: EvidenceLifecyclePolicy | None = None,
        now: datetime | None = None,
    ) -> dict:
        """Inspect retention state without changing stored records."""
        projects = [project_name] if project_name else list(self._index)
        evidences = [
            Evidence.model_validate(doc)
            for project in projects
            for doc in self._index.get(project, [])
        ]
        snapshot = lifecycle_snapshot(evidences, policy, now=now)
        snapshot["projects"] = projects
        return snapshot

    def maintain(
        self,
        project_name: str = "",
        *,
        policy: EvidenceLifecyclePolicy | None = None,
        dry_run: bool = True,
        now: datetime | None = None,
    ) -> dict:
        """Remove invalid/expired records; callers rebuild affected indexes after apply."""
        policy = policy or EvidenceLifecyclePolicy()
        now = now or datetime.now(timezone.utc)
        projects = [project_name] if project_name else list(self._index)
        removed_ids: list[str] = []
        affected_projects: list[str] = []
        before = self.lifecycle_status(project_name, policy=policy, now=now)

        for project in projects:
            docs = self._index.get(project, [])
            retained: list[dict] = []
            project_removed: list[str] = []
            for doc in docs:
                evidence = Evidence.model_validate(doc)
                status, _ = classify_evidence(evidence, policy, now=now)
                if status in {"expired", "invalid"}:
                    project_removed.append(evidence.evidence_id)
                else:
                    retained.append(doc)
            if not project_removed:
                continue
            affected_projects.append(project)
            removed_ids.extend(project_removed)
            if dry_run:
                continue
            for evidence_id in project_removed:
                self._evidence_index.discard(evidence_id)
            if retained:
                self._index[project] = retained
                self._save_project(project)
            else:
                self.clear_project(project)

        return {
            "dry_run": dry_run,
            "policy": {
                "stale_after_days": policy.stale_after_days,
                "expire_after_days": policy.expire_after_days,
            },
            "before": before,
            "removed_count": len(removed_ids),
            "removed_evidence_ids": removed_ids,
            "affected_projects": affected_projects,
        }

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
                    results.append(Evidence.model_validate(doc))

        results.sort(key=lambda item: item.retrieved_at, reverse=True)
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
                self._evidence_index.discard(doc.get("evidence_id", ""))
            del self._index[project_name]
            project_files = self._project_files.pop(
                project_name, {self._get_project_file(project_name)}
            )
            for file_path in project_files:
                if file_path.exists():
                    file_path.unlink()
