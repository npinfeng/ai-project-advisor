"""文档采集与索引管道 — 从 Evidence 到可检索的 RAG 索引。

完整流程：
1. 从 DocumentStore 读取 Evidence 文档
2. DocumentChunker 分块
3. Embedder 向量化
4. HybridRetriever 构建 BM25 + 向量索引
5. 返回索引统计信息
"""

from typing import Optional

from project_advisor.rag.chunker import DocumentChunker
from project_advisor.rag.embedder import Embedder
from project_advisor.rag.hybrid_retriever import HybridRetriever


class IngestionPipeline:
    """文档采集与索引管道。

    将收集到的 Evidence 文档转化为可检索的 RAG 索引。
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        chunker: Optional[DocumentChunker] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
    ):
        self.embedder = embedder or Embedder()
        self.chunker = chunker or DocumentChunker()
        self.hybrid = hybrid_retriever or HybridRetriever(
            embedder=self.embedder,
            chunker=self.chunker,
        )

    def ingest_evidence(
        self,
        project_name: str,
        evidences: list,
    ) -> dict:
        """将 Evidence 对象列表摄入到 RAG 索引。

        Args:
            project_name: 项目名称
            evidences: Evidence 对象列表

        Returns:
            索引统计信息
        """
        documents = []
        for ev in evidences:
            documents.append({
                "evidence_id": ev.evidence_id,
                "content": ev.content,
                "source_url": ev.source_url,
                "source_type": ev.source_type,
                "project_name": ev.project_name,
                "version_info": ev.version_info,
                "retrieved_at": ev.retrieved_at,
            })

        return self.hybrid.index_documents(project_name, documents)

    def ingest_from_store(
        self,
        store,
        project_name: str,
    ) -> dict:
        """从 DocumentStore 读取文档并索引。

        Args:
            store: DocumentStore 实例
            project_name: 项目名称

        Returns:
            索引统计信息
        """
        evidences = store.get_by_project(project_name)
        return self.ingest_evidence(project_name, evidences)

    def search(
        self,
        query: str,
        project_name: Optional[str] = None,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """搜索已索引的文档。

        Args:
            query: 查询
            project_name: 项目名称
            top_k: 返回数量
            filter_metadata: 元数据过滤

        Returns:
            搜索结果
        """
        return self.hybrid.search(
            query=query,
            project_name=project_name,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
