"""文档分块器 — 智能分割技术文档，保留语义边界。

使用 LangChain 的 RecursiveCharacterTextSplitter，
支持按标题、段落和句子边界进行智能分块。
"""

from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 默认分块配置
DEFAULT_CHUNK_SIZE = 1000  # 每个 chunk 的字符数
DEFAULT_CHUNK_OVERLAP = 200  # chunk 之间的重叠字符数


class DocumentChunker:
    """技术文档智能分块器。

    使用递归字符分割，按照 Markdown 标题 → 段落 → 句子 → 字符
    的优先级进行切割，尽可能保留语义完整性。
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """初始化分块器。

        Args:
            chunk_size: 每个 chunk 的目标字符数
            chunk_overlap: 相邻 chunk 的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ",  # Markdown H2
                "\n### ",  # Markdown H3
                "\n#### ",  # Markdown H4
                "\n\n",  # 段落
                "\n",  # 行
                ". ",  # 句子
                " ",  # 词
                "",  # 字符
            ],
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_document(
        self, content: str, metadata: Optional[dict] = None
    ) -> list[dict]:
        """将单篇文档切分为多个 chunk。

        Args:
            content: 文档原始文本
            metadata: 文档元数据（URL、项目名、类型、版本等）

        Returns:
            chunk 列表，每个包含 text 和 metadata
        """
        if metadata is None:
            metadata = {}

        chunks = self._splitter.create_documents(
            texts=[content],
            metadatas=[metadata],
        )

        return [
            {"text": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        ]

    def chunk_documents(
        self, documents: list[dict]
    ) -> list[dict]:
        """批量切分多篇文档。

        Args:
            documents: 文档列表，每个包含 content 和 metadata

        Returns:
            所有 chunk 的列表
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(
                content=doc.get("content", ""),
                metadata={
                    "source_url": doc.get("source_url", ""),
                    "source_type": doc.get("source_type", ""),
                    "project_name": doc.get("project_name", ""),
                    "version_info": doc.get("version_info", ""),
                    "retrieved_at": doc.get("retrieved_at", ""),
                    **(doc.get("metadata", {})),
                },
            )
            all_chunks.extend(chunks)
        return all_chunks
