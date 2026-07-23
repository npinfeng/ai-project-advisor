"""文档分块器 — 按标题、段落和句子边界递归切分技术文档。"""

import hashlib

from typing import Optional

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
        if chunk_size < 1:
            raise ValueError("chunk_size 必须大于 0。")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size。")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = [
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]

    def _segments(self, text: str, separator_index: int = 0) -> list[str]:
        """Recursively split oversized text while retaining separators."""
        if len(text) <= self.chunk_size:
            return [text]
        if separator_index >= len(self.separators):
            return [
                text[index:index + self.chunk_size]
                for index in range(0, len(text), self.chunk_size)
            ]

        separator = self.separators[separator_index]
        if not separator:
            return list(text)
        if separator not in text:
            return self._segments(text, separator_index + 1)

        raw_parts = text.split(separator)
        parts = [raw_parts[0], *[separator + part for part in raw_parts[1:]]]
        segments: list[str] = []
        for part in parts:
            if not part:
                continue
            segments.extend(self._segments(part, separator_index + 1))
        return segments

    def _merge_segments(self, segments: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""
        for segment in segments:
            if len(current) + len(segment) <= self.chunk_size:
                current += segment
                continue
            if current.strip():
                chunks.append(current.strip())
            overlap = current[-self.chunk_overlap:] if self.chunk_overlap else ""
            current = overlap + segment
            while len(current) > self.chunk_size:
                chunks.append(current[:self.chunk_size].strip())
                start = self.chunk_size - self.chunk_overlap
                current = current[start:]
        if current.strip():
            chunks.append(current.strip())
        return chunks

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

        if not content.strip():
            return []
        chunks = self._merge_segments(self._segments(content))
        results = []
        for index, chunk in enumerate(chunks):
            source_url = str(metadata.get("source_url", ""))
            project_name = str(metadata.get("project_name", ""))
            identity = "\x1f".join([project_name, source_url, chunk])
            chunk_id = f"chunk_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"
            chunk_metadata = {
                **metadata,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "content_hash": hashlib.sha256(
                    chunk.encode("utf-8")
                ).hexdigest(),
            }
            results.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": chunk_metadata,
            })
        return results

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
                    "evidence_id": doc.get("evidence_id", ""),
                    **(doc.get("metadata", {})),
                },
            )
            all_chunks.extend(chunks)
        return all_chunks
