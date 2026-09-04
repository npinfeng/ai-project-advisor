---
name: rag-architecture-review
description: Assess a RAG system's ingestion, chunking, retrieval, reranking, grounding, and evaluation from implementation evidence.
---

# RAG architecture review

Evaluate the pipeline by stages instead of treating “supports RAG” as one capability.

- Identify supported inputs and extraction methods first. Explicitly check whether PDF, Office, HTML, tables, images, and scanned documents use native parsing, OCR, or are unsupported.
- Inspect chunking semantics: separators, token/character units, size, overlap, metadata, stable IDs, and whether parent-child or hierarchical retrieval is actually implemented.
- Verify embedding model, query/document modes, normalization, language fit, caching, vector metric, persistence, and index migration behavior.
- Trace retrieval end to end: filters, vector and lexical candidates, fusion, query rewriting, deduplication, recency handling, reranking, and returned context granularity.
- Look for measurable retrieval evaluation such as Recall@K, MRR, nDCG, latency, throughput, ablations, and representative language/domain datasets.
- State observed facts separately from recommended improvements. Never label a planned or mentioned feature as implemented without code or authoritative documentation evidence.
