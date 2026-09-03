# RAG 性能与质量压测

项目提供 `project-advisor-rag-benchmark`，用于测量 Hybrid RAG 核心链路：文档分块、Embedding、ChromaDB 持久化、BM25 建索引、向量/BM25 检索和 RRF 融合。

压测会输出五组消融：`bm25_only`、`vector_only`、`hybrid`、`hybrid_rewrite`、`hybrid_rewrite_rerank`。后两组使用确定性的离线 Query Rewrite / Rerank 代理，以便在 CI 中复现结构差异；它们不代表真实 LLM 的质量、延迟、限流和成本，这些仍需通过应用层端到端压测评估。

## 快速运行

先用确定性的 Hash Embedding 检查 ChromaDB 和 BM25 的吞吐、并发与回归：

```powershell
project-advisor-rag-benchmark `
  --documents 10000 `
  --queries 500 `
  --projects 20 `
  --concurrency 1,4,8,16 `
  --repetitions 3 `
  --output artifacts/rag-benchmark-hash.json
```

Hash Embedding 不代表真实语义质量，也不包含生产模型推理开销。测量当前默认模型时使用：

```powershell
project-advisor-rag-benchmark `
  --provider local `
  --model-name all-MiniLM-L6-v2 `
  --documents 10000 `
  --queries 500 `
  --concurrency 1,4,8 `
  --output artifacts/rag-benchmark-minilm.json
```

第一次运行本地模型可能包含模型下载时间。正式记录基线前应先完成模型下载并预热，再在相同机器、相同电源模式和相同后台负载下运行至少 3 次。

## 使用人工标注数据评价检索效果

合成数据只能证明索引和召回链路工作正常。真实的 Recall、MRR 和 nDCG 必须使用独立标注的数据集：

```json
{
  "documents": [
    {
      "id": "doc-checkpoint",
      "project_name": "LangGraph",
      "content": "...",
      "source_url": "https://example.com/checkpoint",
      "source_type": "official_documentation"
    }
  ],
  "queries": [
    {
      "id": "query-checkpoint",
      "query": "如何恢复中断的工作流？",
      "project_name": "LangGraph",
      "relevant_document_ids": ["doc-checkpoint"],
      "rewritten_queries": ["持久化状态 checkpoint 中断恢复"]
    }
  ]
}
```

运行：

```powershell
project-advisor-rag-benchmark `
  --dataset evals/rag_benchmark.json `
  --provider local `
  --model-name all-MiniLM-L6-v2 `
  --top-k 5 `
  --concurrency 1,4,8 `
  --output artifacts/rag-real.json
```

## 如何判断效果好

需要同时看质量、延迟、吞吐、稳定性和索引成本，不能只看 QPS：

- `Recall@K`：所有相关文档中被召回的比例。RAG 首阶段通常最重要，漏召回后 Reranker 无法补救。
- `MRR`：第一个相关结果排名是否靠前，越接近 1 越好。
- `nDCG@K`：综合评价多个相关结果的排序质量。
- `P95/P99 latency`：95%/99% 请求的延迟上界，比平均延迟更能反映用户体验。
- `QPS`：在指定并发下每秒完成的查询数。并发增加后 QPS 应先增长再趋于饱和。
- `error_rate`：超时、并发错误和查询失败率，正常基线应为 0。
- `cold_chunks_per_second`：首次分块、Embedding 和建索引吞吐。
- `storage_bytes_per_chunk`：估算目标数据规模所需磁盘空间。

报告中的 `quality_breakdown` 保留 `vector_only`、`bm25_only` 和 `hybrid` 兼容视图；完整五组结果位于 `ablation.variants`，每组还包含相对 `hybrid` 的 Recall/MRR/nDCG 差值。人工标注数据可以为每个 Query 提供 `rewritten_queries`；未提供时使用内置确定性规则代理。如果某一路质量较高但混合后变差，通常说明 RRF 权重、候选池大小或融合策略需要调整，而不是 Embedding 模型本身失效。

当前默认融合策略为：向量/BM25 权重各 0.5，归一化分数/归一化排名各占 0.5，候选池为 `top_k * 4`，最终结果至少保留每一路的 1 个最高候选。可在人工标注的中文数据集上调整：

```powershell
project-advisor-rag-benchmark `
  --dataset evals/rag_benchmark_zh.json `
  --provider local `
  --vector-weight 0.5 `
  --bm25-weight 0.5 `
  --rank-weight 0.5 `
  --score-weight 0.5 `
  --candidate-pool-factor 4 `
  --min-results-per-source 1 `
  --output artifacts/rag-zh-balanced.json
```

并发负载使用唯一的查询文本，防止进程内 Embedding 缓存把重复请求变成缓存命中，因此负载延迟包含查询 Embedding。模型和索引会在负载阶段前预热。

绝对阈值必须由业务 SLA 和目标硬件决定。建议先建立同机基线，再把以下回归门禁放入 CI：P95 不恶化超过 20%、QPS 不下降超过 20%、Recall@K 下降不超过 0.02、错误率为 0。

```powershell
project-advisor-rag-benchmark `
  --dataset evals/rag_benchmark.json `
  --provider local `
  --baseline artifacts/rag-baseline.json `
  --gate-concurrency 8 `
  --max-p95-regression-pct 20 `
  --max-qps-drop-pct 20 `
  --max-recall-drop 0.02 `
  --fail-on-gate `
  --output artifacts/rag-current.json
```

也可以加入业务绝对门槛：

```powershell
project-advisor-rag-benchmark `
  --min-recall-at-k 0.85 `
  --min-mrr 0.75 `
  --max-p95-ms 500 `
  --max-error-rate 0 `
  --fail-on-gate
```

## 分层压测建议

1. Hash Embedding：隔离测量 ChromaDB、BM25、RRF 和 Python 并发开销。
2. 本地 MiniLM：加入真实 Embedding 推理成本。
3. 人工标注集：评价真实中文/英文查询的检索质量。
4. 应用端到端：通过真实 HTTP/SSE 请求测量查询改写、LLM Reranker、Agent 工具调用和报告生成。
5. 容量阶梯：按 1 万、10 万、50 万、100 万 chunk 逐级运行，观察 P95、QPS、磁盘和建索引吞吐的拐点。

如果中文是主要语料，应重点比较当前英文模型 `all-MiniLM-L6-v2` 与中文/多语言 Embedding 模型的 Recall@K，而不是只比较速度。
