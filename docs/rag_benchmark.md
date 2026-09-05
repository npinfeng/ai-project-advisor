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

## 建立真实质量基线

合成数据和 Hash Embedding 只能证明索引、召回及并发链路工作正常，不能形成真实质量基线。正式基线采用固定的真实文档快照、脱敏真实查询和独立人工 0–3 级 qrels。可以复制 `evals/rag_quality_dataset.example.json`，但示例文件本身永远不是基线。

相关性等级约定：`0` 不相关，`1` 有背景帮助，`2` 能部分回答，`3` 直接、完整回答。至少准备 20 条查询，并覆盖实际流量中的中文、英文、中英混合查询及权限、部署、集成、故障处理等业务切片。每条查询既要标相关文档，也应加入容易混淆的 hard negative。

```json
{
  "schema_version": 3,
  "metadata": {
    "name": "enterprise-rag-zh-v1",
    "kind": "real",
    "annotation_status": "reviewed",
    "annotation_method": "independent_human",
    "dataset_author": "dataset-owner",
    "annotator": "independent-reviewer",
    "reviewed_at": "2026-09-05T10:00:00+08:00",
    "snapshot_at": "2026-09-04T10:00:00+08:00",
    "query_source": "deidentified-production-sample",
    "corpus_version": "docs-snapshot-2026-09-04",
    "privacy_status": "reviewed",
    "guideline_version": "qrels-v1"
  },
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
      "language": "zh-CN",
      "category": "state-persistence",
      "project_name": "LangGraph",
      "qrels": [
        {"document_id": "doc-checkpoint", "relevance": 3}
      ],
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
  --model-name BAAI/bge-m3 `
  --top-k 5 `
  --concurrency 1,4,8 `
  --minimum-baseline-queries 20 `
  --min-recall-at-k 0.85 `
  --min-mrr 0.75 `
  --max-error-rate 0 `
  --require-publishable-baseline `
  --fail-on-gate `
  --output artifacts/rag-real.json
```

`--require-publishable-baseline` 使用 fail-closed 门禁。以下任一情况都会以状态码 3 拒绝发布：数据集不是 `real`、不是独立人工审核、整理人与审核人相同、缺少快照/查询来源/隐私审核/规范版本、查询不足、任一查询没有分级 qrels、使用 Hash Embedding、未显式设置 Recall/MRR 门槛，或质量性能门禁失败。报告记录数据集、检索配置和运行环境的 SHA-256 指纹；后续回归只有数据集、分块/融合/Embedding 配置和运行环境一致时才可比较。

## 如何判断效果好

需要同时看质量、延迟、吞吐、稳定性和索引成本，不能只看 QPS：

- `Recall@K`：所有相关文档中被召回的比例。RAG 首阶段通常最重要，漏召回后 Reranker 无法补救。
- `MRR`：第一个相关结果排名是否靠前，越接近 1 越好。
- `nDCG@K`：综合评价多个相关结果的排序质量。
- `MAP@K`：评价多个相关文档是否持续排在前面，而非只命中第一条。
- `slices.language/category`：分别观察中文和各业务类别，防止总体均值掩盖局部退化。
- `missed_query_ids` 与 `queries`：定位零召回查询及其实际返回文档，供错误分析和下一轮 hard-negative 补充。
- `confidence_intervals_95`：基于查询级指标的固定种子 Bootstrap 区间；样本较少时不要只比较一个均值小数。
- `P95/P99 latency`：95%/99% 请求的延迟上界，比平均延迟更能反映用户体验。
- `QPS`：在指定并发下每秒完成的查询数。并发增加后 QPS 应先增长再趋于饱和。
- `error_rate`：超时、并发错误和查询失败率，正常基线应为 0。
- `cold_chunks_per_second`：首次分块、Embedding 和建索引吞吐。
- `storage_bytes_per_chunk`：估算目标数据规模所需磁盘空间。

报告中的 `quality_breakdown` 保留 `vector_only`、`bm25_only` 和 `hybrid` 兼容视图；完整五组结果位于 `ablation.variants`，每组还包含相对 `hybrid` 的 Recall/MRR/nDCG 差值。主质量报告同时输出逐查询结果、语言/类别切片、MAP 和零召回列表。人工标注数据可以为每个 Query 提供 `rewritten_queries`；未提供时使用内置确定性规则代理。如果某一路质量较高但混合后变差，通常说明融合权重或候选池需要调整。

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
  --require-publishable-baseline `
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
3. 独立人工分级标注集：评价真实中文/英文查询的检索与排序质量，并建立不可混用的内容指纹。
4. 应用端到端：通过真实 HTTP/SSE 请求测量查询改写、LLM Reranker、Agent 工具调用和报告生成。
5. 容量阶梯：按 1 万、10 万、50 万、100 万 chunk 逐级运行，观察 P95、QPS、磁盘和建索引吞吐的拐点。

如果中文是主要语料，应重点比较当前英文模型 `all-MiniLM-L6-v2` 与中文/多语言 Embedding 模型的 Recall@K，而不是只比较速度。
