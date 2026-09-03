# AI Project Advisor

一个面向技术团队的开源项目选型系统。它采用“薄 Multi-Agent、厚 Workflow”架构：两个专业 Research Agent 负责取证，一个无工具 Reviewer 负责结构化判断，其余调度、缺口检查、评分和报告均由确定性节点完成。

## 核心能力

- 专业化 Agent：Repository Analyst 和 Documentation Researcher 使用相互隔离的工具集。
- 有界 Tool Runtime：统一参数校验、超时、瞬时错误重试、单轮 fan-out 上限和结构化执行记录。
- 受限 Reviewer：只能读取规范化 Evidence，不具备工具权限，也不能决定权重和最终排名。
- 证据驱动：统一 `Evidence` 模型记录来源、时间、版本、置信度与评估维度。
- 确定性评分：LLM 负责分析证据，程序按配置权重计算最终分数和排名。
- Hybrid RAG：BM25、向量检索、RRF 融合、查询改写和 Reranker。
- MCP：内置真实 stdio Server，并支持服务端配置额外 stdio / HTTP Server。
- Web 应用：FastAPI + SSE 推送阶段进度，支持停止、恢复、复制和下载报告。
- 可恢复工作流：SQLite 保存 LangGraph checkpoint 和任务状态，支持多轮澄清、候选确认与服务重启后继续。
- 可观测性：页面展示总耗时、阶段耗时、候选数、引用数、MCP 状态、Token 和成本。
- 关联日志：并发运行时通过 ContextVar 隔离 `task_id`、`research_task_id`、候选项目和 Tool Call 上下文。
- 离线评测：覆盖检索、引用、任务成功率、延迟、Token 与成本指标。

## 架构

```text
用户需求 ──► 可选多轮澄清 ──► 候选确认
  │
  ▼
已确认的结构化计划
          │
          ▼
确定性任务展开与强类型路由
     ├──► Repository Analyst ──► GitHub + 仓库 MCP 白名单
     └──► Documentation Researcher ──► Web + 只读 RAG + 文档 MCP 白名单
          │
          ▼
 Evidence 去重、持久化与覆盖检查
     └──► 明确缺口时最多补充一次
          │
          ▼
      无工具 Reviewer（结构化输出）
          │
          ▼
程序化证据绑定、加权排名 + Markdown 模板
                              │
                              ▼
                 FastAPI / SSE / 运行诊断面板
```

候选预览可以使用一次结构化模型调用，但不进入运行期 Agent 循环。执行阶段直接复用已确认计划；手动候选模式会在执行前生成一次结构化需求计划，再用用户指定的候选项目覆盖模型建议，避免丢失语言、部署、预算和硬性能力等约束。每个研究员仍有工具调用上限，补充研究固定最多一轮。

## 快速开始

项目约定使用 Python 3.11 的 `agent` Conda 环境：

```powershell
conda activate agent
C:\miniconda\envs\agent\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

编辑 `.env`，至少配置所选模型对应的 API Key。默认模型为 `deepseek:deepseek-chat`，需要 `DEEPSEEK_API_KEY`。

DeepSeek 和 OpenAI 通过项目内置的纯 HTTPX Chat Completions 客户端接入，
不依赖 OpenAI SDK 的原生 `jiter` 扩展，适用于启用了 Windows App Control
或 WDAC 的环境。升级代码后需要重启 Web 服务，正在运行的旧进程不会自动加载新客户端。

启动 Web 服务：

```powershell
project-advisor-web
```

或：

```powershell
C:\miniconda\envs\agent\python.exe -m uvicorn project_advisor.app:app --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000`。主要接口：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/health` | 服务和模型运行时健康检查 |
| `POST` | `/api/candidates/suggest` | 生成结构化需求与候选预览 |
| `POST` | `/api/advice/stream` | SSE 技术评估流 |
| `GET` | `/api/tasks` | 最近任务列表 |
| `GET` | `/api/tasks/{task_id}` | 任务状态、等待输入与已完成报告 |
| `POST` | `/api/tasks/{task_id}/resume` | 提交澄清/候选确认，或恢复暂停任务 |
| `GET` | `/api/evaluation` | 最新离线评测基线 |

`/api/health` 在 Web 服务可访问但模型运行时缺少配置时返回 `status=degraded`；页面会显示“服务在线 · 模型未配置”，避免把进程存活误报为完整可用。

默认服务只绑定 `127.0.0.1`。如果暴露到局域网或公网，必须配置
`ADVISOR_API_KEY`，并在反向代理层继续启用 TLS 和访问控制。高成本接口还受
单客户端速率限制、单进程全局并发上限、Token 上限和成本上限保护；成本上限
只有在配置输入/输出 Token 单价后才能生效。网页抓取会拒绝私网、回环、保留
地址以及跳转到这些地址的请求，可通过 `WEB_FETCH_DOMAIN_ALLOWLIST` 进一步收窄域名。

## MCP 配置

内置 MCP Server 默认启用，提供两个确定性工具：

- `estimate_llm_cost`：按请求量、Token 和调用方提供的单价计算成本。
- `check_license_policy`：执行保守的许可证策略预检查；结果不构成法律意见。

相关环境变量：

```dotenv
ENABLE_LOCAL_MCP=true
MCP_REQUIRED=false
MCP_CONNECT_TIMEOUT_SECONDS=20
REPOSITORY_MCP_TOOL_ALLOWLIST=
DOCUMENTATION_MCP_TOOL_ALLOWLIST=
MCP_SERVERS_JSON=
```

额外 stdio Server 示例（JSON 需写在一行）：

```dotenv
MCP_SERVERS_JSON={"docs":{"transport":"stdio","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","C:\\\\team-docs"]}}
```

额外 Streamable HTTP Server 示例：

```dotenv
MCP_SERVERS_JSON={"company":{"transport":"streamable_http","url":"https://mcp.example.com/mcp","headers":{"Authorization":"Bearer replace-me"}}}
```

Research Agent 不再接收全部 MCP 工具。只有工具名出现在对应角色的逗号分隔白名单中才会注入；未分类工具默认拒绝。生产环境应通过密钥管理服务注入认证信息，不要提交真实令牌。浏览器请求不能传入 MCP 命令，外部 Server 只能由服务端环境变量配置。

## 持久化知识库

研究工具产生的结构化 `Evidence` 会自动写入 `data/documents`，供后续任务复用。检索层包含：

- JSON `DocumentStore`：按稳定 Evidence ID 去重；同一 URL 内容变化时保留历史版本供审计，检索只使用最新版本；
- ChromaDB：使用稳定 chunk ID 幂等写入，进程重启后自动发现已有项目 Collection；
- 持久化 BM25：保存分块文本和元数据，重启后恢复关键词索引；
- Hybrid RAG：向量与 BM25 使用同一 chunk ID，经 RRF 合并时不会重复计算同一文档；
- 自动同步：`rag_search` 会先把尚未索引的历史 Evidence 同步到两路索引；
- 生命周期：Evidence 分为 active/stale/expired/invalid；expired 和非法 URL 不进入索引；
- 手动维护：`rag_ingest` 同步指定项目，`rag_rebuild` 强制重建；`rag_maintain` 默认只预览，显式 `apply=true` 才清理 expired/invalid 数据并重建受影响索引。

Reviewer 会按来源权威性、新鲜度和置信度选择证据，压缩重复内容，并严格遵守总字符预算。预算和生命周期阈值可通过 `REVIEWER_CONTEXT_MAX_CHARS`、`REVIEWER_EVIDENCE_MAX_CHARS`、`EVIDENCE_STALE_AFTER_DAYS`、`EVIDENCE_EXPIRE_AFTER_DAYS` 配置；裁剪详情写入运行诊断与报告证据缺口。

### 任务与 checkpoint

Web 服务启动时还会初始化两份本地 SQLite 数据：

- `data/checkpoints.sqlite3`：LangGraph 节点状态和中断点，使用 `task_id` 作为 `thread_id`；
- `data/tasks.sqlite3`：任务列表、状态、待处理交互、诊断数据和最终报告。

勾选“允许交互式澄清”后，工作流可以在需求不足时暂停并等待补充，也会在研究前要求确认候选项目。浏览器刷新或服务重启后，可从最近任务列表继续同一个任务。路径可通过 `CHECKPOINT_DB_PATH` 和 `TASK_DB_PATH` 修改。

SQLite 适合本地演示和单机部署；多实例生产部署应替换为共享的 PostgreSQL checkpointer 与任务存储。持久化内容用于恢复任务上下文，不会自动推断或长期保存用户偏好。

## 运行诊断

每个 SSE `progress` 事件包含当前阶段耗时，最终 `result` 事件包含：

- 总耗时与各工作流阶段耗时（未触发的补充研究显示为跳过）；
- 候选项目数量和报告中唯一引用 URL 数；
- MCP 的连接状态、Server 数量和工具数量；
- 模型返回的输入、输出与总 Token；
- Tool 成功/失败/超时数量、重试次数与累计执行耗时；
- Reviewer 上下文预算使用量、重复压缩、截断和低优先级证据丢弃数量；
- 按 `INPUT_PRICE_PER_MILLION`、`OUTPUT_PRICE_PER_MILLION` 计算的成本。

只有模型响应提供 usage 元数据时才显示 Token；只有显式配置单价时才显示成本，缺失数据不会被估算成真实值。

## 离线评测

评测数据现在显式记录来源、标注状态和可发布状态，避免把模型运行结果反向复制成标准答案。仓库包含三类文件：

- `evals/sample_results.json`：3 个 Case 的公式测试夹具，状态为 `FIXTURE`。
- `evals/real_results.json`：10 个 Case 的模拟演示数据，状态为 `DEMO`；名称为兼容旧配置而保留，不代表真实运行基线。
- `evals/golden_cases.json`：运行前写好的 6 个真实评测题目、候选项目、相关文档、期望引用和成功标准。它仍是 draft，发布采集默认拒绝使用。
- `evals/golden_cases.reviewed.json`：由独立审核人逐 Case 核对后生成；不会由程序自动批准或覆盖 draft。

### 模拟演示指标（10 Case，`K=5`）

| 指标 | 结果 |
| --- | ---: |
| Recall@5 | 89.50% |
| Precision@5 | 78.00% |
| MRR | 100.00% |
| nDCG@5 | 90.40% |
| 引用准确率 | 90.32% |
| 引用覆盖率 | 90.00% |
| Task Success Rate | 80.00% |
| P50 / P95 延迟 | 27.0s / 34.7s |
| 平均 Token | 21,220.00 |
| 平均成本 | $0.0792 |

这些数值只用于演示指标、CLI 和看板接线，不能作为线上效果或正式基线发布。看板会明确显示 `DEMO`，并展示数据质量警告数。

运行演示评测：

```powershell
C:\miniconda\envs\agent\python.exe -m project_advisor.evaluation --input evals\real_results.json
```

若命令用于发布流程，请添加 `--require-publishable`；模拟数据、待审核数据和未通过门禁的数据都会被拒绝。

### 生成并审核真实运行数据

先启动 Web 服务，再按以下闭环执行。发布链路默认采用 fail-closed：草稿标签、缺少真实模型、服务降级、没有候选建议预检、没有 checkpoint 恢复、运行错误或缺少人工审核，任一情况都会阻止发布。

```powershell
# 1. 由独立审核人逐项核对 Golden Case；输出新文件，不修改原 draft
C:\miniconda\envs\agent\python.exe scripts\review_golden_cases.py `
  --reviewer "审核人姓名"

# 2. 对 reviewed suite 运行真实 API 验收并采集实际 Evidence、报告、延迟、Token 和成本
#    预检会调用 health、候选建议；首个 Case 会经过 candidate_confirmation 中断与恢复
C:\miniconda\envs\agent\python.exe scripts\capture_eval_results.py `
  --suite evals\golden_cases.reviewed.json

# 3. 人工独立核对生成引用与任务成功标准
C:\miniconda\envs\agent\python.exe scripts\annotate_eval_results.py `
  --input evals\runs\run-时间戳.pending.json `
  --annotator "审核人姓名"

# 4. 校验运行结果绑定的是同一份 reviewed suite，并执行发布质量阈值
C:\miniconda\envs\agent\python.exe scripts\verify_release_acceptance.py `
  --suite evals\golden_cases.reviewed.json `
  --run evals\runs\run-时间戳.reviewed.json `
  --min-recall 0.80 `
  --min-citation-accuracy 0.80 `
  --min-citation-coverage 0.80 `
  --min-task-success 0.80 `
  --output evals\real_baseline_report.json
```

采集脚本不会自动填写 `supported_citations` 或 `task_success`。只有 `real_run`、`reviewed`、`independent_human`、具名审核人、精确 Golden suite SHA-256、健康预检、候选建议和恢复链路全部存在时，文件才能标记为 `is_publishable=true`。发布验收还会检查 Case 集合完全一致及质量阈值。

`--allow-draft-suite`、`--skip-preflight` 和 `--skip-recovery-check` 只用于开发排障；使用这些选项产生的运行不能通过正式发布门禁。API 启用 `ADVISOR_API_KEY` 时，采集脚本会从环境变量读取并通过请求头发送，不会写入评测文件。

真实运行的 SSE 结果包含 `retrieved_evidences`，每条记录使用工作流实际产生的 `evidence_id` 和 `source_url`。评测脚本不再从报告标题推测检索文档，也不再把检索结果复制进相关文档标签。

### 测试夹具指标（3 Case，`K=3`）

| 指标 | 结果 |
| --- | ---: |
| Recall@3 | 77.78% |
| Precision@3 | 66.67% |
| MRR | 83.33% |
| nDCG@3 | 72.09% |
| 引用准确率 | 80.00% |
| 引用覆盖率 | 66.67% |
| Task Success Rate | 66.67% |
| P50 / P95 延迟 | 23.1s / 29.4s |
| 平均 Token | 18,566.67 |
| 平均成本 | $0.062 |

Web 看板默认仍读取 `evals/real_results.json`，因此显示 `DEMO`。将 `EVALUATION_FILE` 指向审核后的运行文件即可展示最新结果；只有通过发布门禁的文件显示 `PUBLISHED`。Schema 和门禁定义见 [evaluation.py](src/project_advisor/evaluation.py)。

## 测试

```powershell
C:\miniconda\envs\agent\python.exe -m pytest tests -v -p no:cacheprovider
```

MCP 集成测试会实际启动内置 stdio Server、发现工具并调用 `estimate_llm_cost`，不是 Mock 测试。
当前完整回归结果为 `73 passed`。测试覆盖专业化工具隔离、Tool 参数校验/超时/重试/fan-out 限制、Agent Run 端到端超时、Reviewer 上下文预算、RAG 五组消融、Evidence 生命周期与版本冲突、Golden Case 独立审核门禁、真实发布预检与套件哈希绑定、异步 RAG、Reranker 并发闸门、日志上下文隔离、领域异常映射、确定性任务展开、单次补充研究门禁、手动候选结构化计划、健康状态降级、无模型报告渲染、Evidence 重启恢复、SQLite checkpoint/任务存储恢复、孤儿任务恢复、SSE 中断恢复、BM25/Chroma 持久化、SSE 真实 Evidence 采集和可信评测闭环。

## 故障降级演示

- 将 `MCP_SERVERS_JSON` 指向不可用地址，保持 `MCP_REQUIRED=false`：主流程继续，诊断显示“已降级”。
- 设置 `MCP_REQUIRED=true`：MCP 不可用时任务明确失败。
- 临时移除搜索 API Key：可以演示工具错误如何进入 SSE `error`，且浏览器不会展示服务端堆栈。
- 使用 `EVALUATION_FILE` 指向不存在的文件：评测看板显示不可用，主评估流程不受影响。
- 如果日志出现 `jiter ... 应用程序控制策略已阻止此文件`：确认已更新到当前代码并重启 Web 服务；
  `/api/health` 的 `model_runtime.client` 应显示 `OpenAICompatibleChatModel`。

## 已知限制

- SQLite checkpoint 和任务库面向单机部署；多实例服务需要共享 checkpointer 与任务存储。
- 长期知识库存储的是可复用 Evidence，不是用户画像；当前没有跨会话偏好记忆。
- Token/成本门禁依赖模型提供 usage；Provider 不返回 usage 时只能执行时间、步骤和并发预算。
- Web 速率限制与并发计数保存在单进程内存中，多实例部署需要由网关或共享存储统一执行。
- Golden Case 已与运行结果分离，但正式发布质量基线仍需要独立人工复核 ground truth。

## 后续演进

1. 为真实 Golden Case 完成人工标注并在 CI 中加入可发布评测门禁。
2. 增加 OpenTelemetry Trace Exporter，把现有 task/node/tool 诊断接入外部可观测平台。
3. 若部署扩展到多实例，再将 SQLite 和进程内限流替换为共享基础设施；当前阶段不提前引入 Redis/PostgreSQL。

## 项目结构

```text
src/project_advisor/
├── api/             # FastAPI 请求 Schema
├── agents/          # Planner、Researcher、Reviewer
├── observability/   # 关联日志、Token/耗时/成本诊断
├── rag/             # Hybrid RAG
├── schemas/         # Evidence、评分与结构化结果
├── static/          # Web 页面、样式和交互
├── tools/           # GitHub、搜索、文档、评分、引用
├── app.py           # FastAPI、SSE、诊断与评测 API
├── graph.py         # LangGraph 主图和子图
├── persistence.py   # SQLite 任务状态存储
├── mcp_client.py    # MCP 动态发现与降级
├── mcp_server.py    # 内置 FastMCP stdio Server
├── evaluation.py    # 离线评测指标与 CLI
└── errors.py        # 领域异常层级
```

## 许可证

[MIT](LICENSE)
