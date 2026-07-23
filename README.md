# AI Project Advisor

一个面向技术团队的开源项目选型系统。它采用“薄 Multi-Agent、厚 Workflow”架构：两个专业 Research Agent 负责取证，一个无工具 Reviewer 负责结构化判断，其余调度、缺口检查、评分和报告均由确定性节点完成。

## 核心能力

- 专业化 Agent：Repository Analyst 和 Documentation Researcher 使用相互隔离的工具集。
- 受限 Reviewer：只能读取规范化 Evidence，不具备工具权限，也不能决定权重和最终排名。
- 证据驱动：统一 `Evidence` 模型记录来源、时间、版本、置信度与评估维度。
- 确定性评分：LLM 负责分析证据，程序按配置权重计算最终分数和排名。
- Hybrid RAG：BM25、向量检索、RRF 融合、查询改写和 Reranker。
- MCP：内置真实 stdio Server，并支持服务端配置额外 stdio / HTTP Server。
- Web 应用：FastAPI + SSE 推送阶段进度，支持停止、复制和下载报告。
- 可观测性：页面展示总耗时、阶段耗时、候选数、引用数、MCP 状态、Token 和成本。
- 离线评测：覆盖检索、引用、任务成功率、延迟、Token 与成本指标。

## 架构

```text
用户需求
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

候选预览可以使用一次结构化模型调用，但不进入运行期 Agent 循环。执行阶段直接复用已确认计划；手动候选模式则由程序生成固定七维计划。每个研究员仍有工具调用上限，补充研究固定最多一轮。

## 快速开始

项目约定使用 Python 3.11 的 `agent` Conda 环境：

```powershell
conda activate agent
C:\miniconda\envs\agent\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

编辑 `.env`，至少配置所选模型对应的 API Key。默认模型为 `deepseek:deepseek-chat`，需要 `DEEPSEEK_API_KEY`。

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
| `GET` | `/api/health` | 服务健康检查 |
| `POST` | `/api/advice/stream` | SSE 技术评估流 |
| `GET` | `/api/evaluation` | 最新离线评测基线 |

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

- JSON `DocumentStore`：按稳定 Evidence ID 去重，同一 URL 的内容发生变化时保留新版本；
- ChromaDB：使用稳定 chunk ID 幂等写入，进程重启后自动发现已有项目 Collection；
- 持久化 BM25：保存分块文本和元数据，重启后恢复关键词索引；
- Hybrid RAG：向量与 BM25 使用同一 chunk ID，经 RRF 合并时不会重复计算同一文档；
- 自动同步：`rag_search` 会先把尚未索引的历史 Evidence 同步到两路索引；
- 手动维护：`rag_ingest` 同步指定项目，`rag_rebuild` 可在分块或嵌入配置变化后强制重建索引。

这里持久化的是研究证据与知识，不是用户对话偏好；跨会话聊天记忆仍需后续接入 LangGraph Checkpointer 或独立记忆服务。

## 运行诊断

每个 SSE `progress` 事件包含当前阶段耗时，最终 `result` 事件包含：

- 总耗时与七个工作流阶段耗时（未触发的补充研究显示为跳过）；
- 候选项目数量和报告中唯一引用 URL 数；
- MCP 的连接状态、Server 数量和工具数量；
- 模型返回的输入、输出与总 Token；
- 按 `INPUT_PRICE_PER_MILLION`、`OUTPUT_PRICE_PER_MILLION` 计算的成本。

只有模型响应提供 usage 元数据时才显示 Token；只有显式配置单价时才显示成本，缺失数据不会被估算成真实值。

## 离线评测

评测数据现在显式记录来源、标注状态和可发布状态，避免把模型运行结果反向复制成标准答案。仓库包含三类文件：

- `evals/sample_results.json`：3 个 Case 的公式测试夹具，状态为 `FIXTURE`。
- `evals/real_results.json`：10 个 Case 的模拟演示数据，状态为 `DEMO`；名称为兼容旧配置而保留，不代表真实运行基线。
- `evals/golden_cases.json`：运行前写好的 6 个真实评测题目、候选项目、相关文档、期望引用和成功标准。当前 ground truth 仍需人工复核。

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

先启动 Web 服务，再按以下闭环执行：

```powershell
# 1. 执行预先定义的 6 个 Case；只采集实际 Evidence、报告、延迟、Token 和成本
C:\miniconda\envs\agent\python.exe scripts\capture_eval_results.py

# 2. 独立人工核对引用支持情况和任务成功标准
C:\miniconda\envs\agent\python.exe scripts\annotate_eval_results.py `
  --input evals\runs\run-时间戳.pending.json `
  --annotator "审核人姓名"

# 3. 仅对通过来源与人工审核门禁的文件生成正式指标
C:\miniconda\envs\agent\python.exe -m project_advisor.evaluation `
  --input evals\runs\run-时间戳.reviewed.json `
  --require-publishable
```

采集脚本不会自动填写 `supported_citations` 或 `task_success`。只有 `real_run`、`reviewed`、`independent_human`、具名审核人且已确认 golden ground truth 的文件，才能标记为 `is_publishable=true`。

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
当前完整回归结果为 `32 passed`。测试覆盖专业化工具隔离、确定性任务展开、单次补充研究门禁、无模型报告渲染、Evidence 重启恢复、BM25/Chroma 持久化、SSE 真实 Evidence 采集和可信评测闭环。

## 故障降级演示

- 将 `MCP_SERVERS_JSON` 指向不可用地址，保持 `MCP_REQUIRED=false`：主流程继续，诊断显示“已降级”。
- 设置 `MCP_REQUIRED=true`：MCP 不可用时任务明确失败。
- 临时移除搜索 API Key：可以演示工具错误如何进入 SSE `error`，且浏览器不会展示服务端堆栈。
- 使用 `EVALUATION_FILE` 指向不存在的文件：评测看板显示不可用，主评估流程不受影响。

## 项目结构

```text
src/project_advisor/
├── agents/          # Planner、Researcher、Reviewer
├── rag/             # Hybrid RAG
├── schemas/         # Evidence、评分与结构化结果
├── static/          # Web 页面、样式和交互
├── tools/           # GitHub、搜索、文档、评分、引用
├── app.py           # FastAPI、SSE、诊断与评测 API
├── graph.py         # LangGraph 主图和子图
├── mcp_client.py    # MCP 动态发现与降级
├── mcp_server.py    # 内置 FastMCP stdio Server
└── evaluation.py    # 离线评测指标与 CLI
```

## 许可证

MIT
