# AI Project Advisor

一个面向技术团队的开源项目选型系统。它使用 LangGraph 编排多个 Agent，联合 GitHub、Web、Hybrid RAG 与 MCP 工具收集证据，最终输出七维评分、风险说明、引用来源和可观测的运行诊断。

## 核心能力

- 多 Agent 工作流：Planner、Repository Analyst、Documentation Researcher、Reviewer 分工协作。
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
Planner ──► Supervisor
               ├──► Repository Analyst ──► GitHub / RAG / MCP
               └──► Documentation Researcher ──► Web / Docs / RAG / MCP
                              │
                              ▼
                    Reviewer（结构化输出）
                              │
                              ▼
                  程序化加权评分 + Markdown 报告
                              │
                              ▼
                 FastAPI / SSE / 运行诊断面板
```

工作流对研究迭代次数和工具调用次数设有上限。MCP 连接失败默认降级，设置 `MCP_REQUIRED=true` 后则中断任务。

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

生产环境应通过密钥管理服务注入认证信息，不要提交真实令牌。浏览器请求不能传入 MCP 命令，外部 Server 只能由服务端环境变量配置。

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

- 总耗时与五个工作流阶段耗时；
- 候选项目数量和报告中唯一引用 URL 数；
- MCP 的连接状态、Server 数量和工具数量；
- 模型返回的输入、输出与总 Token；
- 按 `INPUT_PRICE_PER_MILLION`、`OUTPUT_PRICE_PER_MILLION` 计算的成本。

只有模型响应提供 usage 元数据时才显示 Token；只有显式配置单价时才显示成本，缺失数据不会被估算成真实值。

## 离线评测

项目提供两套评测数据集：

- `evals/sample_results.json` — 3 个固定 Case 的轻量测试夹具，用于快速验证指标公式。
- `evals/real_results.json` — 默认看板使用的 10 个技术选型场景模拟数据。

运行评测：

```powershell
# 演示数据集（K=3）
C:\miniconda\envs\agent\python.exe -m project_advisor.evaluation --input evals\sample_results.json

# 真实场景数据集（K=5）
C:\miniconda\envs\agent\python.exe -m project_advisor.evaluation --input evals\real_results.json --output evals\real_baseline_report.json
```

也可使用安装后的命令：

```powershell
project-advisor-eval --input evals\real_results.json
```

Web 看板默认读取 `evals/real_results.json`，因此启动后会展示 10 个 Case。需要切换数据集时可设置 `EVALUATION_FILE`。

### 真实场景基线（10 Case，`K=5`）

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

真实场景覆盖了前端框架选型、数据库选型、消息队列、Python Web 框架、可观测性、容器编排、搜索引擎、GraphQL vs REST、分布式缓存和 LLM 编排框架共 10 个技术选型方向。其中搜索引擎和 GraphQL vs REST 两个 Case 设为任务失败，反映这两类场景在实际系统中的高难度。

### 演示基线（3 Case，`K=3`）

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

这组数据用于演示指标公式、CLI 和看板接线，不代表线上生产效果。

### 接入真实运行结果

将每次实际运行的检索结果、引用支持判断、usage 和延迟按 `EvaluationCase` Schema 写入 JSON 文件，通过 `EVALUATION_FILE` 环境变量指向该文件，Web 看板将自动展示最新基线。Schema 定义见 [evaluation.py](src/project_advisor/evaluation.py)。

## 测试

```powershell
C:\miniconda\envs\agent\python.exe -m pytest tests -v -p no:cacheprovider
```

MCP 集成测试会实际启动内置 stdio Server、发现工具并调用 `estimate_llm_cost`，不是 Mock 测试。
当前完整回归结果为 `22 passed`，其中新增测试覆盖 Evidence 重启恢复、BM25 持久化、Chroma Collection 恢复及重复入库幂等性。

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
