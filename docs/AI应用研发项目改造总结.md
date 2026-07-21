# AI Project Advisor 项目改造总结

## 1. 文档目的

本文档汇总 `AI Project Advisor` 为面向 **AI 应用研发工程师面试** 所做的改造，说明项目当前具备的能力、关键技术实现、可用于面试表达的亮点，以及仍未完成或尚未验证的部分。

项目定位已经从“调用大模型生成技术选型建议的 Demo”，升级为一个具备以下要素的 AI 应用工程项目：

- 基于 LangGraph 的多 Agent 工作流
- GitHub、Web、RAG 和 MCP 多工具协作
- 结构化输出与程序化评分
- FastAPI + SSE 实时服务
- 可交互 Web 前端
- 离线评测指标与命令行入口
- 测试和工程化配置

## 2. 当前项目定位

项目面向技术团队的开源项目选型场景。用户输入业务需求、技术约束和候选项目后，系统通过多个 Agent 并行收集 GitHub 数据、官方文档、社区资料和本地 RAG 证据，最后生成带评分、风险说明和引用来源的技术选型报告。

核心流程为：

```text
用户需求
  → 需求澄清
  → 评估计划
  → Supervisor 分派研究任务
  → Repository Analyst / Documentation Researcher 并行研究
  → Reviewer 结构化评分
  → 最终 Markdown 报告
```

## 3. 已完成的核心改造

### 3.1 修通 LangGraph 多 Agent 工作流

对原有工作流进行了拓扑和状态流转修复，确保 Supervisor 和 Researcher 子图可以形成真实的 Agent 工具调用循环，而不是仅按固定顺序执行节点。

主要改造包括：

- Supervisor 可以根据研究任务持续分派 Researcher。
- Repository Analyst 和 Documentation Researcher 可以调用工具后返回 Agent 继续判断。
- 研究达到终止条件后进入压缩和汇总节点。
- 修复 `ResearchComplete` 被当作普通工具执行的问题。
- 保留最大研究迭代次数和最大工具调用次数，避免工作流无限循环。

相关文件：

- `src/project_advisor/graph.py`
- `src/project_advisor/state.py`
- `src/project_advisor/agents/repository_analyst.py`
- `src/project_advisor/agents/documentation_researcher.py`

### 3.2 Reviewer 改为结构化输出

Reviewer 不再直接生成一段无法校验的自由文本，而是通过 Pydantic 模型输出 `ReviewResult`：

- 每个候选项目对应一个 `ProjectScore`。
- 每个项目包含七个评分维度。
- Reviewer 单独输出总体分析和证据缺口。
- LLM 负责证据分析和给出各维度建议分。
- 程序按照配置权重计算最终加权总分并排序。

七个评分维度为：

| 维度 | 默认权重 |
| --- | ---: |
| 功能匹配度 | 30% |
| 工程可靠性 | 20% |
| 社区与维护状态 | 15% |
| 文档与示例质量 | 10% |
| 学习成本 | 10% |
| 扩展能力 | 10% |
| 部署和运行成本 | 5% |

这种设计将“LLM 的定性判断”和“程序的确定性计算”分开，更容易解释、测试和调整。

相关文件：

- `src/project_advisor/agents/reviewer.py`
- `src/project_advisor/schemas/evidence.py`
- `src/project_advisor/tools/scoring.py`

### 3.3 建立统一证据模型

项目已经定义统一的 `Evidence` 数据结构，用于描述不同来源的研究证据，包括：

- 来源 URL
- 来源类型
- 所属候选项目
- 证据内容
- 对应评估维度
- 置信度
- 抓取时间
- 版本信息

同时实现了引用验证、来源新鲜度检查和证据冲突检测，使报告不只是“模型说了什么”，还可以说明“结论来自哪里”。

相关文件：

- `src/project_advisor/schemas/evidence.py`
- `src/project_advisor/tools/citations.py`

### 3.4 完善 GitHub、文档与 Hybrid RAG 工具链

目前工具体系包括：

- GitHub 仓库基本信息
- Release 数据
- Issue 数据
- README 内容
- 单 URL 文档抓取
- 批量文档抓取
- RAG 文档入库
- RAG 检索
- RAG 状态查询
- Tavily 或 DuckDuckGo Web 搜索
- Agent 反思工具

RAG 部分包含：

- 文档采集和标准化
- 文档分块
- BM25 关键词检索
- 向量检索
- Reciprocal Rank Fusion 融合
- Reranker
- 查询改写

相关目录：

- `src/project_advisor/tools/`
- `src/project_advisor/rag/`

### 3.5 新增 FastAPI + SSE 服务

项目已经从纯 Python 调用升级为可访问的 Web 服务：

- `GET /` 返回前端页面。
- `GET /api/health` 提供服务健康检查。
- `POST /api/advice/stream` 接收评估请求。
- 使用 SSE 持续推送 Agent 阶段进度。
- 最终返回结构化评分和 Markdown 报告。
- 支持客户端主动停止任务。

SSE 事件包括：

- `started`：任务开始。
- `progress`：工作流节点完成。
- `result`：返回最终评分和报告。
- `error`：返回可展示的错误信息。

相关文件：

- `src/project_advisor/app.py`

### 3.6 新增完整 Web 交互界面

前端已经实现以下交互：

- 输入技术需求。
- 输入候选项目。
- 一键填充 Agent 框架、RAG、LLM 可观测性示例。
- 实时显示字符数。
- 实时展示五个工作流阶段。
- 显示候选项目加权得分和进度条。
- 渲染 Markdown 标题、列表和表格。
- 复制最终报告。
- 下载 Markdown 报告。
- 停止正在执行的任务。
- 展示服务在线或离线状态。
- 支持桌面端和移动端布局。

相关文件：

- `src/project_advisor/static/index.html`
- `src/project_advisor/static/styles.css`
- `src/project_advisor/static/app.js`

### 3.7 修复项目打包和命令行入口

项目使用 `src` 目录布局，已调整 setuptools 包发现规则，并将静态资源纳入 package data。

当前命令行入口包括：

```text
project-advisor-web   启动 Web 服务
project-advisor-mcp   启动内置 MCP Server
project-advisor-eval  运行离线评测
```

相关文件：

- `pyproject.toml`
- `.gitignore`

## 4. 真实 MCP 接入情况

### 4.1 MCP Server

项目已经新增基于官方 Python MCP SDK `FastMCP` 的真实 stdio Server，而不是仅在 Prompt 中提到 MCP。

内置 MCP Server 提供两个确定性工程工具：

#### `estimate_llm_cost`

输入请求量、平均输入输出 Token 和模型单价，计算：

- 单次请求成本
- 月度总成本
- 月度输入 Token
- 月度输出 Token
- 成本估算假设

#### `check_license_policy`

根据许可证、商业使用和闭源分发要求，输出：

- 风险等级
- 策略判断
- 判断原因
- 法务免责声明

相关文件：

- `src/project_advisor/mcp_server.py`

### 4.2 MCP Client

项目已经新增 `MultiServerMCPClient`，支持：

- 默认连接项目内置 stdio MCP Server。
- 使用当前 Python 解释器启动 Server，保证虚拟环境一致。
- 动态加载 MCP Tool 并转换为 LangChain Tool。
- 支持工具缓存。
- 支持通过 `MCP_SERVERS_JSON` 配置额外 stdio 或 HTTP MCP Server。
- 支持 MCP 连接失败后降级，或配置为必须成功。
- 不允许用户从前端直接传入任意 MCP 启动命令，避免命令执行风险。

相关文件：

- `src/project_advisor/mcp_client.py`
- `src/project_advisor/configuration.py`

### 4.3 MCP 动态注册

当前 `get_all_tools()` 已加入 MCP 工具加载逻辑。Researcher 获取工具列表时，会将 MCP Tool 与 GitHub、搜索、RAG 工具合并。因此 MCP 已进入 Agent 可选择、可调用的真实工具链路。

相关文件：

- `src/project_advisor/utils.py`

### 4.4 MCP 当前验证状态

已经新增真实 stdio MCP 集成测试，测试目标为：

1. 启动内置 MCP Server。
2. 通过 `MultiServerMCPClient` 获取工具。
3. 找到 `estimate_llm_cost`。
4. 实际调用工具。
5. 校验返回的月度成本和 Token 数据。

但是在用户要求暂停继续修改时，**本轮新增的 MCP 动态注册和集成测试尚未在 `agent` 环境中执行最终验证**。因此当前准确表述应为：

> MCP 已按真实协议完成 Server、Client、动态工具注册和调用测试代码，但最后一次完整测试尚未运行。

相关测试：

- `tests/test_mcp_and_evaluation.py`

## 5. 独立系统评测指标

项目原有的七维评分用于比较候选开源项目，不等于 AI 系统自身的效果评测。本次新增了独立的离线评测模块，用于衡量检索、引用、任务完成率、延迟和成本。

### 5.1 检索指标

#### Recall@K

衡量相关文档中有多少被 Top-K 检索结果召回：

```text
Recall@K = Top-K 中相关文档数量 / 全部相关文档数量
```

#### Precision@K

衡量 Top-K 检索结果中相关文档的比例：

```text
Precision@K = Top-K 中相关文档数量 / K
```

#### MRR

衡量第一个相关结果出现的位置：

```text
MRR = 所有 Case 的第一个相关结果排名倒数的平均值
```

#### nDCG@K

衡量相关结果是否排在更靠前的位置，并以理想排序进行归一化。

### 5.2 引用指标

#### Citation Accuracy

衡量生成引用中有多少确实支持报告结论：

```text
引用准确率 = 受支持的生成引用数 / 全部生成引用数
```

#### Citation Coverage

衡量期望引用的关键来源有多少被最终报告覆盖：

```text
引用覆盖率 = 已覆盖的期望引用数 / 全部期望引用数
```

### 5.3 端到端指标

- Task Success Rate：成功完成完整任务的 Case 比例。
- P50 Latency：中位请求延迟。
- P95 Latency：长尾请求延迟。
- Average Tokens：平均输入和输出 Token 总量。
- Average Cost：平均单个 Case 的模型成本。

### 5.4 评测数据和 CLI

已新增包含三个示例 Case 的评测数据：

- Agent 框架选型
- RAG 技术栈选型
- LLM 可观测性平台选型

计划使用方式：

```powershell
C:\miniconda\envs\agent\python.exe -m project_advisor.evaluation --input evals\sample_results.json
```

或者安装后执行：

```powershell
project-advisor-eval --input evals\sample_results.json
```

相关文件：

- `src/project_advisor/evaluation.py`
- `evals/sample_results.json`
- `tests/test_mcp_and_evaluation.py`

### 5.5 评测当前验证状态

评测模型、指标公式、示例数据、CLI 和单元测试均已写入项目，但在暂停修改前，**尚未执行本轮评测测试和 CLI 验证**。

## 6. 测试情况

### 6.1 上一阶段已验证

上一阶段曾在指定虚拟环境中完成测试：

```text
C:\miniconda\envs\agent\python.exe
Python 3.11.15
12 passed
```

已覆盖的主要内容包括：

- Graph 编译。
- Supervisor 和 Researcher 子图循环。
- Pydantic Schema。
- 七维评分引擎。
- 引用与冲突检测。
- GitHub URL 解析。
- 文档采集。
- 文档存储。
- RAG 核心模块。
- FastAPI 页面和 SSE 流。
- Reviewer 结构化输出。

### 6.2 本轮尚未验证

暂停前新增但还未运行的测试包括：

- MCP stdio Server 实际连接。
- MCP Tool 实际调用。
- 离线评测公式。
- 示例评测文件加载。
- MCP 工具加入 `get_all_tools()` 后的完整回归测试。

因此不能把此前的 `12 passed` 当作当前最新代码的最终测试结论。

## 7. 指定运行环境

用户指定的唯一运行环境为：

```text
虚拟环境名称：agent
Python 路径：C:\miniconda\envs\agent\python.exe
Python 版本：3.11.15
```

后续安装、测试和启动建议显式使用该解释器：

```powershell
C:\miniconda\envs\agent\python.exe -m pip install -e .
C:\miniconda\envs\agent\python.exe -m pytest tests -v
C:\miniconda\envs\agent\python.exe -m uvicorn project_advisor.app:app --host 127.0.0.1 --port 8000
```

不要使用 base 环境执行项目命令，以免出现依赖版本和 MCP 子进程解释器不一致。

## 8. 适合面试重点讲解的内容

### 8.1 为什么使用 LangGraph

可以重点说明：

- 技术选型是一个多阶段、长链路、有状态的任务。
- 需要显式控制 Agent 分工、循环、终止条件和状态聚合。
- LangGraph 比简单 Chain 更适合实现可控的多 Agent 工作流。
- 最大迭代次数和工具调用次数用于控制成本及防止死循环。

### 8.2 如何降低 LLM 不确定性

可以从以下四层回答：

1. 使用 Pydantic 结构化输出约束 Reviewer。
2. 使用 GitHub、官方文档、RAG 和 MCP 提供外部事实。
3. 使用程序计算最终加权总分。
4. 使用引用准确率、引用覆盖率和 Task Success Rate 进行离线评测。

### 8.3 MCP 为什么是真实接入

可以说明完整调用链：

```text
Researcher
  → get_all_tools
  → MultiServerMCPClient
  → stdio 启动 FastMCP Server
  → tools/list 获取工具描述
  → tools/call 执行成本或许可证检查
  → 返回 LangChain Tool 结果
  → Agent 继续推理
```

这与“在 Prompt 中告诉模型支持 MCP”有本质区别：项目中存在独立 Server、Client、协议通信、动态工具发现和真实工具执行。

### 8.4 如何评价系统效果

可以将指标分成三层：

| 层级 | 指标 | 解决的问题 |
| --- | --- | --- |
| 检索层 | Recall@K、Precision@K、MRR、nDCG@K | 是否找到了正确证据，排序是否合理 |
| 生成层 | 引用准确率、引用覆盖率 | 报告引用是否真实、是否充分 |
| 系统层 | Task Success、P50/P95、Token、Cost | 系统能否完成任务，性能和成本如何 |

七维项目评分属于业务输出，不能代替上述 AI 系统评测。

### 8.5 工程化与安全设计

可以强调：

- FastAPI 提供稳定服务边界。
- SSE 适合长任务的实时进度反馈，实现复杂度低于 WebSocket。
- MCP 外部连接只能通过服务端环境变量配置。
- 不允许浏览器直接传任意 MCP command，避免远程命令执行。
- MCP 可配置为失败降级或强制可用。
- API Key 通过环境变量管理。
- 工具调用、Agent 迭代和输入长度均有限制。

## 9. 当前尚未完成的部分

用户要求暂停后，以下计划尚未继续实现：

### 9.1 前端运行诊断卡

原计划在现有前端展示：

- 总耗时
- 各工作流阶段耗时
- 候选项目数量
- 引用 URL 数量
- MCP 工具数量和连接状态
- Token 使用量
- 成本估算

当前前端仍主要展示 Agent 时间线、候选项目评分和最终报告，尚未加入上述运行诊断卡。

### 9.2 前端离线评测看板

原计划增加独立评测区域，展示：

- Recall@K
- MRR
- nDCG@K
- 引用准确率
- 引用覆盖率
- Task Success Rate
- P50/P95 延迟
- 平均 Token 和成本

当前评测模块只有 Python 模型、JSON 样例和 CLI，尚未接入 Web 页面。

### 9.3 配置和 README 更新

原计划继续补充：

- `.env.example` 中的 MCP 配置示例。
- README 中的 MCP 架构说明。
- 外部 MCP Server 配置示例。
- 离线评测 CLI 使用说明。
- 最新测试结果。

这些内容尚未完成，本文档暂时承担改造说明作用。

### 9.4 最终回归验证

需要在 `agent` 环境中重新执行：

```powershell
C:\miniconda\envs\agent\python.exe -m pytest tests -v
C:\miniconda\envs\agent\python.exe -m project_advisor.evaluation --input evals\sample_results.json
```

验证通过后，才能将当前版本标记为完整可交付状态。

## 10. 当前完成度判断

从 AI 应用研发工程师面试项目的角度，当前项目已经具备较完整的技术深度：

- 有真实业务问题，而非纯聊天机器人。
- 有多 Agent 编排和状态管理。
- 有 GitHub、Web、RAG、MCP 多类工具。
- 有结构化输出、评分引擎和证据体系。
- 有后端 API、流式交互和完整前端。
- 有独立离线评测指标和测试基础。
- 有安全、降级、成本和可观测性意识。

当前最需要补齐的不是继续堆叠新框架，而是：

1. 完成最新代码的全量测试。
2. 将运行指标和离线评测结果接入前端。
3. 准备一组可复现的真实评测集和基线结果。
4. 在 README 中形成清晰的架构图、启动步骤和效果数据。
5. 准备面试时可演示的固定场景和故障降级场景。

完成以上收尾后，该项目可以作为一个较有说服力的 AI 应用研发面试项目。

## 11. 文件改造清单

| 文件或目录 | 主要作用 |
| --- | --- |
| `src/project_advisor/graph.py` | LangGraph 主图和子图编排 |
| `src/project_advisor/state.py` | 工作流状态和控制模型 |
| `src/project_advisor/agents/` | Planner、Researcher、Reviewer 实现 |
| `src/project_advisor/schemas/evidence.py` | 需求、证据、评分和结构化输出模型 |
| `src/project_advisor/tools/` | GitHub、搜索、文档、评分、引用工具 |
| `src/project_advisor/rag/` | Hybrid RAG 实现 |
| `src/project_advisor/app.py` | FastAPI 和 SSE 服务 |
| `src/project_advisor/static/` | Web 交互界面 |
| `src/project_advisor/mcp_server.py` | 内置 FastMCP Server |
| `src/project_advisor/mcp_client.py` | MCP Client 和动态工具加载 |
| `src/project_advisor/evaluation.py` | 独立离线评测指标和 CLI |
| `evals/sample_results.json` | 示例评测数据 |
| `tests/test_workflow_and_web.py` | 工作流和 Web 测试 |
| `tests/test_mcp_and_evaluation.py` | MCP 与离线评测测试 |
| `pyproject.toml` | 依赖、打包和命令行入口 |

## 12. 总结

本轮改造的核心价值，是让项目从“模型生成一份推荐报告”升级为“有工作流、有工具协议、有证据、有评测、有服务和有交互的 AI 应用系统”。

目前真实 MCP 和独立评测代码已经进入项目，但最新改动仍需在 `agent` 环境执行最终验证；前端运行诊断和评测看板尚未实现。面试时应如实说明完成边界，不要把尚未验证的功能描述成已经稳定运行。
