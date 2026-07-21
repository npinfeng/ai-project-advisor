# AI 技术选型与开源项目评估 Agent

基于 [Open Deep Research](https://github.com/langchain-ai/open_deep_research) 和 LangGraph 构建的多智能体系统，面向开发者和技术团队的 AI 技术选型与开源项目评估工具。

## 功能

- 🔍 **需求解析**：将自然语言需求转换为结构化技术约束
- 📊 **GitHub 分析**：自动获取仓库 Stars、Release、Issue 和维护状态
- 📚 **文档搜索**：搜索官方文档和技术博客确认功能特性
- 🏆 **多维度评分**：7 个维度加权评分，程序计算，LLM 定性分析
- 📝 **结构化报告**：生成带引用溯源的完整技术选型报告
- ⚡ **实时交互界面**：FastAPI + SSE 推送 Agent 执行进度，支持报告复制和下载

## 架构

```
用户输入 → Planner → Supervisor → {并行研究员} → Reviewer → 最终报告
                              ├→ Repository Analyst (GitHub数据)
                              └→ Documentation Researcher (文档搜索)
```

### 4 个核心 Agent

| Agent | 职责 |
|-------|------|
| Planner | 需求解析、候选项目确定、评估维度设计 |
| Repository Analyst | GitHub 仓库分析（Stars/Release/Issue/维护状态） |
| Documentation Researcher | 官方文档搜索、技术能力确认 |
| Reviewer | 证据汇总、评分、冲突检测、报告生成 |

## 快速开始

### 1. 安装依赖

```bash
cd ai-project-advisor
pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 API Keys
```

### 3. 启动 Web 界面

```bash
conda activate agent
project-advisor-web
```

浏览器访问 `http://127.0.0.1:8000`，输入技术需求和候选项目，即可实时查看各 Agent 的执行进度与评分报告。

也可以直接运行：

```bash
python -m uvicorn project_advisor.app:app --host 127.0.0.1 --port 8000
```

### 4. 运行 Python Demo

```python
from project_advisor.graph import graph

# 定义技术选型问题
result = await graph.ainvoke({
    "messages": [{
        "role": "user",
        "content": "我要开发一个支持 MCP、RAG 和人工审批的 Python 多智能体系统，"
                   "应该选择 LangGraph、CrewAI 还是 Microsoft Agent Framework？"
    }]
})

print(result["final_report"])
```

### 5. 运行测试

```bash
pytest tests/ -v
```

## 项目结构

```
ai-project-advisor/
├── src/project_advisor/
│   ├── graph.py              # 主 LangGraph 工作流
│   ├── state.py              # 状态定义
│   ├── configuration.py      # 配置管理
│   ├── app.py                # FastAPI + SSE 服务
│   ├── prompts.py            # 系统提示词
│   ├── utils.py              # 工具函数
│   ├── agents/               # Agent 实现
│   │   ├── planner.py
│   │   ├── repository_analyst.py
│   │   ├── documentation_researcher.py
│   │   └── reviewer.py
│   ├── tools/                # 工具模块
│   │   ├── github.py         # GitHub API
│   │   ├── search.py         # 搜索工具
│   │   ├── scoring.py        # 评分引擎
│   │   └── citations.py      # 引用验证
│   ├── schemas/              # 数据模型
│   │   └── evidence.py
│   └── static/               # Web 前端界面
├── tests/
│   └── test_graph.py
├── pyproject.toml
└── README.md
```

## 评分模型

| 维度 | 权重 | 数据来源 |
|------|------|---------|
| 功能匹配度 | 30% | LLM 分析 + 文档证据 |
| 工程可靠性 | 20% | GitHub Release/Issue 数据 |
| 社区与维护状态 | 15% | GitHub Stars/Contributors |
| 文档和示例质量 | 10% | LLM 分析 |
| 学习成本 | 10% | LLM 评估 |
| 扩展能力 | 10% | LLM 分析架构 |
| 部署和运行成本 | 5% | LLM 分析 |

## 许可

MIT
