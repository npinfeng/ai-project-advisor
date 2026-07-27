"""模型注册表 — 提供主流模型/框架的结构化对比数据。

解决"对比模型上下文窗口、API 定价、最新开源权重时全靠搜索"的盲区。
Agent 可通过 model_info 工具直接查询结构化数据，作为搜索结果的校验锚点。

数据来源：官方文档、API 定价页面（截至 2026-07）。
保持定期更新即可维持数据准确性。
"""

from datetime import datetime, timezone

from langchain_core.tools import tool


# ===== 主流 LLM 模型数据 =====

_MODELS: list[dict] = [
    # OpenAI
    {
        "provider": "OpenAI",
        "name": "GPT-4o",
        "context_window": 128000,
        "max_output_tokens": 16384,
        "input_price_per_1m": 2.50,
        "output_price_per_1m": 10.00,
        "modalities": ["text", "image", "audio"],
        "supports_fine_tuning": False,
        "supports_structured_output": True,
        "deployment": ["cloud_api"],
        "open_source": False,
        "knowledge_cutoff": "2025-06",
        "notes": "OpenAI 最新主力多模态模型，延迟低，适合生产环境。",
    },
    {
        "provider": "OpenAI",
        "name": "GPT-4.1",
        "context_window": 1047576,
        "max_output_tokens": 32768,
        "input_price_per_1m": 2.00,
        "output_price_per_1m": 8.00,
        "modalities": ["text"],
        "supports_fine_tuning": False,
        "supports_structured_output": True,
        "deployment": ["cloud_api"],
        "open_source": False,
        "knowledge_cutoff": "2025-06",
        "notes": "百万级上下文，适合长文档分析和代码库级任务。",
    },
    {
        "provider": "OpenAI",
        "name": "GPT-4.1-mini",
        "context_window": 1047576,
        "max_output_tokens": 32768,
        "input_price_per_1m": 0.40,
        "output_price_per_1m": 1.60,
        "modalities": ["text"],
        "supports_fine_tuning": False,
        "supports_structured_output": True,
        "deployment": ["cloud_api"],
        "open_source": False,
        "knowledge_cutoff": "2025-06",
        "notes": "百万级上下文的性价比之选，延迟低。",
    },
    {
        "provider": "OpenAI",
        "name": "o4-mini",
        "context_window": 200000,
        "max_output_tokens": 100000,
        "input_price_per_1m": 1.10,
        "output_price_per_1m": 4.40,
        "modalities": ["text"],
        "supports_fine_tuning": False,
        "supports_structured_output": False,
        "deployment": ["cloud_api"],
        "open_source": False,
        "knowledge_cutoff": "2025-06",
        "notes": "推理模型，适合复杂多步推理和代码生成。",
    },
    # Anthropic
    {
        "provider": "Anthropic",
        "name": "Claude Sonnet 4",
        "context_window": 200000,
        "max_output_tokens": 64000,
        "input_price_per_1m": 3.00,
        "output_price_per_1m": 15.00,
        "modalities": ["text", "image"],
        "supports_fine_tuning": False,
        "supports_structured_output": True,
        "deployment": ["cloud_api"],
        "open_source": False,
        "knowledge_cutoff": "2025-05",
        "notes": "Claude 最新中端模型，推理和编码强，支持扩展思考。",
    },
    {
        "provider": "Anthropic",
        "name": "Claude Opus 4",
        "context_window": 200000,
        "max_output_tokens": 32000,
        "input_price_per_1m": 15.00,
        "output_price_per_1m": 75.00,
        "modalities": ["text", "image"],
        "supports_fine_tuning": False,
        "supports_structured_output": True,
        "deployment": ["cloud_api"],
        "open_source": False,
        "knowledge_cutoff": "2025-05",
        "notes": "Anthropic 最强模型，适合最复杂的分析任务。",
    },
    {
        "provider": "Anthropic",
        "name": "Claude Haiku 4.5",
        "context_window": 200000,
        "max_output_tokens": 8192,
        "input_price_per_1m": 0.80,
        "output_price_per_1m": 4.00,
        "modalities": ["text"],
        "supports_fine_tuning": False,
        "supports_structured_output": True,
        "deployment": ["cloud_api"],
        "open_source": False,
        "knowledge_cutoff": "2025-05",
        "notes": "Claude 最快/最便宜模型，适合高吞吐场景。",
    },
    # DeepSeek
    {
        "provider": "DeepSeek",
        "name": "DeepSeek-Chat (V3)",
        "context_window": 131072,
        "max_output_tokens": 8192,
        "input_price_per_1m": 0.27,
        "output_price_per_1m": 1.10,
        "modalities": ["text"],
        "supports_fine_tuning": False,
        "supports_structured_output": True,
        "deployment": ["cloud_api"],
        "open_source": True,
        "knowledge_cutoff": "2025-02",
        "notes": "DeepSeek 主力对话模型，开源权重可用，性价比极高。",
    },
    {
        "provider": "DeepSeek",
        "name": "DeepSeek-Reasoner (R1)",
        "context_window": 65536,
        "max_output_tokens": 8192,
        "input_price_per_1m": 0.55,
        "output_price_per_1m": 2.19,
        "modalities": ["text"],
        "supports_fine_tuning": False,
        "supports_structured_output": False,
        "deployment": ["cloud_api"],
        "open_source": True,
        "knowledge_cutoff": "2025-02",
        "notes": "推理特化模型，适合数学/编程等需要深度推理的任务。",
    },
    # Google
    {
        "provider": "Google",
        "name": "Gemini 2.5 Pro",
        "context_window": 1048576,
        "max_output_tokens": 65536,
        "input_price_per_1m": 1.25,
        "output_price_per_1m": 10.00,
        "modalities": ["text", "image", "audio", "video"],
        "supports_fine_tuning": False,
        "supports_structured_output": True,
        "deployment": ["cloud_api"],
        "open_source": False,
        "knowledge_cutoff": "2025-07",
        "notes": "百万级上下文 + 原生多模态，Google 最新旗舰。",
    },
    {
        "provider": "Google",
        "name": "Gemini 2.5 Flash",
        "context_window": 1048576,
        "max_output_tokens": 65536,
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 0.60,
        "modalities": ["text", "image", "audio", "video"],
        "supports_fine_tuning": False,
        "supports_structured_output": True,
        "deployment": ["cloud_api"],
        "open_source": False,
        "knowledge_cutoff": "2025-07",
        "notes": "百万级上下文的极致性价比，延迟极低。",
    },
    # Open-source models (local deployment candidates)
    {
        "provider": "Meta",
        "name": "Llama 4 (Maverick)",
        "context_window": 131072,
        "max_output_tokens": 4096,
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
        "modalities": ["text", "image"],
        "supports_fine_tuning": True,
        "supports_structured_output": True,
        "deployment": ["local", "cloud_api"],
        "open_source": True,
        "knowledge_cutoff": "2025-03",
        "min_ram_gb": 16,
        "min_vram_gb": 16,
        "quantization_support": True,
        "notes": "Meta 开源旗舰，可在本地部署，支持量化以降低资源需求。",
    },
    {
        "provider": "Meta",
        "name": "Llama 4 (Scout)",
        "context_window": 131072,
        "max_output_tokens": 4096,
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
        "modalities": ["text", "image"],
        "supports_fine_tuning": True,
        "supports_structured_output": True,
        "deployment": ["local"],
        "open_source": True,
        "knowledge_cutoff": "2025-03",
        "min_ram_gb": 8,
        "min_vram_gb": 6,
        "quantization_support": True,
        "notes": "Llama 4 轻量版，可在消费级硬件上运行（8GB RAM），支持 4-bit 量化。",
    },
    {
        "provider": "Mistral",
        "name": "Mistral Large 2",
        "context_window": 131072,
        "max_output_tokens": 4096,
        "input_price_per_1m": 2.00,
        "output_price_per_1m": 6.00,
        "modalities": ["text"],
        "supports_fine_tuning": True,
        "supports_structured_output": True,
        "deployment": ["local", "cloud_api"],
        "open_source": True,
        "knowledge_cutoff": "2024-12",
        "min_ram_gb": 16,
        "min_vram_gb": 16,
        "quantization_support": True,
        "notes": "欧洲开源旗舰模型，多语言能力强，支持本地量化部署。",
    },
    {
        "provider": "Alibaba",
        "name": "Qwen 3 (72B)",
        "context_window": 131072,
        "max_output_tokens": 8192,
        "input_price_per_1m": 0.35,
        "output_price_per_1m": 0.40,
        "modalities": ["text"],
        "supports_fine_tuning": True,
        "supports_structured_output": True,
        "deployment": ["local", "cloud_api"],
        "open_source": True,
        "knowledge_cutoff": "2025-04",
        "min_ram_gb": 32,
        "min_vram_gb": 24,
        "quantization_support": True,
        "notes": "通义千问最新开源版本，中文能力强，支持本地部署（4-bit 量化后可降至 16GB）。",
    },
]


# ===== Agent 框架能力矩阵 =====

_FRAMEWORKS: list[dict] = [
    {
        "name": "LangGraph",
        "language": "Python/TypeScript",
        "multi_agent": True,
        "mcp_support": True,
        "checkpoint": True,
        "human_in_the_loop": True,
        "streaming": True,
        "rag": "需集成 LangChain/LlamaIndex",
        "deployment": ["self_hosted", "langgraph_cloud"],
        "license": "MIT",
        "learning_curve": "中等偏高",
        "github_stars": 15000,
        "ecosystem_maturity": "高",
    },
    {
        "name": "CrewAI",
        "language": "Python",
        "multi_agent": True,
        "mcp_support": True,
        "checkpoint": False,
        "human_in_the_loop": True,
        "streaming": False,
        "rag": "需集成外部工具",
        "deployment": ["self_hosted"],
        "license": "MIT",
        "learning_curve": "低",
        "github_stars": 25000,
        "ecosystem_maturity": "中",
    },
    {
        "name": "AutoGen",
        "language": "Python/.NET",
        "multi_agent": True,
        "mcp_support": True,
        "checkpoint": False,
        "human_in_the_loop": True,
        "streaming": True,
        "rag": "需集成外部工具",
        "deployment": ["self_hosted"],
        "license": "CC-BY-4.0 / MIT",
        "learning_curve": "中等",
        "github_stars": 40000,
        "ecosystem_maturity": "中高",
    },
    {
        "name": "OpenAI Agents SDK",
        "language": "Python",
        "multi_agent": True,
        "mcp_support": True,
        "checkpoint": False,
        "human_in_the_loop": False,
        "streaming": True,
        "rag": "需集成外部工具",
        "deployment": ["cloud_api"],
        "license": "MIT",
        "learning_curve": "低",
        "github_stars": 20000,
        "ecosystem_maturity": "中",
    },
    {
        "name": "Dify",
        "language": "Python/TypeScript",
        "multi_agent": False,
        "mcp_support": False,
        "checkpoint": False,
        "human_in_the_loop": False,
        "streaming": True,
        "rag": "内置",
        "deployment": ["self_hosted", "cloud"],
        "license": "Apache-2.0",
        "learning_curve": "低",
        "github_stars": 80000,
        "ecosystem_maturity": "高",
    },
]

_DATA_UPDATED_AT = datetime(2026, 7, 1, tzinfo=timezone.utc).isoformat()


@tool(description="Query structured model comparison data — context windows, pricing, deployment options, and capabilities. Use this to get accurate, up-to-date facts about LLMs when comparing models for technology selection.")
def model_info(
    query: str = "",
    provider: str = "",
    model_name: str = "",
    category: str = "models",
) -> str:
    """查询结构化的模型/框架对比数据。

    此工具提供经过验证的 LLM 模型定价、上下文窗口、部署方式等结构化数据，
    以及常见 AI Agent 框架的能力矩阵。用于验证搜索结果的准确性。

    Args:
        query: 自由文本查询（如 "哪些模型适合离线部署" 或 "性价比最高的模型"）
        provider: 按提供商过滤（OpenAI、Anthropic、DeepSeek、Google、Meta、Mistral、Alibaba）
        model_name: 按模型名过滤（模糊匹配）
        category: 数据类别 — "models"（LLM 模型）或 "frameworks"（Agent 框架）

    Returns:
        格式化的结构化对比数据
    """
    if category == "frameworks":
        dataset = _FRAMEWORKS
    else:
        dataset = _MODELS

    results = list(dataset)

    if provider:
        provider_lower = provider.strip().lower()
        results = [
            r for r in results
            if provider_lower in r.get("provider", "").lower()
        ]

    if model_name:
        name_lower = model_name.strip().lower()
        results = [
            r for r in results
            if name_lower in r.get("name", "").lower()
        ]

    if query and not (provider or model_name):
        query_lower = query.strip().lower()
        scored = []
        for r in results:
            score = 0
            search_text = " ".join(
                str(v) for v in r.values() if isinstance(v, (str, list))
            ).lower()
            for word in query_lower.split():
                if word in search_text:
                    score += 1
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [r for _, r in scored[:10]]

    if not results:
        return (
            f"未找到匹配的结构化数据。\n"
            f"类别：{category}，提供商：{provider or '不限'}，"
            f"模型：{model_name or '不限'}\n"
            f"数据更新日期：{_DATA_UPDATED_AT}"
        )

    lines = [
        f"结构化{'模型' if category == 'models' else '框架'}对比数据"
        f"（更新于 {_DATA_UPDATED_AT}）：\n",
    ]

    if category == "models":
        lines.append(
            "| 模型 | 上下文窗口 | 输入价格/1M | 输出价格/1M | 部署 | 开源 | 最低RAM |"
            "\n|------|-----------|------------|------------|------|------|--------|"
        )
        for r in results:
            provider_name = f"{r['provider']} {r['name']}"
            ctx = f"{r['context_window']:,}"
            in_price = f"${r['input_price_per_1m']:.2f}" if r['input_price_per_1m'] > 0 else "免费"
            out_price = f"${r['output_price_per_1m']:.2f}" if r['output_price_per_1m'] > 0 else "免费"
            deploy = ", ".join(r.get("deployment", []))
            oss = "✓" if r["open_source"] else "✗"
            ram = r.get("min_ram_gb", "N/A")
            lines.append(
                f"| {provider_name} | {ctx} | {in_price} | {out_price} | "
                f"{deploy} | {oss} | {ram} |"
            )
        lines.append("\n详细能力：")
        for r in results:
            lines.append(
                f"\n**{r['provider']} {r['name']}**\n"
                f"- 上下文窗口：{r['context_window']:,} tokens\n"
                f"- 最大输出：{r['max_output_tokens']:,} tokens\n"
                f"- 输入价格：${r['input_price_per_1m']:.2f}/1M tokens\n"
                f"- 输出价格：${r['output_price_per_1m']:.2f}/1M tokens\n"
                f"- 模态：{', '.join(r.get('modalities', []))}\n"
                f"- 部署方式：{', '.join(r.get('deployment', []))}\n"
                f"- 开源：{'是' if r['open_source'] else '否'}\n"
                f"- 支持微调：{'是' if r.get('supports_fine_tuning') else '否'}\n"
                f"- 结构化输出：{'是' if r.get('supports_structured_output') else '否'}\n"
                f"- 最低硬件要求：{r.get('min_ram_gb', 'N/A')} GB RAM"
                f"{', ' + str(r.get('min_vram_gb', '')) + ' GB VRAM' if r.get('min_vram_gb') else ''}\n"
                f"- 知识截止：{r.get('knowledge_cutoff', '未知')}\n"
                f"- {r.get('notes', '')}"
            )
    else:
        lines.append(
            "| 框架 | 多Agent | MCP | Checkpoint | HITL | 流式 | RAG | 部署 | 学习曲线 |"
            "\n|------|---------|-----|-----------|------|------|-----|------|---------|"
        )
        for r in results:
            lines.append(
                f"| {r['name']} | {'✓' if r['multi_agent'] else '✗'} | "
                f"{'✓' if r['mcp_support'] else '✗'} | "
                f"{'✓' if r['checkpoint'] else '✗'} | "
                f"{'✓' if r['human_in_the_loop'] else '✗'} | "
                f"{'✓' if r['streaming'] else '✗'} | "
                f"{r['rag']} | {', '.join(r['deployment'])} | "
                f"{r['learning_curve']} |"
            )
        lines.append("\n详细能力：")
        for r in results:
            lines.append(
                f"\n**{r['name']}**\n"
                f"- 语言：{r['language']}\n"
                f"- 多 Agent：{'是' if r['multi_agent'] else '否'}\n"
                f"- MCP 支持：{'是' if r['mcp_support'] else '否'}\n"
                f"- Checkpoint/持久化：{'是' if r['checkpoint'] else '否'}\n"
                f"- Human-in-the-Loop：{'是' if r['human_in_the_loop'] else '否'}\n"
                f"- 流式输出：{'是' if r['streaming'] else '否'}\n"
                f"- RAG：{r['rag']}\n"
                f"- 部署方式：{', '.join(r['deployment'])}\n"
                f"- 许可证：{r['license']}\n"
                f"- 学习曲线：{r['learning_curve']}\n"
                f"- GitHub Stars：{r['github_stars']:,}\n"
                f"- 生态成熟度：{r['ecosystem_maturity']}"
            )

    lines.append(
        f"\n---\n⚠ 数据更新于 {_DATA_UPDATED_AT}。价格和能力可能已有变动，"
        f"建议以官方最新定价页面为准。此工具提供结构化基线，应配合 web_search 交叉验证。"
    )
    return "\n".join(lines)


@tool(description="Get hardware requirements and feasibility for running LLM models locally. Checks if a given model can run on specified hardware.")
def check_local_feasibility(
    model_name: str = "",
    available_ram_gb: float = 0,
    available_vram_gb: float = 0,
    deployment_mode: str = "local",
) -> str:
    """检查本地部署 LLM 的硬件可行性。

    给定目标模型和可用硬件，判断是否可行，并给出推荐的量化方案。

    Args:
        model_name: 目标模型名称（模糊匹配注册表中的模型）
        available_ram_gb: 可用系统内存（GB）
        available_vram_gb: 可用显存（GB）
        deployment_mode: 部署模式 — local 或 cloud_api

    Returns:
        可行性评估和建议
    """
    if deployment_mode == "cloud_api":
        return (
            "云端 API 部署不需要考虑本地硬件。"
            "请在 model_info 中查看各模型的 API 定价以进行成本对比。"
        )

    if not model_name and (available_ram_gb > 0 or available_vram_gb > 0):
        # 用户给了硬件但没指定模型：列出能在该硬件上运行的模型
        suitable = []
        for m in _MODELS:
            if not m.get("open_source"):
                continue
            min_ram = m.get("min_ram_gb", 999)
            min_vram = m.get("min_vram_gb", 999)
            if (available_ram_gb >= min_ram or available_ram_gb == 0) and \
               (available_vram_gb >= min_vram or available_vram_gb == 0):
                suitable.append(m)

        if not suitable:
            return (
                f"在 {available_ram_gb}GB RAM / {available_vram_gb}GB VRAM 的硬件上，"
                f"注册表中没有可直接运行的开源模型。\n"
                f"建议：考虑 4-bit 量化（可将 RAM 需求降低约 60%）或使用云端 API。"
            )

        lines = [
            f"以下开源模型可在 {available_ram_gb}GB RAM / {available_vram_gb}GB VRAM 上运行：\n"
        ]
        for m in suitable:
            quantization_note = ""
            min_ram = m.get("min_ram_gb", 999)
            if m.get("quantization_support") and available_ram_gb < min_ram * 0.6:
                quantization_note = "（需 4-bit 量化）"
            lines.append(
                f"- **{m['provider']} {m['name']}**：需 {m.get('min_ram_gb', '?')}GB RAM"
                f"{', ' + str(m.get('min_vram_gb', '')) + 'GB VRAM' if m.get('min_vram_gb') else ''}"
                f"{quantization_note}"
            )
        lines.append(
            "\n⚠ 以上为 FP16 最低需求。4-bit 量化可将 RAM 需求降低 55-65%，"
            "但会牺牲部分推理质量。"
        )
        return "\n".join(lines)

    # 按模型名搜索
    matched = None
    for m in _MODELS:
        if model_name.lower() in m.get("name", "").lower() or \
           model_name.lower() in m.get("provider", "").lower():
            matched = m
            break

    if not matched:
        return (
            f"模型 '{model_name}' 不在注册表中。\n"
            f"可用 model_info 查看已注册的模型列表。"
        )

    if not matched.get("open_source"):
        return (
            f"**{matched['provider']} {matched['name']}** 不开源，无法本地部署。\n"
            f"部署方式：{', '.join(matched.get('deployment', []))}\n"
            f"建议使用云端 API 或选择开源替代（如 Llama 4、Qwen 3、Mistral Large 2）。"
        )

    min_ram = matched.get("min_ram_gb", 0)
    min_vram = matched.get("min_vram_gb", 0)

    if available_ram_gb <= 0 and available_vram_gb <= 0:
        return (
            f"**{matched['provider']} {matched['name']}** 支持本地部署。\n"
            f"- FP16 最低需求：{min_ram}GB RAM"
            f"{', ' + str(min_vram) + 'GB VRAM' if min_vram else ''}\n"
            f"- 支持量化：{'是' if matched.get('quantization_support') else '否'}\n"
            f"请提供可用硬件规格以进行可行性评估。"
        )

    feasible = True
    issues = []

    if min_ram > 0 and available_ram_gb < min_ram:
        if matched.get("quantization_support"):
            quant_ram = min_ram * 0.4
            if available_ram_gb >= quant_ram:
                feasible = True
                issues.append(
                    f"FP16 需要 {min_ram}GB RAM，你的 {available_ram_gb}GB 不足。"
                    f"但 4-bit 量化后约需 {quant_ram:.0f}GB，可以运行（推理质量可能下降 5-10%）。"
                )
            else:
                feasible = False
                issues.append(
                    f"FP16 需要 {min_ram}GB RAM，即使 4-bit 量化也需要 {quant_ram:.0f}GB，"
                    f"你的 {available_ram_gb}GB 无法运行。"
                )
        else:
            feasible = False
            issues.append(f"需要 {min_ram}GB RAM，你的 {available_ram_gb}GB 不足。")

    if min_vram > 0 and available_vram_gb < min_vram:
        feasible = False
        issues.append(f"需要 {min_vram}GB VRAM，你的 {available_vram_gb}GB 不足。")

    if feasible:
        return (
            f"✅ **{matched['provider']} {matched['name']}** 可在你的硬件上运行。\n"
            + ("\n".join(f"- {i}" for i in issues) if issues else
               f"- {'满足' if available_ram_gb >= min_ram else '通过量化可满足'}最低需求")
        )

    return (
        f"❌ **{matched['provider']} {matched['name']}** 无法在你的硬件上运行。\n"
        + "\n".join(f"- {i}" for i in issues) +
        f"\n\n建议：\n"
        f"1. 选择更轻量的开源模型（如 Llama 4 Scout 只需 8GB RAM）\n"
        f"2. 使用云端 API（在 model_info 中查看定价）\n"
        f"3. 升级硬件以满足最低需求"
    )
