"""约束可行性分析器 — 在研究投入前检测需求自洽性。

解决三个盲区：
6. 硬约束求解：区分"必须满足"和"尽量满足"，检测约束冲突
7. 可行性预检：在投入研究资源前识别物理矛盾
8. 降级方案：当理想方案不可行时，生成可行的 trade-off 路径
"""

from dataclasses import dataclass, field
from typing import Optional


# ===== 已知物理限制规则 =====

# 模型规模 → 最低硬件需求映射（FP16）
_MODEL_HARDWARE_RULES = {
    "gpt-4": {"min_vram_gb": 48, "min_ram_gb": 64, "offline_possible": False},
    "gpt-4o": {"min_vram_gb": 48, "min_ram_gb": 64, "offline_possible": False},
    "claude": {"min_vram_gb": 48, "min_ram_gb": 64, "offline_possible": False},
    "large_model": {"min_vram_gb": 24, "min_ram_gb": 32, "offline_possible": True},
    "medium_model": {"min_vram_gb": 8, "min_ram_gb": 16, "offline_possible": True},
    "small_model": {"min_vram_gb": 4, "min_ram_gb": 8, "offline_possible": True},
}

# 推理延迟经验值
_LATENCY_RULES = {
    "local_large": "500ms-5s",
    "local_medium": "200ms-2s",
    "local_small": "50ms-500ms",
    "cloud_api": "200ms-2s（含网络延迟）",
    "realtime": "<200ms",
    "interactive": "<2s",
    "batch": "分钟级可接受",
}

# 部署方式 → 隐含约束
_DEPLOYMENT_IMPLICATIONS = {
    "fully_offline": {
        "implies": ["模型必须本地部署", "不能使用云端 API", "需要足够的本地硬件"],
        "excludes": ["openai", "anthropic_api", "google_api", "deepseek_api"],
    },
    "self_hosted": {
        "implies": ["需要运维能力", "需要 GPU 或足够 CPU 内存"],
        "excludes": [],
    },
    "cloud_api": {
        "implies": ["需要网络连接", "数据发送到第三方", "按量付费"],
        "excludes": ["完全离线场景"],
    },
}


@dataclass
class ConstraintViolation:
    """单条约束冲突。"""

    severity: str  # "error" (不可能) | "warning" (严重trade-off) | "info" (注意事项)
    description: str
    suggestion: str = ""


@dataclass
class DegradationPath:
    """一条可行的降级方案。"""

    label: str
    description: str
    trade_offs: list[str] = field(default_factory=list)
    candidate_models: list[str] = field(default_factory=list)
    candidate_frameworks: list[str] = field(default_factory=list)


@dataclass
class FeasibilityReport:
    """约束可行性分析报告。"""

    is_feasible: bool
    violations: list[ConstraintViolation] = field(default_factory=list)
    degradation_paths: list[DegradationPath] = field(default_factory=list)
    hard_constraints: list[str] = field(default_factory=list)
    soft_preferences: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)


def analyze_feasibility(
    requirements: dict,
    candidates: Optional[list[str]] = None,
) -> FeasibilityReport:
    """分析需求的技术可行性。

    在研究开始前进行，避免在不可能的需求组合上浪费 Token。

    Args:
        requirements: 结构化需求字典，包含：
            - deployment: 部署方式
            - required_features: 必须功能列表
            - preferred_features: 偏好功能列表
            - budget_constraints: 预算约束文本
            - language: 编程语言偏好
            - team_level: 团队水平
            - additional_notes: 补充说明
        candidates: 候选框架/项目列表

    Returns:
        FeasibilityReport 包含冲突、风险和降级方案
    """
    violations: list[ConstraintViolation] = []
    hard_constraints: list[str] = []
    soft_preferences: list[str] = []
    degradation_paths: list[DegradationPath] = []
    risk_flags: list[str] = []

    deployment = (requirements.get("deployment") or "").strip().lower()
    budget_text = (requirements.get("budget_constraints") or "").strip()
    additional = (requirements.get("additional_notes") or "").strip()
    all_text = f"{deployment} {budget_text} {additional}".lower()

    required_features = requirements.get("required_features", []) or []
    preferred_features = requirements.get("preferred_features", []) or []
    team_level = (requirements.get("team_level") or "").strip().lower()

    # === 1. 提取硬约束与软偏好 ===
    for feat in required_features:
        hard_constraints.append(feat)

    for feat in preferred_features:
        soft_preferences.append(feat)

    if deployment:
        hard_constraints.append(f"部署方式：{deployment}")

    # === 2. 物理矛盾检测 ===

    # 2a. 离线 + 云端模型
    is_offline = any(
        kw in all_text for kw in ["完全离线", "离线部署", "内网", "air-gap", "offline", "fully_offline", "本地部署"]
    )
    wants_large_model = any(
        kw in all_text for kw in ["gpt-4", "claude", "大模型", "强推理", "高级推理", "gpt4"]
    )
    wants_low_latency = any(
        kw in all_text for kw in ["毫秒", "实时", "低延迟", "millisecond", "realtime", "<100ms", "<200ms"]
    )
    has_low_ram = _extract_ram_constraint(all_text)

    # 离线 + 云端 API 模型 = 物理矛盾
    if is_offline and wants_large_model:
        violations.append(ConstraintViolation(
            severity="error",
            description="需要 GPT-4/Claude 级别推理能力，但要求完全离线部署——这两个约束互相矛盾。"
                "GPT-4、Claude 等顶级闭源模型只能通过云端 API 使用，无法离线部署。",
            suggestion="选择以下折中方案之一：",
        ))
        degradation_paths.append(DegradationPath(
            label="方案 A：接受云端 API",
            description="使用 GPT-4o 或 Claude Sonnet 4 的云端 API，获得最强推理能力。"
                "延迟 200-800ms，需要网络连接。适合对数据安全要求不极端的场景。",
            trade_offs=["需要网络连接", "数据发送至第三方", "按量付费"],
            candidate_models=["GPT-4o", "Claude Sonnet 4", "DeepSeek-Chat"],
        ))
        degradation_paths.append(DegradationPath(
            label="方案 B：本地部署开源大模型",
            description="使用 Llama 4 (Maverick) 或 Qwen 3 (72B) 的 4-bit 量化版本。"
                "推理能力可达 GPT-4 的 70-85%，可完全离线。需要 16-32GB RAM + GPU。",
            trade_offs=["推理能力不如 GPT-4", "需要较强的硬件", "部署和调优复杂度高"],
            candidate_models=["Llama 4 (Maverick) 4-bit", "Qwen 3 (72B) 4-bit", "Mistral Large 2 4-bit"],
        ))

    # 大模型 + 低内存 = 物理矛盾
    if has_low_ram and wants_large_model:
        violations.append(ConstraintViolation(
            severity="error",
            description=f"需要大模型推理能力，但可用内存仅 {has_low_ram}GB——不足以运行任何大模型。"
                f"即使是 4-bit 量化的 7B 模型也需要至少 6-8GB RAM。",
            suggestion="使用云端 API 可以完全避免本地硬件限制。",
        ))

    # 低延迟 + 本地大模型 = 严重 trade-off
    if wants_low_latency and is_offline and wants_large_model:
        violations.append(ConstraintViolation(
            severity="warning",
            description="本地部署大模型 + 毫秒级延迟是极难同时满足的组合。"
                "即使是量化后的 7B 模型，在消费级 GPU 上推理延迟通常在 200ms-2s。",
            suggestion="如果延迟是硬约束，应选择更小的模型（1-3B 参数）或使用云端 API。",
        ))

    # === 3. 部署方式隐含约束分析 ===
    if is_offline:
        risk_flags.append("离线部署：所有依赖必须在本地可用，不能依赖云端服务")
        if team_level == "beginner":
            risk_flags.append(
                "团队经验为初级但要求离线部署——本地模型部署和运维复杂度高，"
                "建议评估是否有 DevOps 支持"
            )

    # === 4. 预算约束分析 ===
    if budget_text:
        if any(kw in budget_text.lower() for kw in ["免费", "开源", "free", "0"]):
            if is_offline and not has_low_ram:
                risk_flags.append("预算要求免费/开源：与离线部署一致。注意开源模型的部署硬件也是一次性成本。")
            else:
                risk_flags.append("预算要求免费/开源：只能使用开源方案。排除所有商业 API 和付费服务。")
                degradation_paths.append(DegradationPath(
                    label="纯开源方案",
                    description="使用完全开源的技术栈：Llama/Mistral/Qwen 等开源模型 + LangGraph/CrewAI 等开源框架。",
                    trade_offs=["需要自行部署和维护", "可能需要更强的工程能力"],
                    candidate_models=["Llama 4 Scout", "Qwen 3 量化版", "Mistral 7B"],
                    candidate_frameworks=["LangGraph", "CrewAI", "AutoGen"],
                ))

    # === 5. 候选框架兼容性检查 ===
    if candidates:
        framework_compat = _check_framework_compatibility(candidates, requirements)
        violations.extend(framework_compat)

    return FeasibilityReport(
        is_feasible=not any(v.severity == "error" for v in violations),
        violations=violations,
        degradation_paths=degradation_paths,
        hard_constraints=hard_constraints,
        soft_preferences=soft_preferences,
        risk_flags=risk_flags,
    )


def _extract_ram_constraint(text: str) -> Optional[int]:
    """从文本中提取内存约束（GB）。"""
    import re

    patterns = [
        r"(\d+)\s*[gG][bB]?\s*(?:内存|ram|内存|运存)",
        r"(?:内存|ram|运存)\s*(?:只有|仅|最多)?\s*(\d+)\s*[gG][bB]?",
        r"(\d+)\s*[gG]\s*(?:笔记本|轻薄本|电脑)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _check_framework_compatibility(
    candidates: list[str],
    requirements: dict,
) -> list[ConstraintViolation]:
    """检查候选框架是否满足用户的硬约束。"""
    violations = []
    deployment = (requirements.get("deployment") or "").strip().lower()
    required_features = [f.strip().lower() for f in (requirements.get("required_features") or [])]

    # 已知框架能力（与 model_registry 保持一致）。
    #
    # 布尔值只用于能否满足硬约束；“需外部集成”仍然是支持，
    # 不能和“不支持”混为一谈。RAG 本来就通常是编排框架 +
    # retriever/vector store 的组合能力。
    framework_caps = {
        "langgraph": {"multi_agent": True, "mcp": True, "checkpoint": True,
                      "hitl": True, "streaming": True, "rag": True, "offline": True},
        "crewai": {"multi_agent": True, "mcp": True, "checkpoint": False,
                   "hitl": True, "streaming": False, "rag": True, "offline": True},
        "autogen": {"multi_agent": True, "mcp": True, "checkpoint": False,
                    "hitl": True, "streaming": True, "rag": True, "offline": True},
        "dify": {"multi_agent": False, "mcp": False, "checkpoint": False,
                  "hitl": False, "streaming": True, "rag": True, "offline": True},
        "openai agents sdk": {"multi_agent": True, "mcp": True, "checkpoint": False,
                              "hitl": False, "streaming": True, "rag": True, "offline": False},
    }

    feature_to_key = {
        "多agent": "multi_agent",
        "多智能体": "multi_agent",
        "multi-agent": "multi_agent",
        "multi_agent": "multi_agent",
        "mcp": "mcp",
        "checkpoint": "checkpoint",
        "持久化": "checkpoint",
        "人工审批": "hitl",
        "human-in-the-loop": "hitl",
        "hitl": "hitl",
        "流式": "streaming",
        "streaming": "streaming",
        "rag": "rag",
        "知识库": "rag",
    }

    for candidate in candidates:
        caps = None
        for key, value in framework_caps.items():
            if key in candidate.lower():
                caps = value
                break

        if caps is None:
            continue

        for feat in required_features:
            cap_key = feature_to_key.get(feat)
            if cap_key and not caps.get(cap_key, False):
                violations.append(ConstraintViolation(
                    severity="warning",
                    description=f"{candidate} 不支持 '{feat}' 功能，"
                    f"但这是你的硬约束之一。",
                    suggestion=f"考虑将 {candidate} 替换为支持 {feat} 的框架，"
                    f"或将 {feat} 降级为可选偏好。",
                ))

        # 离线兼容性
        if "离线" in deployment or "offline" in deployment:
            if not caps.get("offline", True):
                violations.append(ConstraintViolation(
                    severity="error",
                    description=f"需要离线部署，但 {candidate} 依赖云端 API，无法离线使用。",
                    suggestion=f"选择支持本地部署的替代框架。",
                ))

    return violations


def render_feasibility_report(report: FeasibilityReport) -> str:
    """将可行性报告渲染为 Markdown，供后续节点和 Reviewer 使用。"""
    sections = ["## 需求可行性预检\n"]

    # 状态
    if report.is_feasible:
        sections.append("**✅ 整体评估：需求可行，未检测到物理矛盾。**\n")
    else:
        sections.append("**⚠️ 整体评估：检测到物理矛盾或不可行约束，请审阅以下分析。**\n")

    # 硬约束
    if report.hard_constraints:
        sections.append("### 识别的硬约束")
        for c in report.hard_constraints:
            sections.append(f"- **{c}**")
        sections.append("")

    # 冲突
    if report.violations:
        sections.append("### 约束冲突与风险")
        for v in report.violations:
            icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(v.severity, "")
            sections.append(f"{icon} **{v.severity.upper()}**: {v.description}")
            if v.suggestion:
                sections.append(f"  → {v.suggestion}")
            sections.append("")

    # 风险
    if report.risk_flags:
        sections.append("### 风险提示")
        for r in report.risk_flags:
            sections.append(f"- ⚡ {r}")
        sections.append("")

    # 降级方案
    if report.degradation_paths:
        sections.append("### 可行降级路径")
        for i, path in enumerate(report.degradation_paths, 1):
            sections.append(f"**{path.label}**")
            sections.append(f"{path.description}")
            if path.trade_offs:
                sections.append("- 代价：")
                for t in path.trade_offs:
                    sections.append(f"  - {t}")
            if path.candidate_models:
                sections.append(f"- 推荐模型：{', '.join(path.candidate_models)}")
            if path.candidate_frameworks:
                sections.append(f"- 推荐框架：{', '.join(path.candidate_frameworks)}")
            sections.append("")

    return "\n".join(sections)
