const form = document.querySelector("#advisorForm");
const questionInput = document.querySelector("#question");
const candidatesInput = document.querySelector("#candidates");
const submitButton = document.querySelector("#submitButton");
const submitButtonLabel = document.querySelector("#submitButtonLabel");
const cancelButton = document.querySelector("#cancelButton");
const candidateModeInputs = document.querySelectorAll('input[name="candidateMode"]');
const autoCandidateHelp = document.querySelector("#autoCandidateHelp");
const manualCandidateField = document.querySelector("#manualCandidateField");
const candidatePreview = document.querySelector("#candidatePreview");
const candidatePreviewCount = document.querySelector("#candidatePreviewCount");
const candidateList = document.querySelector("#candidateList");
const requirementsSummary = document.querySelector("#requirementsSummary");
const additionalCandidate = document.querySelector("#additionalCandidate");
const addCandidateButton = document.querySelector("#addCandidateButton");
const formError = document.querySelector("#formError");
const charCount = document.querySelector("#charCount");
const serviceStatus = document.querySelector("#serviceStatus");
const runState = document.querySelector("#runState");
const idleCard = document.querySelector("#idleCard");
const results = document.querySelector("#results");
const scoreGrid = document.querySelector("#scoreGrid");
const reportElement = document.querySelector("#report");
const copyButton = document.querySelector("#copyButton");
const downloadButton = document.querySelector("#downloadButton");
const diagnosticsCollectionState = document.querySelector("#diagnosticsCollectionState");
const runtimeTotal = document.querySelector("#runtimeTotal");
const runtimeCandidates = document.querySelector("#runtimeCandidates");
const runtimeCitations = document.querySelector("#runtimeCitations");
const runtimeMcp = document.querySelector("#runtimeMcp");
const runtimeTokens = document.querySelector("#runtimeTokens");
const runtimeCost = document.querySelector("#runtimeCost");
const stageDurations = document.querySelector("#stageDurations");
const evaluationState = document.querySelector("#evaluationState");
const evaluationMeta = document.querySelector("#evaluationMeta");

const stageOrder = [
  "clarify_requirements",
  "plan_evaluation",
  "research_supervisor",
  "review_and_score",
  "generate_report",
];

const examples = {
  agent: {
    question: "我们是一个 5 人 Python 团队，需要开发支持多智能体协作、RAG、人工审批、长任务恢复和私有化部署的企业 AI 应用。请推荐合适的 Agent 框架，并评估工程可靠性和学习成本。",
    candidates: "LangGraph, CrewAI, Microsoft Agent Framework",
  },
  rag: {
    question: "我们要为中文企业文档构建生产级 RAG，数据约 200 万段，需要混合检索、权限过滤、增量更新和可观测性。请比较适合自托管的技术组合。",
    candidates: "Milvus, Qdrant, Elasticsearch",
  },
  observability: {
    question: "团队需要给多个 LLM 应用统一接入 Trace、Prompt 版本、离线评测和线上反馈，要求支持私有化部署。请评估主流 LLM 可观测性平台。",
    candidates: "Langfuse, Phoenix, LangSmith",
  },
};

let controller = null;
let currentReport = "";
let currentPlan = null;
let plannedQuestion = "";

function updateCharacterCount() {
  charCount.textContent = `${questionInput.value.length} / 5000`;
}

function getCandidateMode() {
  return document.querySelector('input[name="candidateMode"]:checked')?.value || "auto";
}

function resetCandidatePlan() {
  currentPlan = null;
  plannedQuestion = "";
  candidatePreview.hidden = true;
  candidateList.innerHTML = "";
  submitButtonLabel.textContent = getCandidateMode() === "auto" ? "生成候选项目" : "开始深度评估";
}

function updateCandidateMode() {
  const automatic = getCandidateMode() === "auto";
  autoCandidateHelp.hidden = !automatic;
  manualCandidateField.hidden = automatic;
  if (!automatic) candidatePreview.hidden = true;
  if (automatic && currentPlan) candidatePreview.hidden = false;
  submitButtonLabel.textContent = automatic
    ? (currentPlan ? "确认并开始深度评估" : "生成候选项目")
    : "开始深度评估";
}

function summarizeRequirements(plan) {
  const requirements = plan.requirements || {};
  const parts = [];
  if (requirements.language) parts.push(`语言：${requirements.language}`);
  if (requirements.deployment) parts.push(`部署：${requirements.deployment}`);
  if (requirements.required_features?.length) parts.push(`必需能力：${requirements.required_features.join("、")}`);
  if (plan.evaluation_focus?.length) parts.push(`评估重点：${plan.evaluation_focus.join("、")}`);
  return parts.join(" · ") || "已根据需求生成候选项目，你可以删除或补充后再确认。";
}

function renderCandidatePlan() {
  const candidates = currentPlan?.candidates || [];
  candidateList.innerHTML = "";
  candidates.forEach((candidate, index) => {
    const card = document.createElement("article");
    card.className = "candidate-card";
    const content = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = candidate.name;
    content.append(name);
    if (candidate.github_url?.startsWith("http")) {
      const link = document.createElement("a");
      link.href = candidate.github_url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = "查看项目主页";
      content.append(document.createTextNode(" · "), link);
    }
    const reason = document.createElement("p");
    reason.textContent = candidate.reason || "用户手动添加的候选项目。";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "candidate-remove";
    remove.textContent = "删除";
    remove.addEventListener("click", () => {
      currentPlan.candidates.splice(index, 1);
      renderCandidatePlan();
    });
    card.append(content, reason, remove);
    candidateList.append(card);
  });
  candidatePreviewCount.textContent = `${candidates.length} 个`;
  requirementsSummary.textContent = summarizeRequirements(currentPlan || {});
  candidatePreview.hidden = false;
  submitButtonLabel.textContent = "确认并开始深度评估";
}

function addCandidate() {
  if (!currentPlan) return;
  const name = additionalCandidate.value.trim();
  if (!name) return;
  if (currentPlan.candidates.some((candidate) => candidate.name.toLowerCase() === name.toLowerCase())) {
    formError.textContent = "该候选项目已经在清单中。";
    return;
  }
  if (currentPlan.candidates.length >= 8) {
    formError.textContent = "候选项目最多 8 个。";
    return;
  }
  currentPlan.candidates.push({ name, github_url: null, reason: "用户在确认阶段手动添加。" });
  additionalCandidate.value = "";
  formError.textContent = "";
  renderCandidatePlan();
}

function resetTimeline() {
  document.querySelectorAll("#timeline li").forEach((item) => {
    item.classList.remove("active", "completed");
    item.querySelector(".timeline-state").textContent = "等待";
  });
}

function formatDuration(milliseconds) {
  const value = Number(milliseconds || 0);
  if (value < 1000) return `${Math.round(value)} ms`;
  if (value < 60000) return `${(value / 1000).toFixed(1)} s`;
  const minutes = Math.floor(value / 60000);
  const seconds = Math.round((value % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
}

function activateStage(nodeName, durationMs) {
  const index = stageOrder.indexOf(nodeName);
  const nextNode = stageOrder[index + 1];
  const item = document.querySelector(`[data-node="${nodeName}"]`);
  if (item) {
    item.classList.remove("active");
    item.classList.add("completed");
    item.querySelector(".timeline-state").textContent = `完成 · ${formatDuration(durationMs)}`;
  }
  if (nextNode) {
    const nextItem = document.querySelector(`[data-node="${nextNode}"]`);
    nextItem.classList.add("active");
    nextItem.querySelector(".timeline-state").textContent = "执行中";
  }
}

function renderDiagnostics(diagnostics) {
  const tokenUsage = diagnostics.token_usage || {};
  const mcp = diagnostics.mcp || {};
  const mcpLabels = {
    connected: "已连接",
    configured: "已配置",
    connecting: "连接中",
    degraded: "已降级",
    disabled: "未启用",
    invalid_configuration: "配置无效",
  };

  runtimeTotal.textContent = formatDuration(diagnostics.total_duration_ms);
  runtimeCandidates.textContent = String(diagnostics.candidate_count ?? 0);
  runtimeCitations.textContent = String(diagnostics.citation_url_count ?? 0);
  runtimeMcp.textContent = `${mcpLabels[mcp.status] || mcp.status || "未知"} / ${mcp.tool_count ?? 0}`;
  runtimeTokens.textContent = tokenUsage.collected
    ? Number(tokenUsage.total_tokens || 0).toLocaleString("zh-CN")
    : "未采集";
  runtimeCost.textContent = diagnostics.cost_configured && tokenUsage.collected
    ? `$${Number(diagnostics.estimated_cost_usd || 0).toFixed(6)}`
    : "未配置单价";
  diagnosticsCollectionState.textContent = tokenUsage.collected
    ? "模型 usage 已采集"
    : "模型未返回 usage";

  stageDurations.innerHTML = "";
  Object.entries(diagnostics.stage_durations_ms || {}).forEach(([nodeName, duration]) => {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const value = document.createElement("strong");
    label.textContent = document.querySelector(`[data-node="${nodeName}"] strong`)?.textContent || nodeName;
    value.textContent = formatDuration(duration);
    row.append(label, value);
    stageDurations.append(row);
  });
}

function renderEvaluation(payload) {
  const report = payload.report || {};
  const percentageMetrics = new Set([
    "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k",
    "citation_accuracy", "citation_coverage", "task_success_rate",
  ]);
  document.querySelectorAll("[data-eval]").forEach((element) => {
    const key = element.dataset.eval;
    const value = Number(report[key] || 0);
    if (percentageMetrics.has(key)) element.textContent = `${(value * 100).toFixed(1)}%`;
    else if (key.startsWith("latency_")) element.textContent = formatDuration(value);
    else if (key === "average_cost_usd") element.textContent = `$${value.toFixed(4)}`;
    else element.textContent = value.toLocaleString("zh-CN", { maximumFractionDigits: 1 });
  });
  const updatedAt = payload.updated_at ? new Date(payload.updated_at).toLocaleString("zh-CN") : "未知";
  evaluationMeta.textContent = `${payload.source} · K=${report.k} · 更新于 ${updatedAt}`;
  evaluationState.textContent = "READY";
}

function setRunning(running) {
  submitButton.disabled = running;
  cancelButton.hidden = !running;
  if (running) runState.textContent = "RUNNING";
  if (!running && runState.textContent === "RUNNING") runState.textContent = "READY";
  idleCard.classList.toggle("running", running);
  idleCard.querySelector("p").textContent = running
    ? "Agent 正在研究。复杂任务可能需要数分钟，请保持页面打开。"
    : "提交任务后，这里会实时展示每个 Agent 的执行状态。";
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function inlineMarkdown(value) {
  return escapeHtml(value).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function renderMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  const output = [];
  let listOpen = false;

  const closeList = () => {
    if (listOpen) {
      output.push("</ul>");
      listOpen = false;
    }
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.trim().startsWith("|") && lines[index + 1]?.includes("---")) {
      closeList();
      const tableLines = [line];
      index += 2;
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        tableLines.push(lines[index]);
        index += 1;
      }
      index -= 1;
      const rows = tableLines.map((row) => row.split("|").slice(1, -1).map((cell) => inlineMarkdown(cell.trim())));
      const header = rows.shift() || [];
      output.push(`<div class="table-wrap"><table><thead><tr>${header.map((cell) => `<th>${cell}</th>`).join("")}</tr></thead><tbody>`);
      rows.forEach((row) => output.push(`<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`));
      output.push("</tbody></table></div>");
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      closeList();
      output.push(`<h${heading[1].length}>${inlineMarkdown(heading[2])}</h${heading[1].length}>`);
      continue;
    }

    const bullet = line.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (!listOpen) {
        output.push("<ul>");
        listOpen = true;
      }
      output.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
      continue;
    }

    closeList();
    if (line.trim()) output.push(`<p>${inlineMarkdown(line)}</p>`);
  }
  closeList();
  return output.join("");
}

function renderScores(scores) {
  scoreGrid.innerHTML = "";
  scores.forEach((score) => {
    const total = Number(score.weighted_total || 0);
    const card = document.createElement("div");
    card.className = "score-card";
    const header = document.createElement("header");
    const name = document.createElement("strong");
    const value = document.createElement("b");
    name.textContent = score.project_name;
    value.textContent = total.toFixed(2);
    header.append(name, value);
    const bar = document.createElement("div");
    bar.className = "score-bar";
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(0, Math.min(total * 10, 100))}%`;
    bar.append(fill);
    card.append(header, bar);
    scoreGrid.append(card);
  });
}

function handleEvent(eventName, data) {
  if (eventName === "started") {
    const first = document.querySelector(`[data-node="${stageOrder[0]}"]`);
    first.classList.add("active");
    first.querySelector(".timeline-state").textContent = "执行中";
  }
  if (eventName === "progress") {
    activateStage(data.node, data.stage_duration_ms);
    if (data.scores?.length) renderScores(data.scores);
  }
  if (eventName === "result") {
    currentReport = data.report || "";
    renderScores(data.scores || []);
    reportElement.innerHTML = renderMarkdown(currentReport);
    renderDiagnostics(data.diagnostics || {});
    results.hidden = false;
    runState.textContent = "DONE";
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  if (eventName === "error") {
    throw new Error(data.message || "评估失败");
  }
}

async function consumeEventStream(response) {
  if (!response.ok || !response.body) throw new Error(`服务请求失败（${response.status}）`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      let eventName = "message";
      let data = "";
      block.split("\n").forEach((line) => {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        if (line.startsWith("data:")) data += line.slice(5).trim();
      });
      if (data) handleEvent(eventName, JSON.parse(data));
    }
    if (done) break;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.textContent = "";
  const mode = getCandidateMode();

  if (mode === "auto" && (!currentPlan || plannedQuestion !== questionInput.value.trim())) {
    setRunning(true);
    runState.textContent = "PLANNING";
    controller = new AbortController();
    try {
      const response = await fetch("/api/candidates/suggest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: questionInput.value }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        throw new Error(errorPayload.detail || `候选项目生成失败（${response.status}）`);
      }
      currentPlan = await response.json();
      plannedQuestion = questionInput.value.trim();
      renderCandidatePlan();
      runState.textContent = "READY";
    } catch (error) {
      if (error.name !== "AbortError") {
        formError.textContent = error.message || "候选项目生成失败，请检查模型配置。";
        runState.textContent = "ERROR";
      }
    } finally {
      controller = null;
      setRunning(false);
    }
    return;
  }

  const candidates = mode === "auto"
    ? (currentPlan?.candidates || []).map((candidate) => candidate.name.trim()).filter(Boolean)
    : candidatesInput.value.split(/[,，]/).map((item) => item.trim()).filter(Boolean);
  const uniqueCandidates = [...new Set(candidates)];
  if (!uniqueCandidates.length) {
    formError.textContent = mode === "auto" ? "请至少保留一个候选项目。" : "请填写至少一个候选项目。";
    return;
  }
  if (uniqueCandidates.length > 8) {
    formError.textContent = "候选项目最多 8 个。";
    return;
  }

  results.hidden = true;
  scoreGrid.innerHTML = "";
  reportElement.innerHTML = "";
  currentReport = "";
  diagnosticsCollectionState.textContent = "正在采集";
  resetTimeline();
  runState.textContent = "READY";
  setRunning(true);
  controller = new AbortController();

  try {
    const response = await fetch("/api/advice/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: questionInput.value,
        candidates: uniqueCandidates,
        allow_clarification: false,
        confirmed_plan: mode === "auto" ? currentPlan : null,
        confirmed_candidates: true,
      }),
      signal: controller.signal,
    });
    await consumeEventStream(response);
  } catch (error) {
    if (error.name !== "AbortError") {
      formError.textContent = error.message || "请求失败，请检查服务配置。";
      runState.textContent = "ERROR";
    }
  } finally {
    controller = null;
    setRunning(false);
  }
});

cancelButton.addEventListener("click", () => controller?.abort());
questionInput.addEventListener("input", () => {
  updateCharacterCount();
  if (getCandidateMode() === "auto" && plannedQuestion !== questionInput.value.trim()) {
    resetCandidatePlan();
  }
});
candidateModeInputs.forEach((input) => input.addEventListener("change", updateCandidateMode));
addCandidateButton.addEventListener("click", addCandidate);
additionalCandidate.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    addCandidate();
  }
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    const example = examples[button.dataset.example];
    questionInput.value = example.question;
    document.querySelector('input[name="candidateMode"][value="auto"]').checked = true;
    candidatesInput.value = "";
    resetCandidatePlan();
    updateCandidateMode();
    updateCharacterCount();
  });
});

copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(currentReport);
  copyButton.textContent = "已复制";
  setTimeout(() => { copyButton.textContent = "复制报告"; }, 1200);
});

downloadButton.addEventListener("click", () => {
  const blob = new Blob([currentReport], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `project-advisor-${new Date().toISOString().slice(0, 10)}.md`;
  link.click();
  URL.revokeObjectURL(url);
});

fetch("/api/health")
  .then((response) => {
    if (!response.ok) throw new Error();
    serviceStatus.classList.add("online");
    serviceStatus.querySelector("span:last-child").textContent = "服务在线";
  })
  .catch(() => { serviceStatus.querySelector("span:last-child").textContent = "服务离线"; });

fetch("/api/evaluation")
  .then((response) => {
    if (!response.ok) throw new Error(`评测数据加载失败（${response.status}）`);
    return response.json();
  })
  .then(renderEvaluation)
  .catch((error) => {
    evaluationState.textContent = "UNAVAILABLE";
    evaluationMeta.textContent = error.message || "离线评测数据不可用";
  });

updateCharacterCount();
updateCandidateMode();
