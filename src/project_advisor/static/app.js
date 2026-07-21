const form = document.querySelector("#advisorForm");
const questionInput = document.querySelector("#question");
const candidatesInput = document.querySelector("#candidates");
const submitButton = document.querySelector("#submitButton");
const cancelButton = document.querySelector("#cancelButton");
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

function updateCharacterCount() {
  charCount.textContent = `${questionInput.value.length} / 5000`;
}

function resetTimeline() {
  document.querySelectorAll("#timeline li").forEach((item) => {
    item.classList.remove("active", "completed");
    item.querySelector(".timeline-state").textContent = "等待";
  });
}

function activateStage(nodeName) {
  const index = stageOrder.indexOf(nodeName);
  const nextNode = stageOrder[index + 1];
  const item = document.querySelector(`[data-node="${nodeName}"]`);
  if (item) {
    item.classList.remove("active");
    item.classList.add("completed");
    item.querySelector(".timeline-state").textContent = "完成";
  }
  if (nextNode) {
    const nextItem = document.querySelector(`[data-node="${nextNode}"]`);
    nextItem.classList.add("active");
    nextItem.querySelector(".timeline-state").textContent = "执行中";
  }
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
    activateStage(data.node);
    if (data.scores?.length) renderScores(data.scores);
  }
  if (eventName === "result") {
    currentReport = data.report || "";
    renderScores(data.scores || []);
    reportElement.innerHTML = renderMarkdown(currentReport);
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
  results.hidden = true;
  scoreGrid.innerHTML = "";
  reportElement.innerHTML = "";
  currentReport = "";
  resetTimeline();
  runState.textContent = "READY";
  setRunning(true);
  controller = new AbortController();

  const candidates = candidatesInput.value.split(/[,，]/).map((item) => item.trim()).filter(Boolean);
  try {
    const response = await fetch("/api/advice/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: questionInput.value, candidates, allow_clarification: false }),
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
questionInput.addEventListener("input", updateCharacterCount);

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    const example = examples[button.dataset.example];
    questionInput.value = example.question;
    candidatesInput.value = example.candidates;
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

updateCharacterCount();
