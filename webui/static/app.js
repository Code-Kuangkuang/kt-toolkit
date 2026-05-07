let runtimeConfig = null;
let selectedJobId = null;
let latestMetricsPayload = null;
let expandedChartType = null;
let selectedChartView = "";

const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

function setOptions(select, values) {
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function valueKind(value) {
  if (typeof value === "boolean") return "bool";
  if (typeof value === "number") return "number";
  return "text";
}

function renderModelConfig() {
  if (!runtimeConfig) return;
  const modelName = $("#jobForm select[name='model_name']").value;
  const modelConfig = runtimeConfig.models[modelName] || {};
  const entries = Object.entries(modelConfig);
  $("#modelConfigSummary").textContent = `${entries.length} fields`;

  const box = $("#modelConfigFields");
  box.innerHTML = "";
  for (const [key, value] of entries) {
    const kind = valueKind(value);
    const label = document.createElement("label");
    label.title = key;
    if (kind === "bool") {
      label.className = "bool-field";
      label.innerHTML = `
        <input class="model-config-input" data-key="${escapeHtml(key)}" data-kind="bool" type="checkbox" ${value ? "checked" : ""} />
        ${escapeHtml(key)}
      `;
    } else {
      const renderedValue = typeof value === "object" && value !== null ? JSON.stringify(value) : value ?? "";
      label.innerHTML = `
        ${escapeHtml(key)}
        <input
          class="model-config-input"
          data-key="${escapeHtml(key)}"
          data-kind="${kind}"
          type="${kind === "number" ? "number" : "text"}"
          ${kind === "number" ? 'step="any"' : ""}
          value="${escapeHtml(renderedValue)}"
        />
      `;
    }
    box.appendChild(label);
  }
}

function readForm() {
  const form = $("#jobForm");
  const data = Object.fromEntries(new FormData(form).entries());
  data.cv = form.elements.cv.checked;
  for (const key of ["fold", "gpu", "seed", "use_wandb", "batch_size", "num_epochs"]) {
    if (data[key] !== undefined && data[key] !== "") data[key] = Number(data[key]);
    else delete data[key];
  }
  for (const key of ["save_dir", "folds"]) {
    if (data[key] === "") delete data[key];
  }

  const modelConfig = {};
  document.querySelectorAll(".model-config-input").forEach((input) => {
    const key = input.dataset.key;
    const kind = input.dataset.kind;
    if (!key) return;
    if (kind === "bool") {
      modelConfig[key] = input.checked;
      return;
    }
    const raw = input.value.trim();
    if (raw === "") return;
    if (kind === "number") {
      modelConfig[key] = Number(raw);
      return;
    }
    modelConfig[key] = raw;
  });
  data.model_config = modelConfig;
  return data;
}

function statusPill(status) {
  return `<span class="status ${escapeHtml(status)}">${escapeHtml(status)}</span>`;
}

function shortTime(value) {
  if (!value) return "";
  return value.replace("T", " ");
}

async function loadConfig() {
  runtimeConfig = await api("/api/configs");
  setOptions($("#jobForm select[name='dataset_name']"), runtimeConfig.dataset_names);
  setOptions($("#jobForm select[name='model_name']"), runtimeConfig.model_names);
  $("#runtimeSummary").textContent = `${runtimeConfig.dataset_names.length} datasets, ${runtimeConfig.model_names.length} models`;

  const trainConfig = runtimeConfig.train_config || {};
  const epochsInput = $("#jobForm input[name='num_epochs']");
  const batchInput = $("#jobForm input[name='batch_size']");
  if (trainConfig.num_epochs) epochsInput.placeholder = String(trainConfig.num_epochs);
  if (trainConfig.batch_size) batchInput.placeholder = String(trainConfig.batch_size);
  renderModelConfig();
}

async function loadJobs() {
  const data = await api("/api/jobs");
  $("#jobCount").textContent = `${data.jobs.length} jobs`;
  const body = $("#jobsBody");
  body.innerHTML = "";
  for (const job of data.jobs) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(job.id)}</td>
      <td>${statusPill(job.status)}</td>
      <td>${escapeHtml(job.dataset_name)}</td>
      <td>${escapeHtml(job.model_name)}</td>
      <td>${escapeHtml(job.cv ? job.folds : job.fold)}</td>
      <td>${escapeHtml(job.gpu)}</td>
      <td>${escapeHtml(shortTime(job.created_at))}</td>
      <td><button type="button" data-job="${escapeHtml(job.id)}">Open</button></td>
    `;
    row.querySelector("button").addEventListener("click", () => selectJob(job.id));
    body.appendChild(row);
  }
}

function metaItem(label, value) {
  const safeValue = escapeHtml(value || "");
  return `<div class="meta-item"><span>${escapeHtml(label)}</span><strong title="${safeValue}">${safeValue}</strong></div>`;
}

async function selectJob(jobId) {
  selectedJobId = jobId;
  $("#reloadDetailBtn").disabled = false;
  await loadDetail();
}

async function loadDetail() {
  if (!selectedJobId) return;
  const [job, log, metrics] = await Promise.all([
    api(`/api/jobs/${selectedJobId}`),
    api(`/api/jobs/${selectedJobId}/log?lines=700`),
    api(`/api/jobs/${selectedJobId}/metrics`),
  ]);
  $("#stopBtn").disabled = !["queued", "running"].includes(job.status);
  $("#detailMeta").innerHTML = [
    metaItem("ID", job.id),
    metaItem("Status", job.status),
    metaItem("Dataset", job.dataset_name),
    metaItem("Model", job.model_name),
    metaItem("Fold", job.cv ? job.folds : job.fold),
    metaItem("PID", job.pid || ""),
    metaItem("Started", shortTime(job.started_at)),
    metaItem("Save Dir", job.save_dir),
  ].join("");
  const logBox = $("#logBox");
  logBox.textContent = log || "";
  logBox.scrollTop = logBox.scrollHeight;
  renderMetrics(metrics);
}

function metricBar(label, value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "";
  const numeric = Number(value);
  const pct = Math.max(0, Math.min(100, numeric * 100));
  return `
    <div class="metric-row">
      <span>${escapeHtml(label)}</span>
      <div class="bar"><span style="width:${pct}%"></span></div>
      <span>${numeric.toFixed(4)}</span>
    </div>
  `;
}

function metricRuns(payload) {
  return (payload?.runs || []).filter((run) => run.metrics && run.metrics.length);
}

function runLabel(run, index) {
  const foldMatch = String(run.name || "").match(/fold(\d+)/i);
  if (foldMatch) return `Fold ${foldMatch[1]} - ${run.name}`;
  return `Run ${index + 1} - ${run.name}`;
}

function mean(values) {
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

function averageMetrics(runs) {
  const byEpoch = new Map();
  for (const run of runs) {
    for (const item of run.metrics || []) {
      const epoch = Number(item.epoch);
      if (!Number.isFinite(epoch)) continue;
      if (!byEpoch.has(epoch)) byEpoch.set(epoch, []);
      byEpoch.get(epoch).push(item);
    }
  }

  return [...byEpoch.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([epoch, items]) => {
      const row = { epoch };
      for (const key of ["train_loss", "valid_auc", "valid_acc"]) {
        const values = items.map((item) => Number(item[key])).filter(Number.isFinite);
        if (values.length) row[key] = mean(values);
      }
      row.fold_count = items.length;
      return row;
    });
}

function updateChartRunOptions(payload) {
  const select = $("#chartRunSelect");
  const runs = metricRuns(payload);
  const previous = selectedChartView || select.value;
  select.innerHTML = "";

  if (!runs.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No metrics";
    select.appendChild(option);
    select.disabled = true;
    selectedChartView = "";
    return;
  }

  select.disabled = false;
  if (runs.length > 1) {
    const avg = document.createElement("option");
    avg.value = "__average__";
    avg.textContent = `Average folds (${runs.length})`;
    select.appendChild(avg);
  }

  runs.forEach((run, index) => {
    const option = document.createElement("option");
    option.value = `run:${run.name}`;
    option.textContent = runLabel(run, index);
    select.appendChild(option);
  });

  const values = [...select.options].map((option) => option.value);
  if (values.includes(previous)) {
    select.value = previous;
  } else {
    select.value = runs.length > 1 ? "__average__" : `run:${runs[0].name}`;
  }
  selectedChartView = select.value;
}

function selectedChartSource(payload) {
  const runs = metricRuns(payload);
  if (!runs.length) return null;
  if (runs.length > 1 && selectedChartView === "__average__") {
    return {
      name: `Average folds (${runs.length})`,
      note: "aligned by epoch",
      metrics: averageMetrics(runs),
    };
  }
  const selectedName = selectedChartView.startsWith("run:") ? selectedChartView.slice(4) : "";
  const run = runs.find((item) => item.name === selectedName) || runs[0];
  return {
    name: run.name,
    note: "",
    metrics: run.metrics,
  };
}

function numericPoints(metrics, key) {
  return metrics
    .filter((item) => Number.isFinite(Number(item[key])))
    .map((item) => ({ x: Number(item.epoch), y: Number(item[key]) }));
}

function drawNoData(canvas, label) {
  const ctx = prepareCanvas(canvas);
  const { width, height } = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#66717f";
  ctx.font = "13px Segoe UI, Arial, sans-serif";
  ctx.fillText(label, 18, 28);
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function drawLineChart(canvas, series, options = {}) {
  const ctx = prepareCanvas(canvas);
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const pad = { left: 44, right: 18, top: 18, bottom: 46 };
  const plotW = Math.max(1, width - pad.left - pad.right);
  const plotH = Math.max(1, height - pad.top - pad.bottom);
  const allPoints = series.flatMap((item) => item.points);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  if (!allPoints.length) {
    drawNoData(canvas, "No metrics yet");
    return;
  }

  const xValues = allPoints.map((p) => p.x);
  const yValues = allPoints.map((p) => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  let yMin = options.yMin ?? Math.min(...yValues);
  let yMax = options.yMax ?? Math.max(...yValues);
  if (yMin === yMax) {
    yMin -= 0.5;
    yMax += 0.5;
  }
  const yPad = options.fixedY ? 0 : (yMax - yMin) * 0.08;
  yMin -= yPad;
  yMax += yPad;

  const sx = (x) => pad.left + ((x - xMin) / Math.max(1, xMax - xMin)) * plotW;
  const sy = (y) => pad.top + (1 - (y - yMin) / Math.max(1e-9, yMax - yMin)) * plotH;

  ctx.strokeStyle = "#d9e0e7";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH * i) / 4;
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
  }
  ctx.stroke();

  ctx.strokeStyle = "#aeb9c5";
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();

  ctx.fillStyle = "#66717f";
  ctx.font = "11px Segoe UI, Arial, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const value = yMax - ((yMax - yMin) * i) / 4;
    ctx.fillText(value.toFixed(options.valueDigits ?? 3), pad.left - 8, pad.top + (plotH * i) / 4);
  }
  const tickCount = Math.min(6, Math.max(2, Math.floor(plotW / 90)));
  const xTicks = [];
  for (let i = 0; i < tickCount; i++) {
    const raw = xMin + ((xMax - xMin) * i) / Math.max(1, tickCount - 1);
    const tick = Math.round(raw);
    if (!xTicks.includes(tick)) xTicks.push(tick);
  }
  if (!xTicks.includes(xMin)) xTicks.unshift(xMin);
  if (!xTicks.includes(xMax)) xTicks.push(xMax);

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const tick of xTicks) {
    const x = sx(tick);
    ctx.strokeStyle = "#d9e0e7";
    ctx.beginPath();
    ctx.moveTo(x, height - pad.bottom);
    ctx.lineTo(x, height - pad.bottom + 4);
    ctx.stroke();
    ctx.fillStyle = "#66717f";
    ctx.fillText(String(tick), x, height - pad.bottom + 8);
  }
  ctx.fillStyle = "#66717f";
  ctx.font = "12px Segoe UI, Arial, sans-serif";
  ctx.fillText("Epoch", pad.left + plotW / 2, height - 14);

  for (const item of series) {
    if (!item.points.length) continue;
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    item.points.forEach((point, idx) => {
      const x = sx(point.x);
      const y = sy(point.y);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  let legendX = pad.left;
  for (const item of series) {
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, 8, 10, 3);
    ctx.fillStyle = "#17202a";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(item.label, legendX + 14, 10);
    legendX += 76;
  }
}

function chartConfig(type, payload) {
  const source = selectedChartSource(payload);
  if (!source) return null;
  if (type === "loss") {
    return {
      title: "Loss",
      subtitle: source.note ? `${source.name}, ${source.note}` : source.name,
      series: [{ label: "loss", color: "#1864ab", points: numericPoints(source.metrics, "train_loss") }],
      options: { valueDigits: 4 },
    };
  }
  return {
    title: "AUC / ACC",
    subtitle: source.note ? `${source.name}, ${source.note}` : source.name,
    series: [
      { label: "auc", color: "#087f5b", points: numericPoints(source.metrics, "valid_auc") },
      { label: "acc", color: "#b7791f", points: numericPoints(source.metrics, "valid_acc") },
    ],
    options: { yMin: 0, yMax: 1, fixedY: true, valueDigits: 2 },
  };
}

function renderCharts(payload) {
  latestMetricsPayload = payload;
  updateChartRunOptions(payload);
  const lossConfig = chartConfig("loss", payload);
  const scoreConfig = chartConfig("score", payload);
  if (!lossConfig || !scoreConfig) {
    $("#lossChartLabel").textContent = "";
    $("#scoreChartLabel").textContent = "";
    drawNoData($("#lossChart"), "No loss data");
    drawNoData($("#scoreChart"), "No score data");
    renderExpandedChart();
    return;
  }
  $("#lossChartLabel").textContent = lossConfig.subtitle;
  $("#scoreChartLabel").textContent = scoreConfig.subtitle;
  drawLineChart($("#lossChart"), lossConfig.series, lossConfig.options);
  drawLineChart($("#scoreChart"), scoreConfig.series, scoreConfig.options);
  renderExpandedChart();
}

function openChartModal(type) {
  expandedChartType = type;
  const modal = $("#chartModal");
  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  requestAnimationFrame(renderExpandedChart);
}

function closeChartModal() {
  expandedChartType = null;
  const modal = $("#chartModal");
  modal.classList.remove("open");
  modal.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

function renderExpandedChart() {
  if (!expandedChartType) return;
  const config = chartConfig(expandedChartType, latestMetricsPayload);
  const canvas = $("#expandedChart");
  if (!config) {
    $("#chartModalTitle").textContent = expandedChartType === "loss" ? "Loss" : "AUC / ACC";
    $("#chartModalSubtitle").textContent = "";
    drawNoData(canvas, "No metrics yet");
    return;
  }
  $("#chartModalTitle").textContent = config.title;
  $("#chartModalSubtitle").textContent = config.subtitle;
  drawLineChart(canvas, config.series, config.options);
}

function renderMetrics(payload) {
  const box = $("#metricsList");
  box.innerHTML = "";
  renderCharts(payload);
  if (!payload.runs.length) {
    box.textContent = "No metrics yet.";
    return;
  }
  for (const run of payload.runs) {
    const last = run.metrics[run.metrics.length - 1] || {};
    const best = run.best_metrics || {};
    const card = document.createElement("div");
    card.className = "run-card";
    card.innerHTML = `
      <header>
        <strong title="${escapeHtml(run.name)}">${escapeHtml(run.name)}</strong>
        <span>${run.metrics.length} epochs</span>
      </header>
      ${metricBar("valid_auc", best.valid_auc ?? last.valid_auc)}
      ${metricBar("valid_acc", best.valid_acc ?? last.valid_acc)}
      ${metricBar("test_auc", best.best_test_auc ?? best.last_test_auc)}
      ${metricBar("test_acc", best.best_test_acc ?? best.last_test_acc)}
    `;
    box.appendChild(card);
  }
}

async function submitJob(event) {
  event.preventDefault();
  $("#submitStatus").textContent = "Submitting...";
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify(readForm()),
    });
    $("#submitStatus").textContent = `Started ${job.id}`;
    await loadJobs();
    await selectJob(job.id);
  } catch (error) {
    $("#submitStatus").textContent = "Submit failed";
    alert(error.message);
  }
}

async function stopSelectedJob() {
  if (!selectedJobId) return;
  await api(`/api/jobs/${selectedJobId}/stop`, { method: "POST" });
  await loadJobs();
  await loadDetail();
}

async function refreshAll() {
  await loadJobs();
  if (selectedJobId) await loadDetail();
}

async function boot() {
  $("#jobForm").addEventListener("submit", submitJob);
  $("#jobForm select[name='model_name']").addEventListener("change", renderModelConfig);
  $("#refreshBtn").addEventListener("click", refreshAll);
  $("#reloadDetailBtn").addEventListener("click", loadDetail);
  $("#stopBtn").addEventListener("click", stopSelectedJob);
  $("#chartRunSelect").addEventListener("change", (event) => {
    selectedChartView = event.target.value;
    if (latestMetricsPayload) renderCharts(latestMetricsPayload);
  });
  $("#chartModalClose").addEventListener("click", closeChartModal);
  $("#chartModal").addEventListener("click", (event) => {
    if (event.target.id === "chartModal") closeChartModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && expandedChartType) closeChartModal();
  });
  document.querySelectorAll("[data-chart]").forEach((node) => {
    node.addEventListener("click", () => openChartModal(node.dataset.chart));
  });
  window.addEventListener("resize", () => {
    if (latestMetricsPayload) renderCharts(latestMetricsPayload);
  });
  await loadConfig();
  await loadJobs();
  setInterval(refreshAll, 5000);
}

boot().catch((error) => {
  $("#runtimeSummary").textContent = "Failed to load runtime config";
  console.error(error);
});
