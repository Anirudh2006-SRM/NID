/**
 * SENTINEL-HYBRID NIDS - SOC DASHBOARD CONTROLLER
 * Handles live telemetry streaming, sandbox evaluation, SHAP explainability,
 * and 2D Decision Space Canvas rendering.
 */

// Application State
const state = {
  presets: [],
  activePresetId: null,
  streamActive: true,
  streamIntervalId: null,
  streamFilter: "ALL",
  streamPackets: [],
  stats: {
    totalFlows: 1248,
    knownAttacks: 412,
    unseenThreats: 64,
    lastLatencyMs: 1.82,
  },
  thresholds: {
    anomaly: 0.4782,
    confidence: 0.65,
  },
  currentPoint: {
    confidence: 0.968,
    anomalyScore: 0.421,
    decision: "NORMAL",
  },
};

// DOM Elements
const elements = {
  presetContainer: document.getElementById("preset-container"),
  presetDesc: document.getElementById("preset-description"),
  flowForm: document.getElementById("flow-form"),
  analyzeBtn: document.getElementById("analyze-btn"),
  streamList: document.getElementById("stream-list"),
  clearStreamBtn: document.getElementById("clear-stream-btn"),
  toggleStreamBtn: document.getElementById("toggle-stream-btn"),
  streamBtnLabel: document.getElementById("stream-btn-label"),
  filterChips: document.querySelectorAll(".filter-chip"),
  
  // Stats
  statTotalFlows: document.getElementById("stat-total-flows"),
  statKnownAttacks: document.getElementById("stat-known-attacks"),
  statUnseenThreats: document.getElementById("stat-unseen-threats"),
  statAnomalyThreshold: document.getElementById("stat-anomaly-threshold"),
  statInferenceLatency: document.getElementById("stat-inference-latency"),
  threatPill: document.getElementById("global-threat-pill"),
  threatLevelText: document.getElementById("threat-level-text"),

  // Verdict Card
  verdictDecisionBadge: document.getElementById("verdict-decision-badge"),
  verdictRiskBadge: document.getElementById("verdict-risk-badge"),
  verdictLatency: document.getElementById("verdict-latency"),
  verdictPredCat: document.getElementById("verdict-pred-cat"),
  verdictConfBar: document.getElementById("verdict-conf-bar"),
  verdictAnomalyScore: document.getElementById("verdict-anomaly-score"),
  verdictAnomBar: document.getElementById("verdict-anom-bar"),
  verdictExplanationText: document.getElementById("verdict-explanation-text"),

  // SHAP & Canvas
  shapBarList: document.getElementById("shap-bar-list"),
  decisionCanvas: document.getElementById("decision-canvas"),
  toast: document.getElementById("toast"),
};

// ==========================================================================
// INITIALIZATION
// ==========================================================================
document.addEventListener("DOMContentLoaded", async () => {
  setupEventListeners();
  drawDecisionCanvas();
  await loadPresets();
  await fetchStreamBatch();
  startStream();
});

function setupEventListeners() {
  // Analyze Button
  elements.analyzeBtn.addEventListener("click", (e) => {
    e.preventDefault();
    evaluateCurrentFlow();
  });

  // Stream Toggle
  elements.toggleStreamBtn.addEventListener("click", () => {
    if (state.streamActive) {
      stopStream();
      elements.streamBtnLabel.textContent = "Resume Stream";
      elements.toggleStreamBtn.classList.remove("btn-secondary");
      elements.toggleStreamBtn.classList.add("btn-primary");
      showToast("Stream ingestion paused.");
    } else {
      startStream();
      elements.streamBtnLabel.textContent = "Pause Stream";
      elements.toggleStreamBtn.classList.add("btn-secondary");
      elements.toggleStreamBtn.classList.remove("btn-primary");
      showToast("Live stream ingestion resumed.");
    }
  });

  // Clear Stream
  elements.clearStreamBtn.addEventListener("click", () => {
    state.streamPackets = [];
    renderStreamList();
    showToast("Stream log cleared.");
  });

  // Stream Filters
  elements.filterChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      elements.filterChips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.streamFilter = chip.dataset.filter;
      renderStreamList();
    });
  });

  // Window resize re-draws canvas
  window.addEventListener("resize", drawDecisionCanvas);
}

// ==========================================================================
// PRESETS & FLOW FORM
// ==========================================================================
async function loadPresets() {
  try {
    const res = await fetch("/api/sample-flows");
    if (!res.ok) throw new Error("Failed to load presets");
    state.presets = await res.json();

    elements.presetContainer.innerHTML = "";
    state.presets.forEach((preset, idx) => {
      const chip = document.createElement("button");
      chip.className = `preset-chip ${idx === 0 ? "active" : ""}`;
      chip.innerHTML = `<span>${getPresetIcon(preset.category)}</span> ${preset.name}`;
      chip.addEventListener("click", () => selectPreset(preset.id));
      elements.presetContainer.appendChild(chip);
    });

    if (state.presets.length > 0) {
      selectPreset(state.presets[0].id);
    }
  } catch (err) {
    console.error("Error loading presets:", err);
    showToast("Could not load presets from server.");
  }
}

function getPresetIcon(category) {
  switch (category) {
    case "Normal": return "🛡️";
    case "DoS": return "⚡";
    case "Exploits": return "💥";
    case "Generic": return "🔑";
    case "Reconnaissance": return "🔍";
    case "Shellcode": return "💉";
    case "Worms": return "⚠️";
    default: return "📦";
  }
}

function selectPreset(presetId) {
  const preset = state.presets.find((p) => p.id === presetId);
  if (!preset) return;

  state.activePresetId = presetId;

  // Update active chip UI
  const chips = elements.presetContainer.querySelectorAll(".preset-chip");
  chips.forEach((c) => c.classList.remove("active"));
  const idx = state.presets.findIndex((p) => p.id === presetId);
  if (chips[idx]) chips[idx].classList.add("active");

  elements.presetDesc.textContent = preset.description;

  // Populate form fields
  populateForm(preset.flow);

  // Trigger evaluation automatically
  evaluateCurrentFlow();
}

function populateForm(flowData) {
  for (const [key, value] of Object.entries(flowData)) {
    const field = elements.flowForm.elements[key];
    if (field) {
      field.value = value;
    }
  }
}

function getFormData() {
  const data = {};
  const formData = new FormData(elements.flowForm);
  for (const [key, val] of formData.entries()) {
    const num = Number(val);
    data[key] = isNaN(num) ? val : num;
  }
  return data;
}

// ==========================================================================
// INFERENCE & EXPLAINABILITY
// ==========================================================================
async function evaluateCurrentFlow() {
  const flow = getFormData();
  elements.analyzeBtn.disabled = true;
  elements.analyzeBtn.innerHTML = `<span>Evaluating Flow...</span>`;

  try {
    // 1. Predict
    const predPromise = fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(flow),
    });

    // 2. Explain with SHAP
    const explainPromise = fetch("/api/explain?top_n=6", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(flow),
    });

    const [predRes, explainRes] = await Promise.all([predPromise, explainPromise]);
    if (!predRes.ok) throw new Error("Prediction API failed");
    const pred = await predRes.json();

    let explainData = null;
    if (explainRes.ok) {
      explainData = await explainRes.json();
    }

    // Update Verdict UI
    renderVerdict(pred);

    // Update SHAP UI
    if (explainData && explainData.top_features) {
      renderShapBars(explainData.top_features);
    }

    // Update 2D Decision Space
    state.currentPoint = {
      confidence: pred.confidence,
      anomalyScore: pred.anomaly_score,
      decision: pred.decision,
    };
    drawDecisionCanvas();

    // Increment local stats
    state.stats.totalFlows++;
    if (pred.decision === "KNOWN ATTACK") state.stats.knownAttacks++;
    if (pred.decision === "UNSEEN / ANOMALOUS") state.stats.unseenThreats++;
    state.stats.lastLatencyMs = pred.inference_time_ms;
    updateStatsDisplay();

    showToast(`Evaluation Complete: ${pred.decision}`);
  } catch (err) {
    console.error("Evaluation error:", err);
    showToast("Evaluation failed: " + err.message);
  } finally {
    elements.analyzeBtn.disabled = false;
    elements.analyzeBtn.innerHTML = `
      <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <span>Evaluate Flow with Hybrid Decision Engine</span>
    `;
  }
}

function renderVerdict(pred) {
  // Decision Badge
  elements.verdictDecisionBadge.textContent = pred.decision;
  elements.verdictDecisionBadge.className = "verdict-badge";
  if (pred.decision === "NORMAL") {
    elements.verdictDecisionBadge.classList.add("verdict-normal");
  } else if (pred.decision === "KNOWN ATTACK") {
    elements.verdictDecisionBadge.classList.add("verdict-attack");
  } else {
    elements.verdictDecisionBadge.classList.add("verdict-unseen");
  }

  // Risk Badge
  elements.verdictRiskBadge.textContent = `${pred.risk_level} RISK`;
  elements.verdictRiskBadge.className = "risk-badge";
  if (pred.risk_level === "LOW") {
    elements.verdictRiskBadge.classList.add("risk-low");
  } else if (pred.risk_level === "CRITICAL") {
    elements.verdictRiskBadge.classList.add("risk-critical");
  } else {
    elements.verdictRiskBadge.classList.add("risk-high");
  }

  // Latency
  elements.verdictLatency.textContent = `Latency: ${pred.inference_time_ms} ms`;

  // Supervised Metrics
  const confPct = Math.round(pred.confidence * 1000) / 10;
  elements.verdictPredCat.textContent = `${pred.predicted_category} (${confPct}%)`;
  elements.verdictConfBar.style.width = `${Math.min(100, confPct)}%`;

  // Anomaly Metrics
  elements.verdictAnomalyScore.textContent = `${pred.anomaly_score.toFixed(4)} / ${pred.anomaly_threshold.toFixed(4)}`;
  const anomPct = Math.min(100, Math.round((pred.anomaly_score / 0.7) * 100));
  elements.verdictAnomBar.style.width = `${anomPct}%`;
  elements.verdictAnomBar.className = "progress-bar-fill " + (pred.is_anomalous ? "fill-red" : "fill-green");

  // Rationale
  elements.verdictExplanationText.textContent = pred.explanation;

  // Global Threat Level
  updateGlobalThreatLevel(pred.decision);
}

function updateGlobalThreatLevel(decision) {
  if (decision === "UNSEEN / ANOMALOUS") {
    elements.threatLevelText.textContent = "DEFCON 1: UNSEEN THREAT";
    elements.threatPill.className = "status-pill";
    elements.threatPill.style.color = "#6d28d9";
    elements.threatPill.style.borderColor = "#ddd6fe";
    elements.threatPill.style.background = "#f5f3ff";
  } else if (decision === "KNOWN ATTACK") {
    elements.threatLevelText.textContent = "DEFCON 2: ATTACK DETECTED";
    elements.threatPill.className = "status-pill";
    elements.threatPill.style.color = "#dc2626";
    elements.threatPill.style.borderColor = "#fecaca";
    elements.threatPill.style.background = "#fef2f2";
  } else {
    elements.threatLevelText.textContent = "DEFCON 5: NORMAL";
    elements.threatPill.className = "status-pill threat-pill";
    elements.threatPill.style.color = "";
    elements.threatPill.style.borderColor = "";
    elements.threatPill.style.background = "";
  }
}

function renderShapBars(features) {
  elements.shapBarList.innerHTML = "";

  if (!features || features.length === 0) {
    elements.shapBarList.innerHTML = `<div class="text-muted font-mono text-xs">No SHAP attribution values available.</div>`;
    return;
  }

  const maxVal = Math.max(...features.map((f) => Math.abs(f.shap_value)), 0.05);

  features.forEach((feat) => {
    const isRisk = feat.shap_value > 0;
    const widthPct = Math.min(100, (Math.abs(feat.shap_value) / maxVal) * 100);

    const item = document.createElement("div");
    item.className = "shap-bar-item";
    item.innerHTML = `
      <div class="shap-feat-header font-mono">
        <span class="shap-feat-name">${cleanFeatureName(feat.feature)}</span>
        <span class="shap-feat-val">${feat.shap_value > 0 ? "+" : ""}${feat.shap_value.toFixed(4)} (val: ${feat.feature_value})</span>
      </div>
      <div class="shap-bar-track">
        <div class="shap-bar-fill ${isRisk ? "shap-fill-risk" : "shap-fill-safe"}" style="width: ${widthPct}%;"></div>
      </div>
    `;
    elements.shapBarList.appendChild(item);
  });
}

function cleanFeatureName(name) {
  return name.replace(/^num_/, "").replace(/^cat_/, "");
}

// ==========================================================================
// 2D HYBRID DECISION CANVAS (LIGHT THEME)
// ==========================================================================
function drawDecisionCanvas() {
  const canvas = elements.decisionCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Padding
  const pad = { top: 20, right: 20, bottom: 35, left: 45 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  // Scales
  const yMin = 0.35;
  const yMax = 0.65;
  const confThresh = state.thresholds.confidence; // 0.65
  const anomThresh = state.thresholds.anomaly;    // 0.4782

  const xToPix = (conf) => pad.left + conf * plotW;
  const yToPix = (anom) => pad.top + ((yMax - anom) / (yMax - yMin)) * plotH;

  // 1. Draw Shaded Decision Regions (Soft light tints)
  const splitX = xToPix(confThresh);
  const splitY = yToPix(anomThresh);

  // Region: Known Attack (Right side)
  ctx.fillStyle = "rgba(220, 38, 38, 0.08)";
  ctx.fillRect(splitX, pad.top, plotW - (splitX - pad.left), plotH);

  // Region: Unseen Threat (Top-left)
  ctx.fillStyle = "rgba(124, 58, 237, 0.10)";
  ctx.fillRect(pad.left, pad.top, splitX - pad.left, splitY - pad.top);

  // Region: Normal (Bottom-left)
  ctx.fillStyle = "rgba(5, 150, 105, 0.08)";
  ctx.fillRect(pad.left, splitY, splitX - pad.left, plotH - (splitY - pad.top));

  // 2. Draw Threshold Lines
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1.5;

  // Anomaly boundary (Horizontal)
  ctx.strokeStyle = "#0284c7";
  ctx.beginPath();
  ctx.moveTo(pad.left, splitY);
  ctx.lineTo(pad.left + plotW, splitY);
  ctx.stroke();

  // Confidence boundary (Vertical)
  ctx.strokeStyle = "#64748b";
  ctx.beginPath();
  ctx.moveTo(splitX, pad.top);
  ctx.lineTo(splitX, pad.top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // 3. Draw Axis Lines & Ticks (Crisp Light Mode)
  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  ctx.fillStyle = "#64748b";
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.fillText("0.0", xToPix(0.0), pad.top + plotH + 14);
  ctx.fillText("0.5", xToPix(0.5), pad.top + plotH + 14);
  ctx.fillText("1.0 (Conf)", xToPix(1.0), pad.top + plotH + 14);

  ctx.textAlign = "right";
  ctx.fillText("0.65", pad.left - 6, yToPix(0.65) + 3);
  ctx.fillText("0.48", pad.left - 6, splitY + 3);
  ctx.fillText("0.35", pad.left - 6, yToPix(0.35) + 3);

  // 4. Draw Current Evaluated Flow Crosshair Point
  const ptX = Math.max(pad.left, Math.min(pad.left + plotW, xToPix(state.currentPoint.confidence)));
  const ptY = Math.max(pad.top, Math.min(pad.top + plotH, yToPix(state.currentPoint.anomalyScore)));

  // Target Glow
  const grad = ctx.createRadialGradient(ptX, ptY, 2, ptX, ptY, 14);
  grad.addColorStop(0, "rgba(2, 132, 199, 0.5)");
  grad.addColorStop(1, "rgba(2, 132, 199, 0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(ptX, ptY, 14, 0, Math.PI * 2);
  ctx.fill();

  // Core Dot
  ctx.fillStyle = "#0284c7";
  ctx.beginPath();
  ctx.arc(ptX, ptY, 4, 0, Math.PI * 2);
  ctx.fill();

  // Crosshair Lines
  ctx.strokeStyle = "#0284c7";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(ptX - 8, ptY);
  ctx.lineTo(ptX + 8, ptY);
  ctx.moveTo(ptX, ptY - 8);
  ctx.lineTo(ptX, ptY + 8);
  ctx.stroke();
}

// ==========================================================================
// LIVE INGESTION STREAM
// ==========================================================================
function startStream() {
  state.streamActive = true;
  if (!state.streamIntervalId) {
    state.streamIntervalId = setInterval(fetchStreamBatch, 3200);
  }
}

function stopStream() {
  state.streamActive = false;
  if (state.streamIntervalId) {
    clearInterval(state.streamIntervalId);
    state.streamIntervalId = null;
  }
}

async function fetchStreamBatch() {
  if (!state.streamActive) return;
  try {
    const res = await fetch("/api/stream?count=4");
    if (!res.ok) return;
    const data = await res.json();
    if (data.packets && data.packets.length > 0) {
      // Prepend packets
      data.packets.forEach((p) => {
        state.streamPackets.unshift(p);
      });
      // Cap at 40 rows
      if (state.streamPackets.length > 40) {
        state.streamPackets = state.streamPackets.slice(0, 40);
      }
      renderStreamList();
    }
  } catch (err) {
    console.error("Stream polling error:", err);
  }
}

function renderStreamList() {
  elements.streamList.innerHTML = "";

  const filtered = state.streamPackets.filter((p) => {
    if (state.streamFilter === "ALL") return true;
    return p.decision === state.streamFilter;
  });

  if (filtered.length === 0) {
    elements.streamList.innerHTML = `<div class="text-muted font-mono text-xs" style="padding: 20px; text-align: center;">No matching packets for filter: ${state.streamFilter}</div>`;
    return;
  }

  filtered.forEach((p) => {
    const row = document.createElement("div");
    row.className = "stream-row " + getRowClass(p.decision);

    const timeStr = new Date(p.timestamp * 1000).toTimeString().split(" ")[0];
    const rateKb = (p.rate / 1000).toFixed(1) + "k";
    const bytesTotal = p.sbytes + p.dbytes;

    row.innerHTML = `
      <span class="stream-time">${timeStr}</span>
      <span class="stream-proto">${p.proto.toUpperCase()}/${p.service}</span>
      <span class="stream-bytes">${rateKb} p/s · ${bytesTotal}B</span>
      <span class="stream-score ${p.anomaly_score >= state.thresholds.anomaly ? "text-danger" : "text-success"}">${p.anomaly_score.toFixed(3)}</span>
      <span class="verdict-tag ${getTagClass(p.decision)}">${p.decision === "UNSEEN / ANOMALOUS" ? "UNSEEN" : p.decision}</span>
    `;

    // Clicking a stream row loads it into the sandbox
    row.addEventListener("click", () => {
      populateForm(p);
      evaluateCurrentFlow();
      showToast(`Loaded ${p.proto.toUpperCase()} packet into inspector`);
    });

    elements.streamList.appendChild(row);
  });
}

function getRowClass(decision) {
  if (decision === "NORMAL") return "row-normal";
  if (decision === "KNOWN ATTACK") return "row-attack";
  return "row-unseen";
}

function getTagClass(decision) {
  if (decision === "NORMAL") return "tag-normal";
  if (decision === "KNOWN ATTACK") return "tag-attack";
  return "tag-unseen";
}

// ==========================================================================
// UTILITIES & STATS
// ==========================================================================
function updateStatsDisplay() {
  elements.statTotalFlows.textContent = state.stats.totalFlows.toLocaleString();
  elements.statKnownAttacks.textContent = state.stats.knownAttacks.toLocaleString();
  elements.statUnseenThreats.textContent = state.stats.unseenThreats.toLocaleString();
  elements.statInferenceLatency.textContent = `${state.stats.lastLatencyMs} ms`;
}

let toastTimer = null;
function showToast(message) {
  if (!elements.toast) return;
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    elements.toast.classList.remove("show");
  }, 2800);
}
