/* ============================================================================
   DisOps — Live Disaster Response Agent (frontend)

   Connects to the FastAPI backend over a WebSocket and renders the full
   vision-to-voice pipeline live. Components:
     LiveVideoFeed · UrgencyBadge · GuidancePanel · PipelineStatusGraph
     SceneAnalysisCard · AgentReasoningCard · EventTimeline · SessionControls
     VoiceWaveform · SafetyNotice
   ============================================================================ */

const API = {
  start: () => post("/api/start"),
  stop: () => post("/api/stop"),
  pause: (enabled) => post("/api/pause", { enabled }),
  mute: (enabled) => post("/api/mute", { enabled }),
  demoMode: (enabled) => post("/api/demo-mode", { enabled }),
  scenario: (scenario) => post("/api/scenario", { scenario }),
  snapshot: () => post("/api/snapshot"),
  replay: () => post("/api/replay"),
  clear: () => post("/api/clear"),
  frame: (frame) => post("/api/frame", { frame }),
};

async function post(url, body) {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : "{}",
    });
    return res.ok ? res.json() : null;
  } catch {
    return null;
  }
}

/* --------------------------------------------------------------- tiny helpers */
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v === true) node.setAttribute(k, "");
    else if (v !== false && v != null) node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}
const $ = (sel) => document.querySelector(sel);
const titleCase = (s) => String(s).replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());

/* ----------------------------------------------------------------- SVG icons */
const ICON = {
  camera: '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
  shard: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  eye: '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
  cloud: '<path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/>',
  brain: '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44A2.5 2.5 0 0 1 4 17.5a2.5 2.5 0 0 1-1-4.78A2.5 2.5 0 0 1 4.5 8 2.5 2.5 0 0 1 7 4.5 2.5 2.5 0 0 1 9.5 2z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44A2.5 2.5 0 0 0 20 17.5a2.5 2.5 0 0 0 1-4.78A2.5 2.5 0 0 0 19.5 8 2.5 2.5 0 0 0 17 4.5 2.5 2.5 0 0 0 14.5 2z"/>',
  voice: '<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/>',
  speaker: '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>',
  arrow: '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
  flame: '<path d="M12 2s4 4 4 8a4 4 0 0 1-8 0c0-1 .5-2 1-3-2 1-4 3-4 6a7 7 0 0 0 14 0c0-5-7-11-7-11z"/>',
  wave: '<path d="M2 12c2-3 4-3 6 0s4 3 6 0 4-3 6 0"/><path d="M2 18c2-3 4-3 6 0s4 3 6 0 4-3 6 0"/>',
  blocks: '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
  block: '<circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>',
  person: '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
  alert: '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
  phone: '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.96.36 1.9.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.34 1.85.57 2.81.7A2 2 0 0 1 22 16.92z"/>',
  replay: '<polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>',
  pause: '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>',
  play: '<polygon points="5 3 19 12 5 21 5 3"/>',
  snapshot: '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
  trash: '<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
  mute: '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>',
};
function svg(name, w = 18) {
  return `<svg viewBox="0 0 24 24" width="${w}" height="${w}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICON[name] || ""}</svg>`;
}

/* hazard → display label + rough overlay box position (% based) */
const HAZARD_BOXES = {
  fire: { label: "Fire", top: 55, left: 8, w: 26, h: 35 },
  smoke: { label: "Smoke", top: 8, left: 12, w: 40, h: 45 },
  flood_water: { label: "Flood Water", top: 62, left: 6, w: 70, h: 32 },
  debris: { label: "Debris", top: 20, left: 30, w: 45, h: 40 },
  blocked_exit: { label: "Blocked Exit", top: 18, left: 60, w: 30, h: 60 },
  low_visibility: { label: "Low Visibility", top: 5, left: 5, w: 90, h: 30 },
  falling_object: { label: "Falling Object", top: 5, left: 40, w: 28, h: 30 },
  crowd: { label: "Crowd", top: 35, left: 35, w: 50, h: 50 },
  person_down: { label: "Person Down", top: 60, left: 25, w: 45, h: 30 },
  gas_leak: { label: "Gas Leak", top: 25, left: 55, w: 30, h: 35 },
  medical: { label: "Medical", top: 58, left: 30, w: 40, h: 30 },
};

const STAGES = [];
const CHIPS = [];

/* ============================================================================
   APP STATE — populated from /api/state on load (not hardcoded)
   ============================================================================ */
const state = {
  status: { running: false, paused: false, muted: false, demo_mode: false, frame_interval_s: 2.5 },
  config: null,
  scenarios: [],
  event: null,
  log: [],
  wsConnected: false,
};

function pipelineStages() {
  return state.config?.pipeline?.stages || STAGES;
}
function statusChips() {
  return state.config?.pipeline?.chips || CHIPS;
}
function idleGuidance() {
  return state.config?.idle_guidance || "Monitoring your environment. I'll speak up if I detect danger.";
}
function idleSituation() {
  return state.config?.idle_situation || "Monitoring environment…";
}
function visionLabel() {
  return state.config?.vision_model_label || "Vision";
}
function reasoningLabel() {
  return state.config?.reasoning_model_label || "Reasoning";
}
function frameIntervalMs() {
  return Math.round((state.status.frame_interval_s || 2.5) * 1000);
}

let mediaStream = null;
let frameTimer = null;
let speaking = false;

/* ============================================================================
   COMPONENT: LiveVideoFeed
   ============================================================================ */
function renderVideoPanel() {
  const ev = state.event;
  const hazards = ev?.detected_hazards || [];
  const running = state.status.running;
  const visionStatus = ev?.pipeline_status?.vision_models || (running ? "complete" : "idle");

  const overlay = el("div", { class: "hazard-overlay" });
  hazards.forEach((h) => {
    const b = HAZARD_BOXES[h];
    if (!b) return;
    overlay.append(
      el("div", { class: "hazard-box", style: `top:${b.top}%;left:${b.left}%;width:${b.w}%;height:${b.h}%` }, [
        el("span", { class: "hazard-box__label", text: b.label }),
      ])
    );
  });

  const stage = el("div", { class: "video-stage" }, [
    el("video", { id: "cam", autoplay: true, muted: true, playsinline: true }),
    el("canvas", { id: "cam-fallback", class: "video-fallback" }),
    el("div", { class: "video-scrim" }),
    overlay,
    el("div", { class: running ? "live-badge" : "live-badge is-idle" }, [
      el("span", { class: "live-badge__dot" }),
      running ? "LIVE" : "OFFLINE",
    ]),
    el("div", { class: "frame-time", id: "frame-time", text: ev ? formatClock(ev.timestamp) : "--:--:--" }),
    hazards.length === 0
      ? el("div", { class: "monitoring-tag" }, [el("span", { class: "scan-dot" }), "Monitoring environment…"])
      : null,
  ]);

  const meta = el("div", { class: "video-meta" }, [
    metaItem("Frame shard", ev ? ev.frame_id : "—"),
    metaItem("Last analyzed", ev ? formatClock(ev.timestamp) : "—"),
    metaItem("Frame interval", `${state.status.frame_interval_s || 2.5}s`),
    metaItem("Vision backend", titleCase(visionStatus), visionStatus === "complete" ? "ok" : "off"),
  ]);

  const panel = $("#video-panel");
  panel.replaceChildren(stage, meta);

  // (re)attach camera to the freshly-created <video> if we have a stream
  if (mediaStream) attachStream();
  else drawFallback();
}
function metaItem(label, value, cls = "") {
  return el("div", { class: "video-meta__item" }, [
    el("div", { class: "video-meta__label", text: label }),
    el("div", { class: `video-meta__value ${cls}`, text: value }),
  ]);
}

/* ============================================================================
   COMPONENT: GuidancePanel (+ UrgencyBadge + VoiceWaveform)
   ============================================================================ */
function renderGuidancePanel() {
  const ev = state.event;
  const urgency = ev?.urgency || "Low";
  const guidance = ev?.voice_guidance || idleGuidance();
  const situation = ev?.situation || idleSituation();
  const lowConf = ev && ev.confidence < 0.6;

  const guidanceText = lowConf
    ? `I'm not fully certain, but ${guidance.charAt(0).toLowerCase()}${guidance.slice(1)}`
    : guidance;

  const bars = el("div", { class: "waveform__bars" });
  for (let i = 0; i < 28; i++) bars.append(el("span"));

  const body = el("div", { class: "guidance-panel__body" }, [
    el("div", { class: "urgency-badge" }, [el("span", { class: "urgency-badge__ring" }), urgency]),
    el("div", { class: "situation-title", text: situation }),
    el("div", { class: "guidance-card" }, [
      el("div", { class: "guidance-card__label", text: "Spoken guidance" }),
      el("div", { class: "guidance-card__text", id: "guidance-text", text: guidanceText }),
    ]),
    el("div", { class: `waveform ${speaking ? "is-playing" : ""}`, id: "waveform" }, [
      el("div", { class: "waveform__icon", html: svg("voice", 20) }),
      bars,
      el("div", { class: "waveform__status", id: "wave-status", text: speaking ? "Voice streaming…" : (state.status.muted ? "Voice muted" : "Voice idle") }),
    ]),
    el("div", { class: "guidance-actions" }, [
      el("button", { class: "btn", html: svg("replay", 16) + "<span>Replay Guidance</span>", onclick: () => API.replay() }),
      el("button", {
        class: state.status.muted ? "btn btn--active" : "btn",
        id: "mute-btn",
        html: svg(state.status.muted ? "mute" : "voice", 16) + `<span>${state.status.muted ? "Unmute Voice" : "Mute Voice"}</span>`,
        onclick: toggleMute,
      }),
      el("button", { class: "btn btn--danger btn--call", html: svg("phone", 16) + "<span>Call Emergency Services</span>", onclick: callEmergency }),
    ]),
  ]);

  $("#guidance-panel").replaceChildren(
    el("div", { class: "panel__head" }, [el("h3", { text: "Emergency Guidance" }), el("span", { class: "panel__sub", text: ev ? `confidence ${(ev.confidence * 100) | 0}%` : "" })]),
    body
  );
}

/* ============================================================================
   COMPONENT: PipelineStatusGraph
   ============================================================================ */
function renderPipeline(statuses = {}, durations = {}) {
  const graph = $("#pipeline-graph");
  const stages = pipelineStages();
  const nodes = [];
  stages.forEach((s, i) => {
    const status = statuses[s.key] || "idle";
    const dur = durations[s.key];
    nodes.push(
      el("div", { class: "pstage", "data-status": status, "data-key": s.key }, [
        el("div", { class: "pstage__top" }, [
          el("div", { class: "pstage__icon", html: svg(s.icon, 16) }),
          el("div", { class: "pstage__name", text: s.name }),
        ]),
        el("div", { class: "pstage__meta" }, [
          el("span", { class: "pstage__status", text: status }),
          el("span", { class: "pstage__dur" }, [el("span", { class: "pstage__spinner" }), dur != null ? `${dur}ms` : ""]),
        ]),
      ])
    );
    if (i < stages.length - 1) {
      nodes.push(el("div", { class: "pstage__connector", html: svg("arrow", 18) }));
    }
  });
  graph.replaceChildren(...nodes);
}
function updatePipelineProgress(data) {
  const { statuses = {}, duration_ms, active } = data;
  pipelineStages().forEach((s) => {
    const node = $(`.pstage[data-key="${s.key}"]`);
    if (!node) return;
    const status = statuses[s.key] || "idle";
    node.setAttribute("data-status", status);
    node.querySelector(".pstage__status").textContent = status;
    if (s.key === active && duration_ms != null) {
      const dur = node.querySelector(".pstage__dur");
      dur.replaceChildren(el("span", { class: "pstage__spinner" }), document.createTextNode(`${duration_ms}ms`));
    }
  });
}

/* ============================================================================
   COMPONENT: SceneAnalysisCard
   ============================================================================ */
function renderSceneAnalysis() {
  const ev = state.event;
  const body = el("div", { class: "kv" }, [
    kvRow("Scene Summary", ev?.scene_summary || idleSituation()),
    kvRowTags("Detected Hazards", ev?.detected_hazards || [], true),
    kvRowTags("Detected Objects", ev?.detected_objects || [], false),
    kvRow("People Detected", String(ev?.people_detected ?? 0)),
    kvRowTags("Environment Conditions", ev?.environment_conditions || [], false),
    confidenceRow(ev?.confidence ?? 0),
    kvRow("Motion Changes", ev?.motion_changes || "Stable"),
    kvRow("Recommended Next Check", ev?.next_check || "Continue monitoring for changes.", true),
  ]);
  $("#scene-analysis").replaceChildren(
    el("div", { class: "panel__head" }, [el("h3", { text: "Scene Analysis" }), el("span", { class: "panel__sub", text: visionLabel() })]),
    el("div", { class: "panel__body" }, [body])
  );
}

/* ============================================================================
   COMPONENT: AgentReasoningCard
   ============================================================================ */
function renderReasoning() {
  const ev = state.event;
  const body = el("div", { class: "kv" }, [
    kvRow("Situation", ev?.situation || "No active incident."),
    el("div", { class: "kv__row" }, [
      el("div", { class: "kv__label", text: "Risk Level" }),
      el("span", { class: "risk-pill", text: ev?.risk_level || ev?.urgency || "Low" }),
    ]),
    kvRow("Safest Next Action", ev?.safest_next_action || "Proceed normally and stay aware."),
    kvRow("Reasoning", ev?.reasoning_summary || "Scene is being monitored for hazard signatures.", true),
    kvRow("Uncertainty", ev?.uncertainty || "None significant.", true),
  ]);
  $("#agent-reasoning").replaceChildren(
    el("div", { class: "panel__head" }, [el("h3", { text: "Agent Reasoning" }), el("span", { class: "panel__sub", text: reasoningLabel() })]),
    el("div", { class: "panel__body" }, [body])
  );
}

function kvRow(label, value, dim = false) {
  return el("div", { class: "kv__row" }, [
    el("div", { class: "kv__label", text: label }),
    el("div", { class: `kv__value ${dim ? "dim" : ""}`, text: value }),
  ]);
}
function kvRowTags(label, items, hazard) {
  const tags = el("div", { class: "tags" });
  if (!items || items.length === 0) tags.append(el("span", { class: "tag tag--empty", text: "none detected" }));
  else items.forEach((t) => tags.append(el("span", { class: hazard ? "tag tag--hazard" : "tag", text: titleCase(t) })));
  return el("div", { class: "kv__row" }, [el("div", { class: "kv__label", text: label }), tags]);
}
function confidenceRow(conf) {
  const pct = Math.round((conf || 0) * 100);
  return el("div", { class: "kv__row" }, [
    el("div", { class: "kv__label", text: "Confidence Score" }),
    el("div", { class: "confidence" }, [
      el("div", { class: "confidence__track" }, [el("div", { class: "confidence__fill", style: `width:${pct}%` })]),
      el("div", { class: "confidence__num", text: `${pct}%` }),
    ]),
  ]);
}

/* ============================================================================
   COMPONENT: EventTimeline
   ============================================================================ */
function renderTimeline() {
  const body = el("div", { class: "timeline", id: "timeline-list" });
  state.log.slice(-40).forEach((entry) => body.append(logLine(entry)));
  $("#event-timeline").replaceChildren(
    el("div", { class: "panel__head" }, [el("h3", { text: "Event Timeline" }), el("span", { class: "panel__sub", text: "live log" })]),
    el("div", { class: "panel__body" }, [body])
  );
  scrollTimeline();
}
function logLine(entry) {
  return el("div", { class: "log-line", "data-level": entry.level || "info" }, [
    el("span", { class: "log-line__time", text: entry.time }),
    el("span", { class: "log-line__dot" }),
    el("span", { class: "log-line__msg", text: entry.message }),
  ]);
}
function appendLog(entry) {
  state.log.push(entry);
  if (state.log.length > 60) state.log = state.log.slice(-60);
  const list = $("#timeline-list");
  if (list) {
    list.append(logLine(entry));
    while (list.children.length > 40) list.removeChild(list.firstChild);
    scrollTimeline();
  }
}
function scrollTimeline() {
  const body = document.querySelector(".timeline-panel .panel__body");
  if (body) body.scrollTop = body.scrollHeight;
}

/* ============================================================================
   COMPONENT: SessionControls (+ scenario selector + Demo Mode toggle)
   ============================================================================ */
function renderControls() {
  const s = state.status;
  const scenarioGrid = el("div", { class: "scenario-grid" });
  state.scenarios.forEach((sc) => {
    scenarioGrid.append(
      el("button", {
        class: `scenario-btn ${sc.id === s.scenario ? "is-active" : ""}`,
        "data-id": sc.id,
        html: svg(sc.icon || "alert", 16) + `<span>${sc.label}</span>`,
        onclick: () => API.scenario(sc.id),
      })
    );
  });

  const controls = el("div", { class: "controls-grid" }, [
    el("button", { class: "btn", html: svg(s.paused ? "play" : "pause", 16) + `<span>${s.paused ? "Resume" : "Pause"} Monitoring</span>`, onclick: () => API.pause(!s.paused), disabled: !s.running }),
    el("button", { class: "btn", html: svg("snapshot", 16) + "<span>Capture Snapshot</span>", onclick: API.snapshot, disabled: !s.running }),
    el("button", { class: "btn", html: svg("replay", 16) + "<span>Replay Instruction</span>", onclick: () => API.replay() }),
    el("button", { class: "btn", html: svg("trash", 16) + "<span>Clear Session</span>", onclick: API.clear }),
  ]);

  const demoToggle = el("div", { class: "toggle-row" }, [
    el("div", { class: "toggle-row__text" }, [
      el("strong", { text: "Demo Mode" }),
      el("span", { text: s.demo_mode ? "Simulating live frames" : (s.vision_backend_available ? "Using live camera + models" : "Live backend not configured") }),
    ]),
    switchEl(s.demo_mode, (checked) => API.demoMode(checked)),
  ]);

  $("#session-controls").replaceChildren(
    el("div", { class: "panel__head" }, [el("h3", { text: "Session Controls" }), el("span", { class: "panel__sub", text: "operations" })]),
    el("div", { class: "panel__body" }, [
      el("div", { class: "control-group" }, [el("div", { class: "control-group__label", text: "Demo Scenario" }), scenarioGrid]),
      el("div", { class: "control-group" }, [el("div", { class: "control-group__label", text: "Controls" }), controls]),
      demoToggle,
    ])
  );
}
function switchEl(checked, onChange) {
  const input = el("input", { type: "checkbox" });
  input.checked = checked;
  input.addEventListener("change", () => onChange(input.checked));
  return el("label", { class: "switch" }, [input, el("span", { class: "switch__slider" })]);
}

/* ============================================================================
   COMPONENT: SafetyNotice + top chips + critical banner
   ============================================================================ */
function renderSafety() {
  $("#safety-notice").replaceChildren(
    el("div", { class: "safety-notice" }, [
      el("div", { html: svg("alert", 20) }),
      el("div", {
        html: "<strong>AI guidance may be imperfect.</strong> In a real emergency, follow official instructions and contact emergency services when it is safe to do so.",
      }),
    ])
  );
}
function renderChips() {
  const running = state.status.running;
  const ps = state.event?.pipeline_status || {};
  const wrap = $("#status-chips");
  wrap.replaceChildren(
    ...statusChips().map((c) => {
      const active = running && (ps[c.key] ? ps[c.key] !== "idle" : true);
      return el("div", { class: `chip ${active ? "chip--on chip--active" : ""}` }, [
        el("span", { class: "chip__dot" }),
        `${c.label}: ${active ? c.on : "Idle"}`,
      ]);
    })
  );
}
function renderBanner() {
  const banner = $("#critical-banner");
  if (state.event?.urgency === "Critical" && state.status.running) {
    banner.hidden = false;
    $("#critical-banner-text").textContent = state.event.situation || "Immediate danger detected. Act now.";
  } else {
    banner.hidden = true;
  }
}

/* ============================================================================
   GLOBAL RENDER + URGENCY THEME
   ============================================================================ */
function applyUrgency() {
  const urgency = (state.status.running && state.event?.urgency) || "Low";
  document.documentElement.setAttribute("data-urgency", urgency);
}
function renderTopMeta() {
  $("#latency-value").textContent = state.event?.latency_ms ? `${state.event.latency_ms}ms` : "—";
  $("#interval-value").textContent = `${state.status.frame_interval_s || 2.5}s`;
  const btn = $("#session-btn");
  btn.textContent = state.status.running ? "Stop Session" : "Start Session";
  btn.className = state.status.running ? "btn btn--danger" : "btn btn--primary";
}
function applyConfig() {
  const subtitle = $("#pipeline-subtitle");
  if (subtitle && state.config?.pipeline?.subtitle) {
    subtitle.textContent = state.config.pipeline.subtitle;
  }
}
function renderAll() {
  applyConfig();
  applyUrgency();
  renderTopMeta();
  renderChips();
  renderBanner();
  renderVideoPanel();
  renderGuidancePanel();
  renderSceneAnalysis();
  renderReasoning();
  renderControls();
  renderTimeline();
  renderSafety();
}

/* ============================================================================
   VOICE — browser speech synthesis + waveform sync
   ============================================================================ */
function speak(text) {
  if (!text || state.status.muted || !("speechSynthesis" in window)) {
    setWaveform(false);
    return;
  }
  try {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.02;
    u.pitch = 1.0;
    u.onstart = () => setWaveform(true);
    u.onend = () => setWaveform(false);
    u.onerror = () => setWaveform(false);
    window.speechSynthesis.speak(u);
  } catch {
    setWaveform(false);
  }
}
function setWaveform(on) {
  speaking = on;
  const wf = $("#waveform");
  const status = $("#wave-status");
  if (wf) wf.classList.toggle("is-playing", on);
  if (status) status.textContent = on ? "Voice streaming…" : (state.status.muted ? "Voice muted" : "Voice idle");
}

/* ============================================================================
   CAMERA — live feed + frame capture loop
   ============================================================================ */
async function startCamera() {
  if (mediaStream) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
    attachStream();
  } catch {
    mediaStream = null;
    drawFallback();
  }
}
function attachStream() {
  const video = $("#cam");
  const fallback = $("#cam-fallback");
  if (video && mediaStream) {
    video.srcObject = mediaStream;
    video.style.display = "block";
    if (fallback) fallback.style.display = "none";
    video.play?.().catch(() => {});
  }
}
function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
}
function drawFallback() {
  const video = $("#cam");
  const canvas = $("#cam-fallback");
  if (video) video.style.display = "none";
  if (!canvas) return;
  canvas.style.display = "block";
  const ctx = canvas.getContext("2d");
  canvas.width = 640; canvas.height = 360;
  ctx.fillStyle = "#04070d";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(56,189,248,0.12)";
  for (let y = 0; y < canvas.height; y += 18) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }
  ctx.fillStyle = "rgba(159,176,195,0.55)";
  ctx.font = "16px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(
    state.status.demo_mode ? "Camera feed unavailable — Demo Mode active" : "Camera feed unavailable",
    canvas.width / 2,
    canvas.height / 2
  );
}
function startFrameLoop() {
  stopFrameLoop();
  const intervalMs = frameIntervalMs();
  frameTimer = setInterval(async () => {
    if (!state.status.running || state.status.paused || state.status.demo_mode) return;
    const frame = captureFrame();
    if (frame) await API.frame(frame);
  }, intervalMs);
}
function stopFrameLoop() {
  if (frameTimer) { clearInterval(frameTimer); frameTimer = null; }
}
function captureFrame() {
  const video = $("#cam");
  if (!video || !mediaStream || video.videoWidth === 0) return null;
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = Math.round((video.videoHeight / video.videoWidth) * 640) || 360;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.7);
}

/* ============================================================================
   ACTIONS
   ============================================================================ */
async function toggleSession() {
  if (state.status.running) {
    await API.stop();
    stopCamera();
    stopFrameLoop();
    window.speechSynthesis?.cancel();
    setWaveform(false);
  } else {
    await startCamera();
    await API.start();
    startFrameLoop();
  }
}
async function toggleMute() {
  const next = !state.status.muted;
  await API.mute(next);
  if (next) { window.speechSynthesis?.cancel(); setWaveform(false); }
}
function callEmergency() {
  appendLog({ time: nowClock(), level: "alert", message: "Call Emergency Services pressed (UI placeholder — not wired to a real call)" });
  alert("UI placeholder: In a real deployment this would dial local emergency services (e.g. 911).\n\nDisOps does not place real calls in this demo build.");
}

/* ============================================================================
   WEBSOCKET
   ============================================================================ */
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => { state.wsConnected = true; };
  ws.onclose = () => { state.wsConnected = false; setTimeout(connectWS, 1500); };
  ws.onerror = () => ws.close();
  ws.onmessage = (msg) => {
    let payload;
    try { payload = JSON.parse(msg.data); } catch { return; }
    handleMessage(payload);
  };
}
function handleMessage({ type, data }) {
  switch (type) {
    case "snapshot":
    case "cleared":
      state.status = data.status || state.status;
      state.config = data.config || state.config;
      state.scenarios = data.scenarios || state.scenarios;
      state.event = data.last_event || (type === "cleared" ? null : state.event);
      state.log = data.log || [];
      renderAll();
      break;
    case "status":
      state.status = data;
      if (state.status.running && !state.status.demo_mode) startFrameLoop();
      else stopFrameLoop();
      renderTopMeta(); renderChips(); renderControls(); renderGuidancePanel();
      if (!data.running) { renderBanner(); applyUrgency(); renderVideoPanel(); }
      break;
    case "event":
      state.event = data;
      applyUrgency();
      renderTopMeta(); renderChips(); renderBanner();
      renderVideoPanel(); renderGuidancePanel(); renderSceneAnalysis(); renderReasoning();
      // Server plays GMI TTS on the host machine — sync waveform only.
      if (data.speak && !state.status.muted) {
        setWaveform(true);
        const ms = Math.max(1800, (data.voice_guidance || "").length * 70);
        setTimeout(() => setWaveform(false), ms);
      }
      break;
    case "replay":
      state.event = data;
      renderGuidancePanel(); renderVideoPanel(); renderSceneAnalysis(); renderReasoning();
      break;
    case "log":
      appendLog(data);
      break;
    case "pipeline_progress":
      updatePipelineProgress(data);
      break;
    case "pipeline_reset":
      renderPipeline(data, {});
      break;
  }
}

/* ----------------------------------------------------------------- utilities */
function formatClock(iso) {
  if (!iso) return "--:--:--";
  const d = new Date(iso);
  return isNaN(d) ? "--:--:--" : d.toLocaleTimeString([], { hour12: false });
}
function nowClock() { return new Date().toLocaleTimeString([], { hour12: false }); }

/* ============================================================================
   INIT — bootstrap from API, then live WebSocket
   ============================================================================ */
async function loadBootstrap() {
  try {
    const res = await fetch("/api/state");
    if (!res.ok) return;
    const data = await res.json();
    state.status = data.status || state.status;
    state.config = data.config || state.config;
    state.scenarios = data.scenarios || [];
    state.event = data.last_event || null;
    state.log = data.log || [];
  } catch {
    /* offline — WebSocket snapshot will hydrate when server is up */
  }
}
async function init() {
  await loadBootstrap();
  $("#session-btn").addEventListener("click", toggleSession);
  renderPipeline(state.event?.pipeline_status || {}, state.event?.stage_durations || {});
  renderAll();
  connectWS();
}
document.addEventListener("DOMContentLoaded", init);
