// Standalone chat UI for the Custom Inference Engine API (server/api_server.py).
// No build step, no dependencies: host this folder on any static host.

const $ = (id) => document.getElementById(id);
const STORE_CFG = "cie.settings.v1";
const STORE_CHAT = "cie.chat.v1";

const state = {
  cfg: { url: "", key: "", maxTokens: 512, system: $("cfg-system").value },
  messages: [],          // [{role, content, metrics?}]
  controller: null,      // AbortController of the in-flight request
  statsTimer: null,
};

// ---------------------------------------------------------------- persistence
function loadState() {
  try { Object.assign(state.cfg, JSON.parse(localStorage.getItem(STORE_CFG) || "{}")); } catch {}
  try { state.messages = JSON.parse(localStorage.getItem(STORE_CHAT) || "[]"); } catch {}
  // Allow `?api=https://...&key=...` deep links (handy when the tunnel URL changes).
  const qs = new URLSearchParams(location.search);
  if (qs.get("api")) state.cfg.url = qs.get("api");
  if (qs.get("key")) state.cfg.key = qs.get("key");
  if (qs.has("api") || qs.has("key")) {
    saveCfg();
    history.replaceState(null, "", location.pathname); // don't leave the key in the address bar
  }
}
const saveCfg = () => localStorage.setItem(STORE_CFG, JSON.stringify(state.cfg));
const saveChat = () => localStorage.setItem(STORE_CHAT, JSON.stringify(state.messages));

// ---------------------------------------------------------------- HTTP
function apiUrl(path) {
  return state.cfg.url.replace(/\/+$/, "") + path;
}
function headers(json = false) {
  const h = { "ngrok-skip-browser-warning": "1" };
  if (state.cfg.key) h.Authorization = `Bearer ${state.cfg.key}`;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
async function getJSON(path) {
  const r = await fetch(apiUrl(path), { headers: headers() });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || body.error || `HTTP ${r.status}`);
  return body;
}

// ---------------------------------------------------------------- status + stats
function setStatus(kind, text) {
  $("status-dot").dataset.state = kind;
  $("status-text").textContent = text;
}

async function refreshHealth() {
  if (!state.cfg.url) { setStatus("idle", "Not connected · open Settings"); return false; }
  try {
    const r = await fetch(apiUrl("/health"), { headers: headers() });
    const h = await r.json();
    if (h.status === "ready") setStatus("ready", `${h.model} · ${h.gpu}`);
    else if (h.status === "error") setStatus("error", `Engine error: ${h.error}`);
    else setStatus("loading", `Engine ${h.status}…`);
    return h.status === "ready";
  } catch {
    setStatus("error", "Unreachable · is the Colab cell still running?");
    return false;
  }
}

const pct = (x) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);
function fmtUptime(s) {
  if (s == null) return "—";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m ${Math.floor(s % 60)}s`;
}

async function refreshStats() {
  if (!(await refreshHealth())) return;
  try {
    const s = await getJSON("/v1/stats");
    $("s-model").textContent = `${s.model} (${s.dtype})`;
    $("s-gpu").textContent = s.gpu ?? "—";
    $("s-active").textContent = `${s.active_requests ?? 0} (run ${s.running ?? 0} / wait ${s.waiting ?? 0})`;
    $("s-kv").textContent = pct(s.kv_block_usage);
    $("s-prefix").textContent = pct(s.prefix_cache_hit_rate);
    $("s-graph").textContent = `${s.cuda_graph_replays ?? 0} (eager ${s.eager_decode_steps ?? 0})`;
    $("s-tokens").textContent = (s.total_generated_tokens ?? 0).toLocaleString();
    $("s-uptime").textContent = fmtUptime(s.uptime_s);
  } catch (e) {
    setStatus("error", e.message);
  }
}

function startStatsPolling() {
  clearInterval(state.statsTimer);
  refreshStats();
  state.statsTimer = setInterval(refreshStats, 5000);
}

// ---------------------------------------------------------------- rendering
// Content is model output: build DOM with textContent only (never innerHTML).
function renderMarkdownLite(container, text) {
  container.replaceChildren();
  const fence = /```([\w+-]*)[^\n]*\n?([\s\S]*?)(?:```|$)/g;
  let last = 0, m;
  while ((m = fence.exec(text)) !== null) {
    appendProse(container, text.slice(last, m.index));
    container.append(codeBlock(m[1], m[2].replace(/\n$/, "")));
    last = fence.lastIndex;
    if (m[0].length === 0) break;
  }
  appendProse(container, text.slice(last));
}

function appendProse(container, text) {
  for (const para of text.split(/\n{2,}/)) {
    if (!para.trim()) continue;
    const p = document.createElement("p");
    para.split(/(`[^`\n]+`)/).forEach((part) => {
      if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
        const c = document.createElement("code");
        c.className = "inline";
        c.textContent = part.slice(1, -1);
        p.append(c);
      } else {
        p.append(part);
      }
    });
    container.append(p);
  }
}

function codeBlock(lang, code) {
  const wrap = document.createElement("div");
  wrap.className = "code-block";
  const head = document.createElement("div");
  head.className = "code-head";
  const label = document.createElement("span");
  label.textContent = lang || "code";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "ghost";
  copy.textContent = "Copy";
  copy.addEventListener("click", async () => {
    await navigator.clipboard.writeText(code);
    copy.textContent = "Copied";
    setTimeout(() => (copy.textContent = "Copy"), 1200);
  });
  head.append(label, copy);
  const pre = document.createElement("pre");
  const c = document.createElement("code");
  c.textContent = code;
  pre.append(c);
  wrap.append(head, pre);
  return wrap;
}

function metricsLine(m) {
  const div = document.createElement("div");
  div.className = "metrics";
  const items = [
    `TTFT ${m.ttft_ms} ms`,
    `${m.tokens_per_s} tok/s`,
    `TPOT ${m.tpot_ms} ms`,
    `${m.completion_tokens} tokens`,
    `prompt ${m.prompt_tokens} (${m.cached_tokens} cached)`,
    m.finish_reason === "length" ? "hit max tokens" : null,
  ].filter(Boolean);
  for (const t of items) {
    const s = document.createElement("span");
    s.textContent = t;
    div.append(s);
  }
  return div;
}

function addMessageEl(msg) {
  $("empty-state")?.remove();
  const el = document.createElement("article");
  el.className = `msg ${msg.role}${msg.error ? " error" : ""}`;
  el.setAttribute("aria-label", msg.role === "user" ? "You" : "Assistant");
  const body = document.createElement("div");
  body.className = "body";
  el.append(body);
  if (msg.role === "user" || msg.error) body.textContent = msg.content;
  else renderMarkdownLite(body, msg.content);
  if (msg.metrics) el.append(metricsLine(msg.metrics));
  $("chat").append(el);
  return el;
}

function renderAll() {
  const chat = $("chat");
  chat.querySelectorAll(".msg").forEach((n) => n.remove());
  state.messages.forEach(addMessageEl);
  chat.scrollTop = chat.scrollHeight;
}

const nearBottom = (el) => el.scrollHeight - el.scrollTop - el.clientHeight < 80;

// ---------------------------------------------------------------- chat
async function send(text) {
  if (!state.cfg.url) { openSettings(); return; }

  const userMsg = { role: "user", content: text };
  state.messages.push(userMsg);
  addMessageEl(userMsg);
  saveChat();

  const reply = { role: "assistant", content: "" };
  const el = addMessageEl(reply);
  el.classList.add("streaming");
  const body = el.querySelector(".body");
  const chat = $("chat");
  chat.scrollTop = chat.scrollHeight;

  const history = state.messages
    .filter((m) => !m.error)
    .map(({ role, content }) => ({ role, content }));
  const payload = {
    messages: [{ role: "system", content: state.cfg.system }, ...history],
    max_tokens: Number(state.cfg.maxTokens) || 512,
    stream: true,
  };

  setBusy(true);
  state.controller = new AbortController();
  let pending = false;
  const paint = () => {
    pending = false;
    const stick = nearBottom(chat);
    renderMarkdownLite(body, reply.content);
    if (stick) chat.scrollTop = chat.scrollHeight;
  };

  try {
    const r = await fetch(apiUrl("/v1/chat/completions"), {
      method: "POST",
      headers: headers(true),
      body: JSON.stringify(payload),
      signal: state.controller.signal,
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }

    const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        const chunk = JSON.parse(data);
        if (chunk.error) throw new Error(chunk.error.message);
        const delta = chunk.choices?.[0]?.delta?.content;
        if (delta) {
          reply.content += delta;
          if (!pending) { pending = true; requestAnimationFrame(paint); }
        }
        if (chunk.metrics) reply.metrics = chunk.metrics;
      }
    }
  } catch (e) {
    if (e.name === "AbortError") {
      reply.content += reply.content ? "\n\n[stopped]" : "[stopped]";
    } else {
      reply.error = true;
      reply.content = `Error: ${e.message}`;
      el.classList.add("error");
    }
  } finally {
    el.classList.remove("streaming");
    if (reply.error) body.textContent = reply.content;
    else paint();
    if (reply.metrics) el.append(metricsLine(reply.metrics));
    state.messages.push(reply);
    saveChat();
    setBusy(false);
    state.controller = null;
    refreshStats();
  }
}

function setBusy(busy) {
  $("send").disabled = busy;
  $("send").hidden = busy;
  $("stop").hidden = !busy;
}

// ---------------------------------------------------------------- settings dialog
const dlg = $("settings");

function openSettings() {
  $("cfg-url").value = state.cfg.url;
  $("cfg-key").value = state.cfg.key;
  $("cfg-max").value = state.cfg.maxTokens;
  $("cfg-system").value = state.cfg.system;
  $("cfg-result").textContent = "";
  $("cfg-result").className = "hint";
  dlg.showModal();
}

dlg.addEventListener("close", () => {
  if (dlg.returnValue === "save") {
    state.cfg.url = $("cfg-url").value.trim();
    state.cfg.key = $("cfg-key").value.trim();
    state.cfg.maxTokens = Number($("cfg-max").value) || 512;
    state.cfg.system = $("cfg-system").value;
    saveCfg();
    startStatsPolling();
  }
  dlg.returnValue = "";
});

$("cfg-test").addEventListener("click", async () => {
  const out = $("cfg-result");
  const prev = { ...state.cfg };
  state.cfg.url = $("cfg-url").value.trim();
  state.cfg.key = $("cfg-key").value.trim();
  out.className = "hint";
  out.textContent = "Testing…";
  try {
    const h = await (await fetch(apiUrl("/health"), { headers: headers() })).json();
    await getJSON("/v1/models"); // validates the API key
    out.className = h.status === "ready" ? "hint ok" : "hint";
    out.textContent = h.status === "ready"
      ? `Connected: ${h.model} on ${h.gpu}`
      : `Reachable, engine is ${h.status}…`;
  } catch (e) {
    out.className = "hint bad";
    out.textContent = `Failed: ${e.message}`;
  } finally {
    Object.assign(state.cfg, { url: prev.url, key: prev.key });
  }
});

// Light-dismiss fallback where <dialog closedby> is unsupported (Safari).
if (!("closedBy" in HTMLDialogElement.prototype)) {
  const outside = (e) => {
    const r = dlg.getBoundingClientRect();
    return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
  };
  let pressedOutside = false;
  dlg.addEventListener("pointerdown", (e) => { pressedOutside = outside(e); });
  dlg.addEventListener("pointerup", (e) => { if (pressedOutside && outside(e)) dlg.close(); });
}

// ---------------------------------------------------------------- composer
const prompt = $("prompt");

prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    // IME-safe: don't submit while composing (keyCode 229 covers Safari's event ordering).
    if (event.isComposing || event.keyCode === 229) return;
    $("composer").requestSubmit();
  }
});

// Auto-grow fallback where CSS `field-sizing: content` is unsupported.
if (!CSS.supports("field-sizing", "content")) {
  prompt.addEventListener("input", () => {
    prompt.style.height = "auto";
    prompt.style.height = `${prompt.scrollHeight}px`;
  });
}

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = prompt.value.trim();
  if (!text || state.controller) return;
  prompt.value = "";
  prompt.style.height = "";
  send(text);
});

$("stop").addEventListener("click", () => state.controller?.abort());
$("open-settings").addEventListener("click", openSettings);
$("new-chat").addEventListener("click", () => {
  state.controller?.abort();
  state.messages = [];
  saveChat();
  location.reload();
});

// ---------------------------------------------------------------- boot
loadState();
renderAll();
if (state.cfg.url) startStatsPolling();
else { setStatus("idle", "Not connected · open Settings"); openSettings(); }

