"use strict";

const chatEl = document.getElementById("chat");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("input");
const sendEl = document.getElementById("send");
const statsEl = document.getElementById("stats");
const statusEl = document.getElementById("status");
const providerEl = document.getElementById("provider");
const modelEl = document.getElementById("model");
const modelListEl = document.getElementById("model-list");
const keyBtn = document.getElementById("keybtn");

const history = [];
let providers = {};        // имя -> запись из /api/providers
let allModels = {};        // имя -> {models:[{id,free}], source, error, configured, label}
let defaultProvider = "ollama";
let defaultModel = "";

const LS_PROVIDER = "catlu.provider";
const MAX_OWN_PAID = 60;   // сколько платных моделей своего провайдера показывать
const MAX_OTHER = 40;      // сколько бесплатных чужих моделей показывать на провайдера

const savedModel = (provider) => localStorage.getItem(`catlu.model.${provider}`) || "";
const saveModel = (provider, model) => localStorage.setItem(`catlu.model.${provider}`, model || "");

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

const currentProvider = () => providerEl.value;
const currentModel = () => modelEl.value.trim();
const setStatus = (text) => { statusEl.textContent = text || ""; };

/** Разбирает выбор вида «[groq] llama-3.1-8b-instant». */
function choice(value) {
  const match = value.match(/^\[([^\]]+)\]\s+(.+)$/);
  return match ? { provider: match[1], model: match[2] } : null;
}

function updateStatus() {
  const info = providers[currentProvider()];
  if (!info) { setStatus(""); return; }
  const models = allModels[currentProvider()]?.models || [];
  const free = models.filter((m) => m.free).length;
  if (info.needs_key && !info.configured) {
    setStatus(`нужен ключ для «${info.label}» — нажмите 🔑, чтобы увидеть модели`);
  } else if (!currentModel()) {
    setStatus("выберите модель из списка");
  } else if (free) {
    setStatus(`бесплатных моделей: ${free}`);
  } else {
    setStatus("");
  }
}

/** Наполняет список моделей для провайдера + бесплатные модели других. */
function renderOptions(provider) {
  modelListEl.innerHTML = "";

  const own = allModels[provider]?.models || [];
  const free = own.filter((m) => m.free);
  const paid = own.filter((m) => !m.free).slice(0, MAX_OWN_PAID);
  free.concat(paid).forEach((m) => {
    const opt = el("option", null, m.free ? `${m.id} — бесплатно` : m.id);
    opt.value = m.id;
    opt.label = opt.textContent;
    modelListEl.appendChild(opt);
  });

  Object.keys(allModels).forEach((name) => {
    if (name === provider) return;
    const info = allModels[name];
    (info.models || []).filter((m) => m.free).slice(0, MAX_OTHER).forEach((m) => {
      const value = `[${name}] ${m.id}`;
      const opt = el("option", null, `${m.id} — бесплатно (${info.label || name})`);
      opt.value = value;
      opt.label = opt.textContent;
      modelListEl.appendChild(opt);
    });
  });
}

function selectModel(provider) {
  const info = allModels[provider] || {};
  const models = info.models || [];
  const ids = models.map((m) => m.id);
  const candidates = [
    savedModel(provider),
    provider === defaultProvider ? defaultModel : "",
    info.preferred || "",
  ];
  for (const candidate of candidates) {
    if (candidate && ids.includes(candidate)) {
      modelEl.value = candidate;
      saveModel(provider, candidate);
      return;
    }
  }
  const first = models.find((m) => m.free) || models[0];
  modelEl.value = first ? first.id : (candidates[0] || "");
  saveModel(provider, currentModel());
}

function fillModels(provider) {
  renderOptions(provider);
  selectModel(provider);
}

async function loadProviders() {
  let data;
  try {
    data = await (await fetch("/api/providers")).json();
  } catch {
    return;
  }
  providers = {};
  data.providers.forEach((p) => { providers[p.name] = p; });
  defaultProvider = data.default?.provider || "ollama";
  defaultModel = data.default?.model || "";

  providerEl.innerHTML = "";
  data.providers.forEach((p) => {
    const opt = el("option", null, p.configured || !p.needs_key ? p.label : `${p.label} — нужен ключ`);
    opt.value = p.name;
    providerEl.appendChild(opt);
  });

  // Сразу показываем запасной список из реестра, затем уточняем его
  // актуальными данными провайдеров (это может занять пару секунд).
  allModels = {};
  data.providers.forEach((p) => {
    allModels[p.name] = {
      provider: p.name,
      label: p.label,
      preferred: "",
      source: "fallback",
      configured: p.configured,
      models: (p.models || []).map((id) => ({ id, free: id.endsWith(":free") })),
    };
  });

  const saved = localStorage.getItem(LS_PROVIDER);
  providerEl.value = providers[saved] ? saved : defaultProvider;
  fillModels(providerEl.value);
  updateStatus();
  await refreshModels();
}

/** Запрашивает списки моделей у всех провайдеров и перерисовывает поле. */
async function refreshModels() {
  try {
    const data = await (await fetch("/api/models?all=1")).json();
    allModels = {};
    (data.providers || []).forEach((p) => { allModels[p.provider] = p; });
  } catch {
    allModels = {};
  }
  fillModels(currentProvider());
  updateStatus();
}

providerEl.addEventListener("change", () => {
  localStorage.setItem(LS_PROVIDER, currentProvider());
  fillModels(currentProvider());
  updateStatus();
});

modelEl.addEventListener("change", applyChoice);

/** Если выбран пункт вида «[openrouter] ... :free», переключаем и провайдера. */
function applyChoice() {
  const pick = choice(modelEl.value);
  if (pick && providers[pick.provider]) {
    providerEl.value = pick.provider;
    localStorage.setItem(LS_PROVIDER, pick.provider);
    modelEl.value = pick.model;
    renderOptions(pick.provider);
  }
  saveModel(currentProvider(), currentModel());
  updateStatus();
}

keyBtn.addEventListener("click", async () => {
  const name = currentProvider();
  const label = providers[name]?.label || name;
  const key = window.prompt(`API-ключ для «${label}»:\n(хранится только на этом компьютере)`);
  if (key == null) return;
  try {
    const r = await fetch("/api/keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider: name, key }),
    });
    if (!r.ok) throw new Error(await r.text());
    setStatus(key.trim() ? "ключ сохранён, обновляю список…" : "ключ удалён");
    await loadProviders();
  } catch {
    setStatus("не удалось сохранить ключ");
  }
});

function addMessage(role, text) {
  const wrap = el("div", "msg " + role);
  const bubble = el("div", "bubble", text);
  wrap.appendChild(bubble);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return { wrap, bubble };
}

function addSources(wrap, sources) {
  if (!sources || !sources.length) return;
  const box = el("div", "sources");
  box.appendChild(el("div", "sources-title", "Источники:"));
  for (const s of sources) {
    const page = s.page != null ? `, ${s.unit || "стр."} ${s.page}` : "";
    box.appendChild(el("div", "source", `${s.source}${page}`));
  }
  wrap.appendChild(box);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function parseEvent(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  const raw = dataLines.join("\n");
  let data;
  try { data = JSON.parse(raw); } catch { data = { text: raw }; }
  return { event, data };
}

async function streamChat(message, prior) {
  const { wrap, bubble } = addMessage("assistant", "");
  bubble.classList.add("typing");

  let text = "";
  let sources = null;

  let response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history: prior,
        provider: currentProvider(),
        model: currentModel(),
      }),
    });
  } catch {
    bubble.classList.remove("typing");
    bubble.textContent = "⚠️ Нет связи с сервером.";
    return;
  }

  if (!response.ok || !response.body) {
    bubble.classList.remove("typing");
    bubble.textContent = "⚠️ Ошибка сервера: " + response.status;
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const ev = parseEvent(block);
      if (!ev) continue;

      if (ev.event === "token") {
        text += ev.data.text || "";
        bubble.textContent = text;
        chatEl.scrollTop = chatEl.scrollHeight;
      } else if (ev.event === "sources") {
        sources = ev.data;
      } else if (ev.event === "model") {
        setStatus(`${ev.data.provider} · ${ev.data.model}`);
      } else if (ev.event === "error") {
        text += (text ? "\n\n" : "") + "⚠️ " + (ev.data.message || "Ошибка");
        bubble.textContent = text;
      }
    }
  }

  bubble.classList.remove("typing");
  if (sources) addSources(wrap, sources);
  history.push({ role: "assistant", content: text });
}

async function submit(text) {
  addMessage("user", text);
  const prior = history.slice();
  history.push({ role: "user", content: text });
  await streamChat(text, prior);
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  applyChoice();
  inputEl.value = "";
  inputEl.style.height = "auto";
  sendEl.disabled = true;
  try {
    await submit(text);
  } finally {
    sendEl.disabled = false;
    inputEl.focus();
  }
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
});

fetch("/api/stats")
  .then((r) => r.json())
  .then((s) => { statsEl.textContent = `${s.total_chunks} фрагментов · ${s.sources_count} файлов`; })
  .catch(() => { statsEl.textContent = "нет данных"; });

loadProviders();

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}
