"use strict";

/* ---- Элементы --------------------------------------------------------- */
const chatEl = document.getElementById("chat");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("input");
const sendEl = document.getElementById("send");
const statsEl = document.getElementById("stats");
const statusEl = document.getElementById("status");
const agentEl = document.getElementById("agent");
const modelLabelEl = document.getElementById("model-label");
const gearEl = document.getElementById("gear");

const panelEl = document.getElementById("panel");
const panelCloseEl = document.getElementById("panel-close");
const panelNoteEl = document.getElementById("panel-note");
const tabsEl = document.getElementById("tabs");
const agentTitleEl = document.getElementById("agent-title");
const agentCreateEl = document.getElementById("agent-create");
const agentListEl = document.getElementById("agent-list");
const bookFilesEl = document.getElementById("book-files");
const indexBtnEl = document.getElementById("index-btn");
const jobEl = document.getElementById("job");
const bookListEl = document.getElementById("book-list");
const providerEl = document.getElementById("provider");
const modelEl = document.getElementById("model-input");
const modelListEl = document.getElementById("model-list");
const keyBtn = document.getElementById("keybtn");
const modelStatusEl = document.getElementById("model-status");
const baseStatsEl = document.getElementById("base-stats");
const baseClearEl = document.getElementById("base-clear");
const agentRenameEl = document.getElementById("agent-rename");
const agentDeleteEl = document.getElementById("agent-delete");

const JSON_HEADER = { "Content-Type": "application/json" };

/* ---- Состояние -------------------------------------------------------- */
const history = [];              // переписка с текущим агентом
let agents = [];                 // список агентов из /api/agents
let currentSlug = localStorage.getItem("catlu.agent") || "";
let selectedTab = "agents";
let providers = {};              // имя -> запись из /api/providers
let allModels = {};              // имя -> {models:[{id,free}], source, error, configured, label}
let defaultProvider = "ollama";
let defaultModel = "";
let jobTimer = null;             // опрос хода индексации
let jobRunning = false;
let indexRunning = false;        // у сервера уже идёт индексация
let bookCount = 0;               // сколько книг у активного агента

const MAX_OWN_PAID = 60;         // сколько платных моделей своего провайдера показывать
const MAX_OTHER = 40;            // сколько бесплатных чужих моделей показывать на провайдера

/* ---- Мелкие помощники ------------------------------------------------- */
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

/** Запрос к серверу: ошибка сервера превращается в понятный текст. */
async function api(path, options) {
  const response = await fetch(path, options);
  let data = null;
  try { data = await response.json(); } catch { data = null; }
  if (!response.ok) {
    throw new Error((data && (data.detail || data.error)) || `Ошибка ${response.status}`);
  }
  return data || {};
}

function humanSize(size) {
  let value = Number(size) || 0;
  const units = ["Б", "КБ", "МБ", "ГБ"];
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return unit === 0 ? `${Math.round(value)} ${units[unit]}`
                    : `${value.toFixed(1)} ${units[unit]}`;
}

const agentBySlug = (slug) => agents.find((agent) => agent.slug === slug) || null;
const currentProvider = () => providerEl.value;
const currentModel = () => modelEl.value.trim();
const setStatus = (text) => { statusEl.textContent = text || ""; };
const setModelStatus = (text) => { modelStatusEl.textContent = text || ""; };

let noteTimer = null;
function showNote(text, kind) {
  panelNoteEl.hidden = !text;
  panelNoteEl.textContent = text || "";
  panelNoteEl.classList.toggle("bad", kind === "error");
  clearTimeout(noteTimer);
  if (text) noteTimer = setTimeout(() => { panelNoteEl.hidden = true; }, 8000);
}

/* ---- Настройки: открыть, закрыть, вкладки ----------------------------- */
function openPanel(tab) {
  panelEl.hidden = false;
  selectTab(tab || selectedTab);
}

function closePanel() {
  panelEl.hidden = true;
}

function selectTab(name) {
  selectedTab = name;
  tabsEl.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  panelEl.querySelectorAll(".tab-page").forEach((page) => {
    page.hidden = page.dataset.page !== name;
  });
  if (name === "agents") refreshAgents();
  if (name === "books") refreshBooks();
  if (name === "base") refreshAgents();
}

gearEl.addEventListener("click", () => openPanel());
modelLabelEl.addEventListener("click", () => openPanel("models"));
panelCloseEl.addEventListener("click", closePanel);
panelEl.addEventListener("click", (event) => {
  if (event.target === panelEl) closePanel();      // клик по затемнению
});
tabsEl.addEventListener("click", (event) => {
  const tab = event.target.closest(".tab");
  if (tab) selectTab(tab.dataset.tab);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !panelEl.hidden) closePanel();
});

/* ---- Агенты ----------------------------------------------------------- */
async function refreshAgents() {
  let data;
  try {
    data = await api("/api/agents");
  } catch (error) {
    showNote(error.message, "error");
    return;
  }
  agents = data.agents || [];
  if (!agentBySlug(currentSlug)) {
    currentSlug = agents.length ? agents[0].slug : "";
  }
  if (currentSlug) localStorage.setItem("catlu.agent", currentSlug);
  renderAgentSelect();
  renderAgentList();
  renderBase();
  setComposerEnabled(Boolean(currentSlug));
  refreshStats();
}

function renderAgentSelect() {
  agentEl.innerHTML = "";
  if (!agents.length) {
    const option = el("option", null, "агентов нет");
    option.value = "";
    agentEl.appendChild(option);
    agentEl.disabled = true;
    return;
  }
  agentEl.disabled = false;
  agents.forEach((agent) => {
    const option = el("option", null, agent.title);
    option.value = agent.slug;
    agentEl.appendChild(option);
  });
  agentEl.value = currentSlug;
}

function renderAgentList() {
  agentListEl.innerHTML = "";
  if (!agents.length) {
    agentListEl.appendChild(el("div", "empty", "Пока ни одного агента. Создайте первого — например «Кулинария»."));
    return;
  }
  agents.forEach((agent) => {
    const row = el("div", "row");
    row.dataset.slug = agent.slug;

    const main = el("div", "row-main");
    const title = el("div", "row-title", agent.title);
    if (agent.slug === currentSlug) title.appendChild(el("span", "badge", "активен"));
    if (agent.error) title.appendChild(el("span", "badge warn", "база не открылась"));
    main.appendChild(title);
    const sub = `${agent.folder} · книг ${agent.books} · фрагментов ${agent.chunks} · ${agent.size_text}`;
    main.appendChild(el("div", "row-sub", sub));
    row.appendChild(main);

    const actions = el("div", "row-actions");
    if (agent.slug !== currentSlug) {
      const use = el("button", "icon", "✓");
      use.type = "button";
      use.dataset.act = "use";
      use.title = "Сделать активным";
      actions.appendChild(use);
    }
    const rename = el("button", "icon", "✎");
    rename.type = "button";
    rename.dataset.act = "rename";
    rename.title = "Переименовать";
    actions.appendChild(rename);

    const remove = el("button", "icon danger", "🗑");
    remove.type = "button";
    remove.dataset.act = "delete";
    remove.title = "Удалить агента вместе с книгами и базой";
    actions.appendChild(remove);

    row.appendChild(actions);
    agentListEl.appendChild(row);
  });
}

agentListEl.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-act]");
  if (!button) return;
  const agent = agentBySlug(button.closest(".row").dataset.slug);
  if (!agent) return;
  if (button.dataset.act === "use") switchAgent(agent.slug);
  if (button.dataset.act === "rename") renameAgent(agent);
  if (button.dataset.act === "delete") deleteAgent(agent);
});

agentEl.addEventListener("change", () => switchAgent(agentEl.value));

function switchAgent(slug) {
  if (slug === currentSlug || !agentBySlug(slug)) return;
  currentSlug = slug;
  localStorage.setItem("catlu.agent", slug);
  history.length = 0;                 // у каждого агента своя переписка
  chatEl.innerHTML = "";
  agentEl.value = slug;
  renderAgentList();
  renderBase();
  refreshStats();
  refreshBooks();
}

agentCreateEl.addEventListener("click", createAgent);
agentTitleEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); createAgent(); }
});

async function createAgent() {
  const title = agentTitleEl.value.trim();
  if (!title) { agentTitleEl.focus(); return; }
  agentCreateEl.disabled = true;
  try {
    const agent = await api("/api/agents", {
      method: "POST", headers: JSON_HEADER, body: JSON.stringify({ title }),
    });
    agentTitleEl.value = "";
    currentSlug = agent.slug;
    localStorage.setItem("catlu.agent", agent.slug);
    history.length = 0;
    chatEl.innerHTML = "";
    await refreshAgents();
    selectTab("books");             // дальше обычно загружают книги
    showNote(`Агент «${agent.title}» создан: папка ${agent.folder}. Загрузите книги и нажмите «Индексировать».`);
  } catch (error) {
    showNote(error.message, "error");
  } finally {
    agentCreateEl.disabled = false;
  }
}

async function renameAgent(agent) {
  const title = window.prompt("Новое название агента:", agent.title);
  if (title == null || !title.trim() || title.trim() === agent.title) return;
  try {
    await api(`/api/agents/${encodeURIComponent(agent.slug)}`, {
      method: "PATCH", headers: JSON_HEADER, body: JSON.stringify({ title }),
    });
    await refreshAgents();
    showNote("Название изменено");
  } catch (error) {
    showNote(error.message, "error");
  }
}

async function deleteAgent(agent) {
  const question = `Удалить агента «${agent.title}» вместе с книгами и базой?\n` +
                   "Папка будет стёрта полностью, отменить это нельзя.";
  if (!window.confirm(question)) return;
  try {
    await api(`/api/agents/${encodeURIComponent(agent.slug)}`, { method: "DELETE" });
    if (currentSlug === agent.slug) {
      currentSlug = "";
      localStorage.removeItem("catlu.agent");
      history.length = 0;
      chatEl.innerHTML = "";
    }
    await refreshAgents();
    refreshBooks();
    showNote(`Агент «${agent.title}» удалён`);
  } catch (error) {
    showNote(error.message, "error");
  }
}

/* ---- Книги: список, загрузка, удаление, индексация -------------------- */
async function refreshBooks() {
  if (!currentSlug) {
    bookListEl.innerHTML = "";
    bookListEl.appendChild(el("div", "empty", "Сначала выберите или создайте агента."));
    bookFilesEl.disabled = true;
    indexBtnEl.disabled = true;
    jobEl.hidden = true;
    return;
  }
  bookFilesEl.disabled = false;
  let data;
  try {
    data = await api(`/api/agents/${encodeURIComponent(currentSlug)}/books`);
  } catch (error) {
    showNote(error.message, "error");
    return;
  }
  const books = data.books || [];
  bookCount = books.length;
  bookListEl.innerHTML = "";
  if (!books.length) {
    bookListEl.appendChild(el("div", "empty", data.chunks
      ? "Файлов в папке нет, но в базе остались фрагменты проиндексированных книг — они продолжают работать, пока вы не очистите базу."
      : "В папке агента пока нет книг."));
  }
  books.forEach((book) => bookListEl.appendChild(bookRow(book)));
  indexBtnEl.disabled = !books.length || indexRunning;
  pollJob();
}

function bookRow(book) {
  const row = el("div", "row");
  row.dataset.name = book.name;

  const main = el("div", "row-main");
  const title = el("div", "row-title", book.name);
  if (!book.supported) title.appendChild(el("span", "badge warn", "формат не читается"));
  main.appendChild(title);
  const state = book.chunks ? `фрагментов ${book.chunks}` : "не проиндексирована";
  main.appendChild(el("div", "row-sub", `${book.size_text || humanSize(book.size)} · ${state}`));
  row.appendChild(main);

  const actions = el("div", "row-actions");
  const remove = el("button", "icon danger", "🗑");
  remove.type = "button";
  remove.dataset.act = "delete";
  remove.title = "Удалить книгу и её фрагменты";
  actions.appendChild(remove);
  row.appendChild(actions);
  return row;
}

bookListEl.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-act='delete']");
  if (!button || !currentSlug) return;
  const name = button.closest(".row").dataset.name;
  if (!window.confirm(`Удалить «${name}» из книг агента и из базы?`)) return;
  try {
    const data = await api(
      `/api/agents/${encodeURIComponent(currentSlug)}/books/${encodeURIComponent(name)}`,
      { method: "DELETE" });
    showNote(`«${name}» удалена: убрано фрагментов ${data.removed}`);
    refreshBooks();
    refreshAgents();
  } catch (error) {
    showNote(error.message, "error");
  }
});

bookFilesEl.addEventListener("change", async () => {
  const files = Array.from(bookFilesEl.files || []);
  bookFilesEl.value = "";
  if (!files.length || !currentSlug) return;
  await uploadBooks(files);
});

/** Загрузка книг: показываем проценты (для больших файлов это важно). */
function uploadBooks(files) {
  return new Promise((resolve) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/agents/${encodeURIComponent(currentSlug)}/books`);
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      renderProgress(`Загрузка книг: ${files.length} файл(ов)`, event.loaded / event.total);
    };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { data = {}; }
      if (xhr.status < 200 || xhr.status >= 300) {
        renderJob({ upload: true, state: "error", error: data.detail || `Ошибка ${xhr.status}` });
        resolve();
        return;
      }
      const uploaded = data.uploaded || [];
      const failed = data.failed || [];
      const names = uploaded.map((item) => item.name).join(", ");
      renderJob({
        upload: true,
        state: failed.length ? "error" : "done",
        error: failed.length
          ? failed.map((item) => `${item.name} — ${item.error}`).join("; ")
          : "",
        log: uploaded.length ? [`Загружено: ${names}. Теперь нажмите «Индексировать».`] : [],
      });
      refreshBooks();
      refreshAgents();
      resolve();
    };
    xhr.onerror = () => {
      renderJob({ upload: true, state: "error", error: "нет связи с сервером" });
      resolve();
    };
    renderProgress(`Загрузка книг: ${files.length} файл(ов)`, 0);
    xhr.send(form);
  });
}

indexBtnEl.addEventListener("click", async () => {
  if (!currentSlug) return;
  indexBtnEl.disabled = true;
  try {
    const data = await api(`/api/agents/${encodeURIComponent(currentSlug)}/index`,
                           { method: "POST" });
    renderJob(data.job);
    jobRunning = data.job && data.job.state === "running";
    indexRunning = jobRunning;
    showNote(jobRunning ? "Индексация пошла: можно закрыть настройки и пользоваться чатом."
                        : "Индексировать нечего: сначала загрузите книги.");
    if (jobRunning) pollJob();
  } catch (error) {
    showNote(error.message, "error");
  } finally {
    indexBtnEl.disabled = false;
  }
});

/** Ход индексации: сервер отдаёт состояние, страница за ним присматривает. */
async function pollJob() {
  if (!currentSlug) return;
  const slug = currentSlug;
  let data;
  try {
    data = await api(`/api/agents/${encodeURIComponent(slug)}/job`);
  } catch {
    return;
  }
  if (slug !== currentSlug) return;
  const job = data.job;
  const running = Boolean(job && job.state === "running");

  if (running) {
    indexRunning = true;
    indexBtnEl.disabled = true;
    renderJob(job);
    jobTimer = setTimeout(pollJob, 1000);
  } else if (jobRunning) {
    jobRunning = false;             // индексация только что закончилась
    indexRunning = false;
    indexBtnEl.disabled = !bookCount;
    if (job) renderJob(job);
    refreshBooks();
    refreshAgents();
  }
  jobRunning = running;
}

function showJob(node) {
  jobEl.innerHTML = "";
  jobEl.hidden = !node;
  if (node) jobEl.appendChild(node);
}

function renderProgress(label, percent) {
  const line = el("div", "job-line");
  line.appendChild(el("span", "job-label", label));
  const bar = el("div", "bar");
  const fill = el("div", "fill");
  fill.style.width = `${Math.round((percent || 0) * 100)}%`;
  bar.appendChild(fill);
  line.appendChild(bar);
  showJob(line);
}

/** Состояние индексации: что читается сейчас и сколько уже записано. */
function renderJob(job) {
  if (!job) { showJob(null); return; }
  const failed = job.failed || 0;
  const done = job.state === "done";
  const error = job.state === "error";

  const line = el("div", "job-line" + (error ? " error" : ""));
  let label;
  if (job.upload) {
    label = error ? `Не получилось: ${job.error || "ошибка загрузки"}`
                  : ((job.log && job.log[0]) || "Книги загружены");
  } else if (error) {
    label = `Не получилось: ${job.error || "неизвестная ошибка"}`;
  } else if (done) {
    label = `Готово: ${job.chunks} фрагментов` + (failed ? `, сбоев ${failed}` : "");
  } else if (job.file) {
    label = `Индексация ${job.index}/${job.total || "?"}: ${job.file}`;
  } else {
    label = "Индексация: готовлюсь…";
  }
  line.appendChild(el("span", "job-label", label));

  if (!job.upload && !done && !error) {
    const bar = el("div", "bar");
    const fill = el("div", "fill");
    const percent = job.total ? Math.max(0, (job.index - 1) / job.total) : 0;
    fill.style.width = `${Math.round(percent * 100)}%`;
    bar.appendChild(fill);
    line.appendChild(bar);
    if (job.chunks) line.appendChild(el("span", "job-extra", `записано фрагментов: ${job.chunks}`));
  } else if (error && job.log && job.log.length) {
    line.appendChild(el("span", "job-extra", job.log[job.log.length - 1]));
  }
  showJob(line);
}

/* ---- База ------------------------------------------------------------- */
function renderBase() {
  const agent = agentBySlug(currentSlug);
  baseStatsEl.innerHTML = "";
  const indexed = !agent ? "—"
    : agent.books ? `${agent.indexed} из ${agent.books}`
    : agent.indexed ? `${agent.indexed} книг (файлы уже не в папке)`
    : "ничего";
  const rows = agent ? [
    ["Агент", agent.title],
    ["Папка", agent.folder],
    ["Книг в папке", String(agent.books)],
    ["Проиндексировано", indexed],
    ["Фрагментов в базе", String(agent.chunks)],
    ["Занимает", agent.size_text],
    ["Создан", agent.created || "—"],
  ] : [["Агент", "не выбран"]];
  rows.forEach(([key, value]) => {
    const row = el("div", "stat");
    row.appendChild(el("span", "stat-key", key));
    row.appendChild(el("span", "stat-value", value));
    baseStatsEl.appendChild(row);
  });
  baseClearEl.disabled = !agent || !agent.chunks;
  agentRenameEl.disabled = !agent;
  agentDeleteEl.disabled = !agent;
}

baseClearEl.addEventListener("click", async () => {
  const agent = agentBySlug(currentSlug);
  if (!agent) return;
  if (!window.confirm(`Очистить базу агента «${agent.title}»?\nКниги останутся, фрагменты удалятся.`)) return;
  try {
    const data = await api(`/api/agents/${encodeURIComponent(agent.slug)}/base`,
                           { method: "DELETE" });
    showNote(`База очищена: убрано фрагментов ${data.removed}`);
    refreshAgents();
    refreshBooks();
  } catch (error) {
    showNote(error.message, "error");
  }
});

agentRenameEl.addEventListener("click", () => {
  const agent = agentBySlug(currentSlug);
  if (agent) renameAgent(agent);
});

agentDeleteEl.addEventListener("click", () => {
  const agent = agentBySlug(currentSlug);
  if (agent) deleteAgent(agent);
});

function setComposerEnabled(enabled) {
  inputEl.disabled = !enabled;
  inputEl.placeholder = enabled ? "Спросите что-нибудь…"
                                : "Создайте агента в настройках (⚙)";
}

async function refreshStats() {
  const agent = agentBySlug(currentSlug);
  if (!agent) { statsEl.textContent = "агентов нет — откройте настройки (⚙)"; return; }
  try {
    const data = await api(`/api/stats?agent=${encodeURIComponent(agent.slug)}`);
    statsEl.textContent = data.total_chunks
      ? `«${agent.title}»: ${data.total_chunks} фрагментов · ${data.sources_count} книг в базе`
      : `«${agent.title}»: база пуста — загрузите книги и нажмите «Индексировать»`;
  } catch (error) {
    statsEl.textContent = error.message;
  }
}

/* ---- Модели и ключи --------------------------------------------------- */
const savedModel = (provider) => localStorage.getItem(`catlu.model.${provider}`) || "";
const saveModel = (provider, model) => localStorage.setItem(`catlu.model.${provider}`, model || "");

function updateModelLabel() {
  const provider = currentProvider();
  const model = currentModel();
  modelLabelEl.textContent = model ? `${provider} · ${model}` : "выбрать модель";
}

/** Разбирает выбор вида «[groq] llama-3.1-8b-instant». */
function choice(value) {
  const match = String(value).match(/^\[([^\]]+)\]\s+(.+)$/);
  return match ? { provider: match[1], model: match[2] } : null;
}

function updateModelStatus() {
  const info = providers[currentProvider()];
  updateModelLabel();
  if (!info) { setModelStatus(""); return; }
  const models = (allModels[currentProvider()] || {}).models || [];
  const free = models.filter((model) => model.free).length;
  if (info.needs_key && !info.configured) {
    setModelStatus(`Нужен ключ для «${info.label}» — нажмите «Ввести API-ключ», чтобы увидеть модели.`);
  } else if (!currentModel()) {
    setModelStatus("Выберите модель из списка.");
  } else if (free) {
    setModelStatus(`Бесплатных моделей: ${free}.`);
  } else {
    setModelStatus("");
  }
}

/** Наполняет список моделей для провайдера + бесплатные модели других. */
function renderOptions(provider) {
  modelListEl.innerHTML = "";

  const own = (allModels[provider] || {}).models || [];
  const free = own.filter((model) => model.free);
  const paid = own.filter((model) => !model.free).slice(0, MAX_OWN_PAID);
  free.concat(paid).forEach((model) => {
    const option = el("option", null, model.free ? `${model.id} — бесплатно` : model.id);
    option.value = model.id;
    option.label = option.textContent;
    modelListEl.appendChild(option);
  });

  Object.keys(allModels).forEach((name) => {
    if (name === provider) return;
    const info = allModels[name];
    (info.models || []).filter((model) => model.free).slice(0, MAX_OTHER).forEach((model) => {
      const option = el("option", null, `${model.id} — бесплатно (${info.label || name})`);
      option.value = `[${name}] ${model.id}`;
      option.label = option.textContent;
      modelListEl.appendChild(option);
    });
  });
}

function selectModel(provider) {
  const info = allModels[provider] || {};
  const models = info.models || [];
  const ids = models.map((model) => model.id);
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
  const first = models.find((model) => model.free) || models[0];
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
    data = await api("/api/providers");
  } catch {
    setModelStatus("Не удалось получить список провайдеров.");
    return;
  }
  providers = {};
  (data.providers || []).forEach((provider) => { providers[provider.name] = provider; });
  defaultProvider = (data.default || {}).provider || "ollama";
  defaultModel = (data.default || {}).model || "";

  providerEl.innerHTML = "";
  (data.providers || []).forEach((provider) => {
    const option = el("option", null,
      provider.configured || !provider.needs_key ? provider.label : `${provider.label} — нужен ключ`);
    option.value = provider.name;
    providerEl.appendChild(option);
  });

  // Сразу показываем запасной список из реестра, затем уточняем его
  // актуальными данными провайдеров (это может занять пару секунд).
  allModels = {};
  (data.providers || []).forEach((provider) => {
    allModels[provider.name] = {
      provider: provider.name,
      label: provider.label,
      preferred: "",
      source: "fallback",
      configured: provider.configured,
      models: (provider.models || []).map((id) => ({ id, free: id.endsWith(":free") })),
    };
  });

  const saved = localStorage.getItem("catlu.provider");
  providerEl.value = providers[saved] ? saved : defaultProvider;
  fillModels(providerEl.value);
  updateModelStatus();
  await refreshModels();
}

/** Запрашивает списки моделей у всех провайдеров и перерисовывает поле. */
async function refreshModels() {
  try {
    const data = await api("/api/models?all=1");
    allModels = {};
    (data.providers || []).forEach((provider) => { allModels[provider.provider] = provider; });
  } catch {
    allModels = {};
  }
  fillModels(currentProvider());
  updateModelStatus();
}

providerEl.addEventListener("change", () => {
  localStorage.setItem("catlu.provider", currentProvider());
  fillModels(currentProvider());
  updateModelStatus();
});

modelEl.addEventListener("change", applyChoice);

/** Если выбран пункт вида «[openrouter] ... :free», переключаем и провайдера. */
function applyChoice() {
  const pick = choice(modelEl.value);
  if (pick && providers[pick.provider]) {
    providerEl.value = pick.provider;
    localStorage.setItem("catlu.provider", pick.provider);
    modelEl.value = pick.model;
    renderOptions(pick.provider);
  }
  saveModel(currentProvider(), currentModel());
  updateModelStatus();
}

keyBtn.addEventListener("click", async () => {
  const name = currentProvider();
  const label = (providers[name] || {}).label || name;
  const key = window.prompt(`API-ключ для «${label}»:\n(хранится только на этом компьютере)`);
  if (key == null) return;
  try {
    await api("/api/keys", {
      method: "POST", headers: JSON_HEADER,
      body: JSON.stringify({ provider: name, key }),
    });
    showNote(key.trim() ? "Ключ сохранён, обновляю список моделей…" : "Ключ удалён");
    await loadProviders();
  } catch (error) {
    showNote(error.message, "error");
  }
});

/* ---- Разметка ответа ---------------------------------------------------
   Модель отвечает в markdown: заголовки, списки, таблицы. Рисуем это HTML,
   иначе на экране видны «**» и «|». Разметка небольшая, но с таблицами,
   потому что рецепты удобнее читать таблицей.
   ---------------------------------------------------------------------- */

const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const escapeHtml = (text) => String(text).replace(/[&<>"]/g, (c) => HTML_ESCAPES[c]);

const LIST_ITEM = /^\s*(?:[-*+]|\d+[.)])\s+/;
const TABLE_SEPARATOR = /^\s*\|?[\s:|-]*-[\s:|-]*$/;

/** Внутристрочная разметка: код, жирный, курсив, зачёркнутый, ссылки. */
function renderInline(text) {
  const codes = [];
  let out = escapeHtml(text).replace(/`([^`\n]+)`/g, (_, code) => {
    codes.push(code);
    return `\u0001${codes.length - 1}\u0001`;
  });
  out = out
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(«"'])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return out.replace(/\u0001(\d+)\u0001/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
}

/** Строка вида «| Грибы | 15 г |» превращается в ячейки. */
const tableCells = (row) => row
  .replace(/^\s*\|/, "").replace(/\|\s*$/, "")
  .split("|").map((cell) => cell.trim());

function renderTable(rows) {
  const head = tableCells(rows[0]);
  const body = rows.slice(2).map(tableCells);
  const section = (tag, cells) => `<tr>${cells
    .map((cell) => `<${tag}>${renderInline(cell)}</${tag}>`).join("")}</tr>`;
  const parts = [`<div class="table-wrap"><table><thead>${section("th", head)}</thead>`];
  if (body.length) parts.push(`<tbody>${body.map((row) => section("td", row)).join("")}</tbody>`);
  parts.push("</table></div>");
  return parts.join("");
}

/** Список: вложенные пункты помечаем классом, отступ делает CSS. */
function renderList(items, ordered) {
  const inside = items.map((item) => {
    const level = item.level ? ' class="nested"' : "";
    return `<li${level}>${renderInline(item.text)}</li>`;
  }).join("");
  return ordered ? `<ol>${inside}</ol>` : `<ul>${inside}</ul>`;
}

function renderMarkdown(text) {
  const lines = String(text).replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let i = 0;

  const flush = () => {
    if (!paragraph.length) return;
    html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    paragraph = [];
  };

  while (i < lines.length) {
    const line = lines[i];

    // Блок кода ``` ... ```
    if (/^\s*```/.test(line)) {
      flush();
      const code = [];
      i += 1;
      while (i < lines.length && !/^\s*```/.test(lines[i])) { code.push(lines[i]); i += 1; }
      i += 1;
      html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    // Таблица: строка с «|» и разделитель «|---|» под ней
    if (line.includes("|") && i + 1 < lines.length &&
        TABLE_SEPARATOR.test(lines[i + 1]) && lines[i + 1].includes("-")) {
      flush();
      const rows = [lines[i], lines[i + 1]];
      i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        rows.push(lines[i]);
        i += 1;
      }
      html.push(renderTable(rows));
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flush();
      const level = Math.min(heading[1].length + 2, 6);
      html.push(`<h${level}>${renderInline(heading[2].trim())}</h${level}>`);
      i += 1;
      continue;
    }

    if (LIST_ITEM.test(line)) {
      flush();
      const ordered = /^\s*\d+[.)]\s+/.test(line);
      const items = [];
      while (i < lines.length && LIST_ITEM.test(lines[i])) {
        items.push({
          level: /^\s{2,}/.test(lines[i]) ? 1 : 0,
          text: lines[i].replace(LIST_ITEM, "").trim(),
        });
        i += 1;
      }
      html.push(renderList(items, ordered));
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      flush();
      const quote = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quote.push(lines[i].replace(/^\s*>\s?/, "").trim());
        i += 1;
      }
      html.push(`<blockquote>${renderInline(quote.join(" "))}</blockquote>`);
      continue;
    }

    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) {   // --- разделитель
      flush();
      html.push("<hr>");
      i += 1;
      continue;
    }

    if (!line.trim()) {                          // пустая строка — конец абзаца
      flush();
      i += 1;
      continue;
    }

    paragraph.push(line.trim());
    i += 1;
  }

  flush();
  return html.join("");
}

/* ---- Чат -------------------------------------------------------------- */
function addMessage(role, text) {
  const wrap = el("div", "msg " + role);
  const col = el("div", "col");
  const bubble = el("div", "bubble");
  if (role === "user") bubble.textContent = text;   // вопрос показываем как есть
  col.appendChild(bubble);
  wrap.appendChild(col);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return { wrap, bubble, col };
}

/** Читаемое имя файла: «..._Litmir_club_949271_original_ocred.pdf» → название. */
function shortSource(name) {
  let text = String(name || "Неизвестный источник");
  text = text.replace(/\.[A-Za-z0-9]{2,5}$/, "");           // расширение
  text = text.replace(/[_★*]+/g, " ").replace(/\s+/g, " ").trim();
  // Технические хвосты от файлохранилищ: « Litmir club 949270 ocred».
  text = text.replace(
    /\s*[-–—]?\s*(?:ocred|original|litmir|rulit|sovietime|библиотека|bookz)[\s\S]*$/i, "").trim();
  return text || String(name);
}

/** Источники — внизу ответа, номера совпадают со ссылками в тексте ответа. */
function addSources(wrap, sources) {
  const found = (sources || []).filter((source) => source && source.source);
  if (!found.length) return;

  const groups = [];
  const byKey = new Map();
  for (const source of found) {
    const key = `${source.source}|${source.page}`;
    let group = byKey.get(key);
    if (!group) {
      group = {
        source: source.source,
        page: source.page,
        unit: source.unit || "стр.",
        snippet: source.snippet || "",
        indexes: [],
      };
      byKey.set(key, group);
      groups.push(group);
    }
    if (source.index != null) group.indexes.push(source.index);
  }

  const box = el("details", "sources");
  box.open = true;
  const title = el("summary", "sources-head", `📚 Источники · ${groups.length}`);
  title.title = "Нумерация совпадает со ссылками «Источник N» в ответе";
  box.appendChild(title);

  const list = el("div", "source-list");
  for (const group of groups) {
    const row = el("div", "source");
    row.appendChild(el("span", "src-index", group.indexes.join(", ") || "•"));
    const name = el("span", "src-name", shortSource(group.source));
    const snippet = group.snippet ? `\n\n${group.snippet.trim()}` : "";
    name.title = `${group.source}${snippet}`;
    row.appendChild(name);
    row.appendChild(el("span", "src-page",
      group.page != null ? `${group.unit} ${group.page}` : ""));
    list.appendChild(row);
  }
  box.appendChild(list);
  (wrap.querySelector(".col") || wrap).appendChild(box);
  chatEl.scrollTop = chatEl.scrollHeight;
}

/** Последний элемент ответа: в него дописываем мигающий курсор. */
function tailOf(node) {
  const last = node.lastElementChild;
  if (!last) return node;
  if (last.tagName === "TABLE") {
    const cells = last.querySelectorAll("tr:last-child > *");
    return tailOf(cells.length ? cells[cells.length - 1] : last);
  }
  return tailOf(last);
}

/** Показывает ответ: markdown + курсор, пока текст ещё идёт. */
function renderAnswer(bubble, text, typing) {
  bubble.innerHTML = renderMarkdown(text);
  if (typing) tailOf(bubble).appendChild(el("span", "cursor"));
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
  renderAnswer(bubble, "", true);

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
        agent: currentSlug,
      }),
    });
  } catch {
    bubble.textContent = "⚠️ Нет связи с сервером.";
    return;
  }

  if (!response.ok || !response.body) {
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
        renderAnswer(bubble, text, true);
        chatEl.scrollTop = chatEl.scrollHeight;
      } else if (ev.event === "sources") {
        sources = ev.data;
      } else if (ev.event === "model") {
        setStatus(`${ev.data.provider} · ${ev.data.model} · агент «${ev.data.agent}»`);
      } else if (ev.event === "error") {
        text += (text ? "\n\n" : "") + "⚠️ " + (ev.data.message || "Ошибка");
        renderAnswer(bubble, text, true);
      }
    }
  }

  renderAnswer(bubble, text, false);
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
  if (!text || !currentSlug) return;
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

/* ---- Запуск ----------------------------------------------------------- */
async function boot() {
  setComposerEnabled(false);
  await refreshAgents();
  if (!agents.length) {
    openPanel("agents");
    showNote("Создайте первого агента: это папка со своими книгами и своей базой.");
  }
  await loadProviders();
}

boot();

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}
