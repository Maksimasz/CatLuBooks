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
const agentTopicEl = document.getElementById("agent-topic");
const agentCreateEl = document.getElementById("agent-create");
const agentListEl = document.getElementById("agent-list");
const agentPromptEl = document.getElementById("agent-prompt");
const agentPromptTopicEl = document.getElementById("agent-prompt-topic");
const agentPromptStatusEl = document.getElementById("agent-prompt-status");
const agentPromptSaveEl = document.getElementById("agent-prompt-save");
const agentPromptComposeEl = document.getElementById("agent-prompt-compose");
const bookFilesEl = document.getElementById("book-files");
const indexBtnEl = document.getElementById("index-btn");
const reindexBtnEl = document.getElementById("reindex-btn");
const jobEl = document.getElementById("job");
const bookListEl = document.getElementById("book-list");
const ocrToggleEl = document.getElementById("ocr-toggle");
const mergeSwitchEl = document.getElementById("merge-switch");
const mergeStatusEl = document.getElementById("merge-status");
const ocrStatusEl = document.getElementById("ocr-status");
const ocrHintEl = document.getElementById("ocr-hint");
const providerEl = document.getElementById("provider");
const modelEl = document.getElementById("model-input");
const modelListEl = document.getElementById("model-list");
const keyBtn = document.getElementById("keybtn");
const modelsRefreshEl = document.getElementById("models-refresh");
const modelStatusEl = document.getElementById("model-status");
const baseStatsEl = document.getElementById("base-stats");
const historyListEl = document.getElementById("history-list");
const historyClearEl = document.getElementById("history-clear");
const overlayEl = document.getElementById("overlay");
const overlayTextEl = document.getElementById("overlay-text");
const sourcesToggleEl = document.getElementById("sources-toggle");
const sourcesPanelEl = document.getElementById("sources-panel");
const sourcesPanelListEl = document.getElementById("sources-panel-list");
const sourcesCloseEl = document.getElementById("sources-close");
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
let indexFile = "";              // какую книгу сервер читает прямо сейчас
let jobTicks = 0;                // сколько раз спросили сервер (для редких обновлений списка)
let booksCache = [];             // книги активного агента: список и подсказки OCR
let ocrLoaded = false;           // ответил ли сервер про tesseract
let ocrState = { available: false, enabled: false, version: "",
                 languages: [], has_russian: false, install_url: "" };

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
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    // Сервер не отвечает: он перезапускается, остановлен или недоступен по сети.
    throw new Error("Сервер недоступен — проверьте, запущен ли он");
  }
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

/** «3 мин 20 с» — чтобы во время индексации было видно, что работа идёт. */
function humanTime(seconds) {
  const total = Math.round(Number(seconds) || 0);
  const minutes = Math.floor(total / 60);
  if (!minutes) return `${total} с`;
  return `${minutes} мин ${total % 60} с`;
}

/** Список «ключ — значение»: вкладки «База» и «Сервер» показывают одно и то же. */
function renderStats(container, rows) {
  container.innerHTML = "";
  rows.forEach(([key, value]) => {
    const row = el("div", "stat");
    row.appendChild(el("span", "stat-key", key));
    row.appendChild(el("span", "stat-value", value));
    container.appendChild(row);
  });
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
  if (name === "books") { refreshBooks(); loadOcr(); loadMerge(); }
  if (name === "base") refreshAgents();
  if (name === "server") refreshServer();
  if (name === "history") refreshHistory();
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
  renderAgentPrompt();
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
    agentListEl.appendChild(el("div", "empty", "Пока ни одного агента. Создайте первого — например «Документы»."));
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
  renderAgentPrompt();
  renderBase();
  refreshStats();
  refreshBooks();
}

/* ---- Промпт агента: свой у каждого раздела знаний --------------------- */
function renderAgentPrompt() {
  const agent = agentBySlug(currentSlug);
  agentPromptEl.disabled = !agent;
  agentPromptTopicEl.disabled = !agent;
  agentPromptSaveEl.disabled = !agent;
  agentPromptComposeEl.disabled = !agent;
  // Не затираем то, что человек печатает прямо сейчас: список агентов
  // обновляется и после фоновой индексации.
  if (document.activeElement !== agentPromptEl) {
    agentPromptEl.value = agent ? (agent.prompt || "") : "";
  }
  if (document.activeElement !== agentPromptTopicEl) {
    agentPromptTopicEl.value = agent ? (agent.description || "") : "";
  }
  agentPromptStatusEl.textContent = !agent
    ? "выберите агента"
    : (agent.prompt
        ? "свой промпт — можно поправить"
        : "своего промпта нет: работает общий из config.yaml");
}

function replaceAgent(updated) {
  agents = agents.map((item) => (item.slug === updated.slug ? { ...item, ...updated } : item));
  renderAgentList();
  if (updated.slug === currentSlug) renderAgentPrompt();
}

agentPromptSaveEl.addEventListener("click", async () => {
  const agent = agentBySlug(currentSlug);
  if (!agent) return;
  try {
    const updated = await api(`/api/agents/${encodeURIComponent(agent.slug)}/prompt`, {
      method: "PUT", headers: JSON_HEADER,
      body: JSON.stringify({
        prompt: agentPromptEl.value,
        description: agentPromptTopicEl.value,
      }),
    });
    replaceAgent(updated);
    showNote("Промпт сохранён");
  } catch (error) {
    showNote(error.message, "error");
  }
});

agentPromptComposeEl.addEventListener("click", async () => {
  const agent = agentBySlug(currentSlug);
  if (!agent) return;
  agentPromptComposeEl.disabled = true;
  const label = agentPromptComposeEl.textContent;
  agentPromptComposeEl.textContent = "Составляю…";
  try {
    await composePromptFor(agent.slug, agentPromptTopicEl.value);
  } finally {
    agentPromptComposeEl.textContent = label;
    agentPromptComposeEl.disabled = false;
  }
});

/* Промпт составляет модель: у облака это секунды, у локальной модели на CPU —
   до нескольких минут, поэтому создание агента его не ждёт. */
async function composePromptFor(slug, description) {
  const agent = agentBySlug(slug);
  if (!agent) return null;
  try {
    const updated = await api(`/api/agents/${encodeURIComponent(slug)}/prompt`, {
      method: "POST", headers: JSON_HEADER,
      body: JSON.stringify({
        description,
        provider: currentProvider(),
        model: currentModel(),
      }),
    });
    replaceAgent(updated);
    showNote(updated.prompt_error ||
      `Промпт агента «${updated.title}» составлен и сохранён`);
    return updated;
  } catch (error) {
    showNote(`Промпт не составлен: ${error.message}`, "error");
    return null;
  }
}

agentCreateEl.addEventListener("click", createAgent);
agentTitleEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); createAgent(); }
});

async function createAgent() {
  const title = agentTitleEl.value.trim();
  const topic = agentTopicEl.value.trim();
  if (!title) { agentTitleEl.focus(); return; }
  agentCreateEl.disabled = true;
  agentCreateEl.textContent = "Создаю…";
  try {
    const agent = await api("/api/agents", {
      method: "POST", headers: JSON_HEADER,
      body: JSON.stringify({
        title, description: topic, generate_prompt: false,
      }),
    });
    agentTitleEl.value = "";
    agentTopicEl.value = "";
    currentSlug = agent.slug;
    localStorage.setItem("catlu.agent", agent.slug);
    history.length = 0;
    chatEl.innerHTML = "";
    await refreshAgents();
    selectTab("books");             // дальше обычно загружают книги
    // Промпт составляет модель, и на слабом CPU это может занять минуту:
    // папку агента не держим из-за этого, книги можно загружать сразу.
    showNote(`Агент «${agent.title}» создан. Загрузите книги и нажмите «Индексировать».`);
    composePromptFor(agent.slug, topic);
  } catch (error) {
    showNote(error.message, "error");
  } finally {
    agentCreateEl.disabled = false;
    agentCreateEl.textContent = "Создать";
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
    booksCache = [];
    bookListEl.appendChild(el("div", "empty", "Сначала выберите или создайте агента."));
    bookFilesEl.disabled = true;
    indexBtnEl.disabled = true;
    reindexBtnEl.disabled = true;
    jobEl.hidden = true;
    renderOcrHint();
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
  booksCache = books;
  bookListEl.innerHTML = "";
  if (!hasBooksToRead()) {
    bookListEl.appendChild(el("div", "empty", books.length
      ? "В папке книг нет: ниже остались только фрагменты книг, убранных мимо интерфейса."
      : "В папке агента пока нет книг."));
  }
  books.forEach((book) => bookListEl.appendChild(bookRow(book)));
  indexBtnEl.disabled = !hasBooksToRead() || indexRunning;
  reindexBtnEl.disabled = indexBtnEl.disabled;
  renderOcrHint();
  pollJob();
}

/** Подновляет подписи книг во время индексации, не перестраивая список:
    видно, какая книга читается и сколько из неё уже записано, а прокрутка
    списка при этом не сбивается. */
async function refreshBookStates() {
  if (!currentSlug) return;
  const slug = currentSlug;
  let data;
  try {
    data = await api(`/api/agents/${encodeURIComponent(slug)}/books`);
  } catch {
    return;
  }
  if (slug !== currentSlug) return;
  booksCache = data.books || [];
  indexBtnEl.disabled = !hasBooksToRead() || indexRunning;
  reindexBtnEl.disabled = indexBtnEl.disabled;
  applyBookStates();
  renderOcrHint();
}

function applyBookStates() {
  const byName = new Map(booksCache.map((book) => [book.name, book]));
  bookListEl.querySelectorAll(".row").forEach((row) => {
    const book = byName.get(row.dataset.name);
    if (!book) return;
    const sub = row.querySelector(".row-sub");
    if (sub) sub.textContent = bookLine(book);
    row.classList.toggle("reading", isReading(book));
  });
}

const isReading = (book) => Boolean(indexRunning) && book.name === indexFile;

/** Подпись книги: размер и что с ней уже сделали (или делают сейчас). */
function bookLine(book) {
  const size = book.size_text || (book.size ? humanSize(book.size) : "");
  return [size, bookState(book)].filter(Boolean).join(" · ");
}

/** Есть ли в папке книги, которые можно читать (без убранных мимо интерфейса). */
const hasBooksToRead = () => booksCache.some((book) => !book.missing);

function bookState(book) {
  if (book.missing) {
    return `файла нет — прежних фрагментов ${book.chunks}`;
  }
  if (isReading(book)) {
    return book.chunks ? `фрагментов ${book.chunks} — читается сейчас` : "читается сейчас…";
  }
  if (book.chunks) return `фрагментов ${book.chunks}`;
  return indexRunning ? "в очереди" : "не проиндексирована";
}

/** Кнопка удаления книги: только файл (📄) или вместе с фрагментами (🗑). */
function deleteButton(icon, mode, title) {
  const button = el("button", mode === "full" ? "icon danger" : "icon", icon);
  button.type = "button";
  button.dataset.act = "delete";
  button.dataset.mode = mode;
  button.title = title;
  return button;
}

function bookRow(book) {
  const row = el("div", "row");
  row.dataset.name = book.name;
  row.classList.toggle("reading", isReading(book));

  const main = el("div", "row-main");
  const title = el("div", "row-title", book.name);
  if (book.missing) title.appendChild(el("span", "badge warn", "нет файла"));
  if (!book.supported) title.appendChild(el("span", "badge warn", "формат не читается"));
  if (book.needs_ocr) title.appendChild(el("span", "badge warn", "скан"));
  main.appendChild(title);
  main.appendChild(el("div", "row-sub", bookLine(book)));
  row.appendChild(main);

  const actions = el("div", "row-actions");
  if (book.missing) {
    actions.appendChild(deleteButton("🗑", "full",
      "Убрать фрагменты из базы (файла уже нет)"));
  } else {
    actions.appendChild(deleteButton("📄", "file",
      "Удалить только файл с диска — фрагменты в базе останутся"));
    actions.appendChild(deleteButton("🗑", "full",
      "Удалить книгу и её фрагменты из базы"));
  }
  row.appendChild(actions);
  return row;
}

bookListEl.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-act='delete']");
  if (!button || !currentSlug) return;
  const name = button.closest(".row").dataset.name;
  const book = booksCache.find((item) => item.name === name) || {};
  const mode = button.dataset.mode || "full";
  let question;
  if (book.missing) {
    question = `Файла «${name}» в папке уже нет. Убрать из базы оставшиеся фрагменты (${book.chunks || 0})?`;
  } else if (mode === "file") {
    question = `Удалить только файл «${name}» с диска?\nФрагменты в базе останутся — знания продолжат искаться.`;
  } else {
    question = `Удалить «${name}» полностью: файл с диска и фрагменты из базы?`;
  }
  if (!window.confirm(question)) return;
  try {
    const data = await api(
      `/api/agents/${encodeURIComponent(currentSlug)}/book?name=${encodeURIComponent(name)}&mode=${mode}`,
      { method: "DELETE" });
    if (mode === "file") {
      showNote(`«${name}»: файл удалён, база не тронута.`);
    } else {
      showNote(data.file
        ? `«${name}» удалена: файл и фрагментов ${data.removed}`
        : `«${name}»: файла уже не было, убрано фрагментов ${data.removed}`);
    }
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

indexBtnEl.addEventListener("click", () => startIndex(false));
reindexBtnEl.addEventListener("click", () => startIndex(true));

async function startIndex(full) {
  if (!currentSlug) return;
  if (full && !window.confirm("Перечитать все книги заново? Это займёт столько же "
                              + "времени, сколько первая индексация.")) return;
  indexBtnEl.disabled = true;
  reindexBtnEl.disabled = true;
  try {
    const suffix = full ? "?full=true" : "";
    const data = await api(`/api/agents/${encodeURIComponent(currentSlug)}/index${suffix}`,
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
    reindexBtnEl.disabled = false;
  }
}

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
    indexFile = job.file || "";
    indexBtnEl.disabled = true;
    reindexBtnEl.disabled = true;
    renderJob(job);
    jobTicks += 1;
    // Раз в пять секунд подновляем и список книг: он должен показывать,
    // какая книга читается и сколько из неё уже записано.
    if (jobTicks % 5 === 0) refreshBookStates();
    jobTimer = setTimeout(pollJob, 1000);
  } else if (jobRunning) {
    jobRunning = false;             // индексация только что закончилась
    indexRunning = false;
    indexFile = "";
    indexBtnEl.disabled = !hasBooksToRead();
    reindexBtnEl.disabled = indexBtnEl.disabled;
    if (job) renderJob(job);
    if (job && !job.added && job.ready) {
      showNote(`Все книги уже были в базе: перечитывать нечего (${job.ready}).`);
    }
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
    const parts = [`в базе ${job.chunks} фрагментов`];
    if (job.added) parts.push(`прочитано книг: ${job.added}`);
    if (job.ready) parts.push(`уже было: ${job.ready}`);
    if (failed) parts.push(`сбоев ${failed}`);
    if (job.skipped) parts.push(`пропущено ${job.skipped}`);
    label = `Готово: ${parts.join(", ")}`;
  } else if (job.file) {
    label = `Читаю книгу ${job.index} из ${job.total || "?"}: ${job.file}`;
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
    // Фрагменты считаются по текущей книге: в базе лежат и прошлые книги.
    const parts = [];
    if (job.chunks) parts.push(`в этой книге: ${job.chunks} фрагментов`);
    if (job.seconds) parts.push(`идёт ${humanTime(job.seconds)}`);
    if (parts.length) line.appendChild(el("span", "job-extra", parts.join(" · ")));
  } else if (error && job.log && job.log.length) {
    line.appendChild(el("span", "job-extra", job.log[job.log.length - 1]));
  }
  showJob(line);
}

/* ---- OCR: распознавание сканов ---------------------------------------
   Tesseract — отдельная программа. Сервер сообщает, установлена ли она и
   какие языки есть; интерфейс либо включает распознавание, либо объясняет,
   что скачать и установить, и предлагает проверить снова.
   ---------------------------------------------------------------------- */
async function loadOcr() {
  try {
    ocrState = await api("/api/ocr");
  } catch {
    ocrState = { available: false, enabled: false, version: "",
                 languages: [], has_russian: false, install_url: "" };
  }
  ocrLoaded = true;
  renderOcr();
}

function renderOcr() {
  const on = Boolean(ocrState.enabled && ocrState.available);
  ocrToggleEl.classList.toggle("on", on);
  ocrToggleEl.setAttribute("aria-checked", on ? "true" : "false");
  ocrToggleEl.disabled = !ocrState.available;

  const parts = [];
  if (!ocrState.available) {
    parts.push("Tesseract не найден");
  } else {
    parts.push(ocrState.enabled ? "включён" : "выключен");
    if (ocrState.version) parts.push(`tesseract ${ocrState.version}`);
    if ((ocrState.languages || []).length) parts.push(ocrState.languages.join(", "));
  }
  ocrStatusEl.textContent = parts.join(" · ");
  renderOcrHint();
}

/* ---- Сведение источников --------------------------------------------- */
let mergeMode = "off";

async function loadMerge() {
  try {
    const data = await api("/api/merge");
    mergeMode = data.mode || "off";
  } catch {
    mergeMode = "off";
  }
  renderMerge();
}

function renderMerge() {
  mergeSwitchEl.querySelectorAll("button").forEach((button) => {
    const on = button.dataset.mode === mergeMode;
    button.classList.toggle("on", on);
    button.setAttribute("aria-pressed", on ? "true" : "false");
  });
  const texts = {
    off: "выключено: модель отвечает по тому, что нашла",
    fast: "быстро: модель сама сведёт найденное в один ответ",
    full: "точно: два прохода — черновик, потом ответ (медленнее)",
  };
  mergeStatusEl.textContent = texts[mergeMode] || texts.off;
}

mergeSwitchEl.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-mode]");
  if (!button) return;
  const mode = button.dataset.mode;
  if (mode === mergeMode) return;
  mergeSwitchEl.querySelectorAll("button").forEach((item) => { item.disabled = true; });
  try {
    const data = await api("/api/merge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    mergeMode = data.mode || mode;
    showNote(mergeMode === "off"
      ? "Сведение выключено."
      : "Сведение включено: ответы будут собираться из всех найденных книг.");
  } catch (error) {
    showNote(error.message, "error");
  } finally {
    mergeSwitchEl.querySelectorAll("button").forEach((item) => { item.disabled = false; });
    renderMerge();
  }
});

/** Подсказка под переключателем: чего не хватает и что делать дальше. */
function renderOcrHint() {
  ocrHintEl.innerHTML = "";
  const scans = booksCache.filter((book) => book.needs_ocr);

  if (!ocrLoaded) {          // ещё не знаем, есть ли tesseract — молчим
    ocrHintEl.hidden = true;
    return;
  }

  if (!ocrState.available) {
    ocrHintEl.hidden = false;
    ocrHintEl.appendChild(el("div", "hint-title", "Чтобы читать сканы, нужен Tesseract"));
    ocrHintEl.appendChild(el("div", "hint-text",
      "Это бесплатная программа распознавания текста. Скачайте установщик для Windows, " +
      "при установке отметьте дополнительный язык Russian (без него русский текст не " +
      "распознаётся), затем нажмите «Проверить снова» — сервер подхватит программу без " +
      "перезапуска. Обычные текстовые PDF, EPUB, TXT и FB2 работают и без неё."));
    const link = el("a", "hint-link", "Скачать Tesseract для Windows");
    link.href = ocrState.install_url || "https://github.com/UB-Mannheim/tesseract/wiki";
    link.target = "_blank";
    link.rel = "noopener";
    ocrHintEl.appendChild(link);
    addHintButton("Проверить снова", "ghost", "ocr-recheck");
    return;
  }

  if (!ocrState.has_russian) {
    ocrHintEl.hidden = false;
    ocrHintEl.appendChild(el("div", "hint-title", "Не установлен русский язык"));
    ocrHintEl.appendChild(el("div", "hint-text",
      "Tesseract найден, но среди его языков нет русского (rus). Запустите установщик " +
      "ещё раз, выберите «Additional language data» и отметьте Russian, затем нажмите " +
      "«Проверить снова»."));
    addHintButton("Проверить снова", "ghost", "ocr-recheck");
    return;
  }

  if (!scans.length) {
    ocrHintEl.hidden = true;
    return;
  }

  const names = scans.slice(0, 3).map((book) => `«${book.name}»`).join(", ");
  const more = scans.length > 3 ? ` и ещё ${scans.length - 3}` : "";
  ocrHintEl.hidden = false;
  if (!ocrState.enabled) {
    ocrHintEl.appendChild(el("div", "hint-title", "Похоже на скан без текста"));
    ocrHintEl.appendChild(el("div", "hint-text",
      `${names}${more}: текстового слоя нет, искать по этим страницам без распознавания ` +
      `не получится. Tesseract установлен — можно включить OCR.`));
    addHintButton("Включить OCR", "primary", "ocr-enable");
  } else {
    ocrHintEl.appendChild(el("div", "hint-title", "OCR включён — осталось проиндексировать"));
    ocrHintEl.appendChild(el("div", "hint-text",
      `${names}${more}: страницы-сканы распознаются во время индексации. Это может занять ` +
      `минуты — примерно 1–2 секунды на страницу, — зато текст появится в базе.`));
    addHintButton("Индексировать", "primary", "ocr-index");
  }
}

function addHintButton(text, kind, action) {
  const button = el("button", kind, text);
  button.type = "button";
  button.dataset.act = action;
  ocrHintEl.appendChild(button);
}

ocrToggleEl.addEventListener("click", toggleOcr);

ocrHintEl.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-act]");
  if (!button) return;
  if (button.dataset.act === "ocr-recheck") loadOcr().then(refreshBooks);
  if (button.dataset.act === "ocr-enable") enableOcr();
  if (button.dataset.act === "ocr-index") startIndex();
});

async function toggleOcr() {
  if (!ocrState.available) { loadOcr(); return; }
  const wanted = !ocrState.enabled;
  try {
    ocrState = await api("/api/ocr", {
      method: "POST", headers: JSON_HEADER,
      body: JSON.stringify({ enabled: wanted }),
    });
    renderOcr();
    showNote(wanted ? "Распознавание сканов включено."
                    : "Распознавание сканов выключено.");
  } catch (error) {
    showNote(error.message, "error");
    await loadOcr();
  }
}

async function enableOcr() {
  try {
    ocrState = await api("/api/ocr", {
      method: "POST", headers: JSON_HEADER,
      body: JSON.stringify({ enabled: true }),
    });
    renderOcr();
    showNote(ocrState.has_russian ? "OCR включён — нажмите «Индексировать»."
                                  : "OCR включён, но русского языка в tesseract нет.");
  } catch (error) {
    showNote(error.message, "error");
    await loadOcr();
  }
}

/* ---- База ------------------------------------------------------------- */
function renderBase() {
  const agent = agentBySlug(currentSlug);
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
  renderStats(baseStatsEl, rows);
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

/* ---- Сервер ----------------------------------------------------------- */
let restarting = false;        // ждём, пока перезапущенный сервер поднимется

function showOverlay(text) {
  overlayTextEl.textContent = text;
  overlayEl.hidden = false;
}

function hideOverlay() {
  overlayEl.hidden = true;
}

/** Проверка связи: сервер упал или вернулся — показываем/прячем оверлей. */
setInterval(async () => {
  if (document.hidden) return;
  let ok = false;
  try {
    const response = await fetch("/health", { cache: "no-store" });
    ok = response.ok;
  } catch { ok = false; }
  if (!ok) {
    serverDown = true;
    failedChecks += 1;
    if (!overlayEl.hidden) return;
    showOverlay(restarting ? "Сервер перезапускается…"
                           : "Сервер недоступен. Проверьте, запущен ли он.");
  } else {
    // Сервер снова отвечает. Если он недоступен был два раза подряд (≈6 секунд) —
    // это перезапуск: обновляем страницу, иначе она останется со старыми
    // данными и нерабочими запросами. Одиночный сбой трогать не будем.
    if (serverDown && failedChecks >= 2) {
      serverDown = false;
      failedChecks = 0;
      hideOverlay();
      reloadPage();
      return;
    }
    failedChecks = 0;
    if (!overlayEl.hidden) hideOverlay();
  }
}, 3000);

const serverStatusEl = document.getElementById("server-status");
const serverRestartEl = document.getElementById("server-restart");
const serverStopEl = document.getElementById("server-stop");
const serverCommandEl = document.getElementById("server-command");

async function refreshServer() {
  let data;
  try {
    data = await api("/api/server");
  } catch (error) {
    showNote(error.message, "error");
    return;
  }
  renderServerStatus(data);
}

function renderServerStatus(data) {
  const uptime = humanTime(data.uptime);
  const rows = [
    ["Адрес для телефона", data.lan_url || "нет (ПК не в сети?)"],
    ["На этом ПК", data.urls[0]],
    ["Порт", String(data.port)],
    ["Работает", uptime],
    ["Процесс", String(data.pid)],
    ["Python", data.python],
    ["Папка проекта", data.folder],
  ];
  renderStats(serverStatusEl, rows);
  if (data.start_command) {
    serverCommandEl.hidden = false;
    serverCommandEl.textContent = "Запустить вручную:\n" + data.start_command;
  }
}

serverRestartEl.addEventListener("click", async () => {
  if (!window.confirm("Перезапустить сервер?\nОн выключится и включится заново через пару секунд.")) return;
  serverRestartEl.disabled = true;
  serverStopEl.disabled = true;
  restarting = true;
  showOverlay("Сервер перезапускается…");
  try {
    await api("/api/server/restart", { method: "POST" });
  } catch (error) {
    restarting = false;
    serverRestartEl.disabled = false;
    serverStopEl.disabled = false;
    hideOverlay();
    showNote(error.message, "error");
    return;
  }
  await waitForServer();
});

serverStopEl.addEventListener("click", async () => {
  if (!window.confirm("Остановить сервер?\nЗапустить обратно можно только на компьютере (команда ниже).")) return;
  serverRestartEl.disabled = true;
  serverStopEl.disabled = true;
  try {
    const data = await api("/api/server/stop", { method: "POST" });
    showNote("Сервер останавливается. Запустить обратно: " + data.start_command.replace(/\\n/g, " "), "error");
  } catch (error) {
    serverRestartEl.disabled = false;
    serverStopEl.disabled = false;
    showNote(error.message, "error");
  }
});

let reloading = false;        // страница уже перезагружается
let serverDown = false;       // сервер только что был недоступен
let failedChecks = 0;         // сколько проверок подряд сервер не отвечал

/** Перезагружает страницу через короткую паузу — чтобы точно сработало. */
function reloadPage() {
  if (reloading) return;
  reloading = true;
  setTimeout(() => location.reload(), 300);
}

/** Ждёт, пока сервер уйдёт и вернётся, и перезагружает страницу. */
async function waitForServer() {
  const started = Date.now();
  // Сначала дожидаемся, пока старый сервер действительно прекратит отвечать:
  // иначе можно перезагрузить страницу, а сервер умрёт сразу после этого.
  while (Date.now() - started < 30000) {
    try {
      await fetch("/health", { cache: "no-store" });
    } catch {
      break;                    // сервер ушёл — теперь ждём его возвращения
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  // Теперь ждём, пока поднимется новый.
  const back = Date.now();
  while (Date.now() - back < 120000) {
    try {
      const response = await fetch("/health", { cache: "no-store" });
      if (response.ok) {
        reloadPage();
        return;
      }
    } catch { /* сервер ещё поднимается */ }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  if (!reloading) {
    showNote("Сервер не ответил за две минуты. Запустите его на компьютере: uv run python app.py", "error");
  }
}

/* ---- История поиска --------------------------------------------------- */
async function refreshHistory() {
  if (!currentSlug) { historyListEl.innerHTML = ""; return; }
  let data;
  try {
    data = await api(`/api/agents/${encodeURIComponent(currentSlug)}/history`);
  } catch (error) { showNote(error.message, "error"); return; }
  renderHistory(data.entries || []);
}

function renderHistory(entries) {
  historyListEl.innerHTML = "";
  if (!entries.length) {
    historyListEl.appendChild(el("div", "empty", "История пуста: ответы чата сохраняются сюда."));
    return;
  }
  entries.forEach((entry) => historyListEl.appendChild(historyItem(entry)));
}

function historyItem(entry) {
  const wrap = el("div", "row history-item");
  wrap.dataset.id = entry.id;

  const main = el("div", "row-main");
  const title = el("div", "row-title", entry.question || "(без вопроса)");
  main.appendChild(title);
  const date = entry.created ? new Date(entry.created * 1000).toLocaleString() : "";
  const preview = (entry.answer || "").replace(/\s+/g, " ").slice(0, 120);
  main.appendChild(el("div", "row-sub", `${date} · ${preview}${preview.length >= 120 ? "…" : ""}`));
  wrap.appendChild(main);

  const actions = el("div", "row-actions");
  const remove = el("button", "icon danger", "🗑");
  remove.type = "button";
  remove.title = "Удалить запись";
  remove.addEventListener("click", async (event) => {
    event.stopPropagation();
    if (!window.confirm("Удалить запись из истории?")) return;
    try {
      await api(`/api/agents/${encodeURIComponent(currentSlug)}/history/${encodeURIComponent(entry.id)}`,
                { method: "DELETE" });
      refreshHistory();
    } catch (error) { showNote(error.message, "error"); }
  });
  actions.appendChild(remove);
  wrap.appendChild(actions);

  // Разворачивание: полный ответ + источники.
  const detail = el("div", "history-detail");
  detail.hidden = true;
  const answer = el("div", "history-answer");
  answer.innerHTML = renderMarkdown(entry.answer || "");
  detail.appendChild(answer);
  if (entry.sources && entry.sources.length) {
    const src = el("div", "sources");
    const list = el("div", "src-list");
    (entry.sources || []).forEach((group, i) => {
      const row = el("div", "src-row");
      row.appendChild(el("span", "src-index", `[${i + 1}]`));
      row.appendChild(el("span", "src-file", group.file || ""));
      row.appendChild(el("span", "src-page",
        group.page != null ? `${group.unit || "стр."} ${group.page}` : ""));
      list.appendChild(row);
    });
    src.appendChild(list);
    detail.appendChild(src);
  }
  wrap.appendChild(detail);

  wrap.addEventListener("click", (event) => {
    if (event.target.closest("button")) return;
    detail.hidden = !detail.hidden;
  });
  return wrap;
}

historyClearEl.addEventListener("click", async () => {
  if (!currentSlug) return;
  if (!window.confirm("Очистить всю историю агента?")) return;
  try {
    const data = await api(`/api/agents/${encodeURIComponent(currentSlug)}/history`,
                          { method: "DELETE" });
    showNote(`История очищена: убрано ${data.removed} записей.`);
    refreshHistory();
  } catch (error) { showNote(error.message, "error"); }
});

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
  } else if (!models.length) {
    const info2 = allModels[currentProvider()] || {};
    setModelStatus(info2.error
      ? info2.error
      : (info2.source === "fallback"
          ? "Загружаю список моделей…"
          : "Список моделей пуст — введите название модели вручную."));
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
async function refreshModels(fresh) {
  try {
    const data = await api(`/api/models?all=1${fresh ? "&fresh=1" : ""}`);
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

modelsRefreshEl.addEventListener("click", async () => {
  modelsRefreshEl.disabled = true;
  const label = modelsRefreshEl.textContent;
  modelsRefreshEl.textContent = "Обновляю…";
  try {
    await refreshModels(true);       // без кэша: сервер могли только запустить
  } finally {
    modelsRefreshEl.textContent = label;
    modelsRefreshEl.disabled = false;
  }
});

/* ---- Разметка ответа ---------------------------------------------------
   Модель отвечает в markdown: заголовки, списки, таблицы. Рисуем это HTML,
   иначе на экране видны «**» и «|». Разметка небольшая, но с таблицами:
   иначе таблицу в ответе не прочитать.
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

/** Источники — в боковой панели справа, номера совпадают со ссылками [N] в ответе. */
function addSources(sources) {
  const found = (sources || []).filter((source) => source && source.source);
  if (!found.length) {
    sourcesPanelListEl.innerHTML = "";
    sourcesPanelListEl.appendChild(el("div", "sources-empty",
      "Источников нет."));
    sourcesToggleEl.disabled = true;
    return;
  }

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
    // У источника могут быть свои номера: на одной странице книги попало
    // несколько фрагментов, и модель сослётся на любой из них.
    const own = (source.indexes && source.indexes.length) ? source.indexes
               : (source.index != null ? [source.index] : []);
    own.forEach((number) => {
      if (!group.indexes.includes(number)) group.indexes.push(number);
    });
  }

  sourcesPanelListEl.innerHTML = "";
  groups.forEach((group, i) => {
    const row = el("div", "source");
    row.appendChild(el("span", "src-index", `[${group.indexes.join(", ") || i + 1}]`));
    const name = el("span", "src-name", shortSource(group.source));
    const snippet = group.snippet ? `\n\n${group.snippet.trim()}` : "";
    name.title = `${group.source}${snippet}`;
    row.appendChild(name);
    row.appendChild(el("span", "src-page",
      group.page != null ? `${group.unit} ${group.page}` : ""));
    sourcesPanelListEl.appendChild(row);
  });
  sourcesToggleEl.disabled = false;
  sourcesPanelEl.hidden = false;
  chatEl.scrollTop = chatEl.scrollHeight;
}

sourcesToggleEl.addEventListener("click", () => {
  sourcesPanelEl.hidden = !sourcesPanelEl.hidden;
});
sourcesCloseEl.addEventListener("click", () => {
  sourcesPanelEl.hidden = true;
});

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
  const { wrap, bubble, col } = addMessage("assistant", "");
  renderAnswer(bubble, "", true);

  let text = "";
  let sources = null;
  let stageEl = null;

  /** Промежуточный шаг — отдельной строкой над ответом, а не внутри текста. */
  const showStage = (message) => {
    if (!stageEl) {
      stageEl = el("div", "stage");
      col.insertBefore(stageEl, bubble);
    }
    stageEl.textContent = "⏳ " + message;
  };
  const hideStage = () => {
    if (stageEl) { stageEl.remove(); stageEl = null; }
  };

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
        hideStage();
        text += ev.data.text || "";
        renderAnswer(bubble, text, true);
        chatEl.scrollTop = chatEl.scrollHeight;
      } else if (ev.event === "sources") {
        sources = ev.data;
      } else if (ev.event === "stage") {
        // Промежуточный шаг (сведение в два прохода, дописывание ответа).
        showStage(ev.data.text || "");
      } else if (ev.event === "model") {
        setStatus(`${ev.data.provider} · ${ev.data.model} · агент «${ev.data.agent}»`);
      } else if (ev.event === "error") {
        text += (text ? "\n\n" : "") + "⚠️ " + (ev.data.message || "Ошибка");
        renderAnswer(bubble, text, true);
      }
    }
  }

  hideStage();
  renderAnswer(bubble, text, false);
  if (sources) addSources(sources);
  // Найденный ответ сохраняется в историю, чтобы потом не искать снова.
  if (text && currentSlug && !text.includes("⚠️")) {
    saveHistory(message, text, sources);
  }
  history.push({ role: "assistant", content: text });
}

async function saveHistory(question, answer, sources) {
  try {
    await api(`/api/agents/${encodeURIComponent(currentSlug)}/history`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        answer,
        sources: sources || [],
        provider: currentProvider(),
        model: currentModel(),
      }),
    });
  } catch { /* история не сохранилась — ответ уже показан, не критично */ }
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
