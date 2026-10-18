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
        setStatus(`${ev.data.provider} · ${ev.data.model}`);
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
