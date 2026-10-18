"use strict";

const chatEl = document.getElementById("chat");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("input");
const sendEl = document.getElementById("send");
const statsEl = document.getElementById("stats");

const history = [];

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

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
    const page = s.page != null ? `, стр. ${s.page}` : "";
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
      body: JSON.stringify({ message, history: prior }),
    });
  } catch (err) {
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
  .then((s) => {
    statsEl.textContent = `${s.total_chunks} фрагментов · ${s.sources_count} файлов`;
  })
  .catch(() => { statsEl.textContent = "нет данных"; });

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}
