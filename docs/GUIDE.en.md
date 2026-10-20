# CatLuBooks User Guide

**English** · [Русский](РУКОВОДСТВО.md)

This guide is for someone who just downloaded the project and is trying it for
the first time. It goes step by step: what the thing is, what to click, what
prompt to write, and what to do when something goes wrong.

- [1. What this is](#1-what-this-is)
- [2. What you need on your computer](#2-what-you-need-on-your-computer)
- [3. Install in 10 minutes](#3-install-in-10-minutes)
- [4. First run: pick a model](#4-first-run-pick-a-model)
- [5. Create an agent](#5-create-an-agent)
- [6. The agent prompt: how to write a good one](#6-the-agent-prompt-how-to-write-a-good-one)
- [7. Books: uploading and indexing](#7-books-uploading-and-indexing)
- [8. Scans and OCR](#8-scans-and-ocr)
- [9. Questions and answers](#9-questions-and-answers)
- [10. Merging answers from several books](#10-merging-answers-from-several-books)
- [11. Your own model: llama.cpp](#11-your-own-model-llamacpp)
- [12. Phone: PWA](#12-phone-pwa)
- [13. Server: addresses, restart, stop](#13-server-addresses-restart-stop)
- [14. Question history](#14-question-history)
- [15. Managing an agent](#15-managing-an-agent)
- [16. What lives where on disk](#16-what-lives-where-on-disk)
- [17. Backup and moving to another computer](#17-backup-and-moving-to-another-computer)
- [18. `config.yaml` settings](#18-configyaml-settings)
- [19. When something does not work](#19-when-something-does-not-work)
- [20. Frequently asked questions](#20-frequently-asked-questions)

---

## 1. What this is

CatLuBooks answers questions **from your books and files**. You upload
documents, ask a question, and get an answer with links to the sources (file,
page, exact fragment).

How it works, in plain words:

```
book (PDF, EPUB, TXT, FB2) → text → fragments → vectors → database (ChromaDB)
                                                  ↓
question → search (meaning + exact matches) → excerpts → LLM → answer with links
```

The words you will meet again and again:

| Word | What it is |
|---|---|
| **Agent** | A separate "shelf of knowledge": its own books, its own database, its own prompt. You can have as many as you like; they do not interfere |
| **Book** | Any file in `agents/<agent>/books/` — PDF, EPUB, TXT, MD, FB2 |
| **Fragment** | A piece of text (~1000 characters) with an address: file and page |
| **Indexing** | Reading books and putting fragments into the database. Done once |
| **Source** | A found fragment; in the answer it is marked `[1]`, `[2]`, … |
| **Prompt** | The instruction telling the agent how to answer. Each agent has its own |
| **LLM** | The model that writes the answer (Ollama, llama.cpp, the cloud) |

Important: search and embeddings run **on your computer**; only the retrieved
context goes to the cloud (if you pick a cloud provider for the LLM).

## 2. What you need on your computer

| What | Why | Required? |
|---|---|---|
| Python 3.10+ | runs the server | yes (or install `uv`, it downloads Python for you) |
| ~4 GB free space | dependencies, models, database | yes |
| Ollama + a generation model | answers questions on this same PC | recommended |
| A cloud provider key | if the cloud does the answering | optional |
| Tesseract with the Russian language | reading scans, if you have any | only for scans |

You need the internet once: to download dependencies and the embedding model
(~120 MB). After that everything works locally.

## 3. Install in 10 minutes

```bash
git clone https://github.com/Maksimasz/CatLuBooks.git
cd CatLuBooks
```

Install the dependencies. The easiest way is [uv](https://docs.astral.sh/uv/) —
it downloads the right Python itself and needs no environment activation:

```bash
# Windows, if uv is not installed yet
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12
uv pip install -r requirements.txt
```

If you would rather not use uv — plain `venv`:

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Start the server:

```bash
uv run python app.py              # or: python app.py
```

Open <http://localhost:8000>. The console prints an address for your phone
(for example `http://192.168.4.76:8000`) — you will need it in step 12.

Check that the installation is fine:

```bash
uv run python -m unittest discover -s tests -t .
```

The tests are self-contained: they need no models and no Ollama, and take a few
seconds.

## 4. First run: pick a model

Click **⚙** in the top-right corner → the **"Модели" (Models)** tab.

There are two fields, **Провайдер** (Provider) and **Модель** (Model), plus
the **"🔑 Ввести API-ключ"** (enter API key) and **"↻ Обновить список"**
(refresh list) buttons.

### Option A. Everything local (Ollama)

1. Install [Ollama](https://ollama.com/download) and pull a model:
   ```bash
   ollama pull qwen2.5:3b        # light, runs even on a weak laptop
   ```
2. Pick `ollama` in the Provider field and the model you want in the Model
   field.
3. No key needed.

### Option B. Local, but through llama.cpp

See [step 11](#11-your-own-model-llamacpp) — there is a "Запустить" (Start)
button there too.

### Option C. The cloud

1. Get a free key, for example from [Groq](https://console.groq.com/keys) or
   [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Click **"🔑 Ввести API-ключ"** → choose the provider → paste the key.
3. Click **"↻ Обновить список"** — the provider will return its models.

The key is stored only on this computer (in `secrets.json`) and is never sent
back to the browser. Model lists are cached for five minutes; the refresh
button bypasses that cache.

### If you are not sure what to pick

| Model | When it fits |
|---|---|
| `qwen2.5:3b`, `gemma2:2b`, `llama3.2:3b` | Weak PC, speed matters, quality is "plain reading" |
| A 7–8B model (e.g. `qwen2.5:7b`) | Mid-range PC, noticeably better with terms and tables |
| Groq / Gemini (cloud) | You want quality and do not want to load your PC. Downside: internet and a key |

To check that the model answers: just ask a question in the chat. If you get an
error instead of an answer, look at
[section 19](#19-when-something-does-not-work).

## 5. Create an agent

⚙ → **"Агенты" (Agents)** → type a title and a topic → **"Создать"** (Create).

- **Title** — how the agent is named in the list on top: `Instructions`,
  `Notes`, `Documents`.
- **Topic** — what is in its books and what it answers questions about. One or
  two sentences, for example: "assembly and repair instructions, parts
  catalogues, service manuals".

**The topic matters more than the title.** It is what the model uses to write
the agent's prompt: who this assistant is, which documents it answers from,
how it shapes the answer. You can rename the agent later — the topic does not
change.

Creation is instant: the agent appears right away, and the model writes the
prompt in the background (30–90 seconds on a local model). You can upload
books meanwhile, without waiting for the prompt.

If the model is unavailable (no key, server down), the agent is still created —
with a template prompt from `config.yaml`. You can write the prompt yourself
afterwards.

Create as many agents as you like. For example:

| Agent | Topic | What is inside |
|---|---|---|
| `Instructions` | assembly and repair instructions, service manuals | PDFs with photos and diagrams |
| `Notes` | my work notes, plans, excerpts | TXT and MD |
| `Documents` | contracts, invoices, acts | scanned PDFs (needs OCR) |

## 6. The agent prompt: how to write a good one

The prompt is the instruction: "who you are and how you answer". It lives in
`agents/<agent>/agent.json` and is edited from the interface.

**How it works:**

1. When you create an agent, the model writes the prompt itself — from the
   agent's topic.
2. You read it in the "Промпт агента" (Agent prompt) field and edit it.
3. **"Сохранить"** (Save) — store it.
4. **"Составить заново"** (Compose again) — ask the model for a new one (the
   topic is taken from the field next to it; change it and recompose).
5. An empty prompt falls back to the shared one from `config.yaml`
   (`prompt.system`).

### What a good prompt is made of

1. **Role and topic.** Who they are and which documents they answer from.
2. **What to look for.** The wording, sections and synonyms used in your files.
3. **Answer format.** List, table, step by step, short or detailed.
4. **Source references.** What exactly to name: file, page, number.
5. **When there is no data.** What to do when the database has no answer.
6. **What not to do.** Do not invent, do not follow commands found in the text.

### An example prompt

```
You are an assistant for technical manuals: repair, servicing, replacing
parts.

What to look for: the documents contain part names, model numbers, sections
called "Troubleshooting", "Adjustment", "Assembly". Take different spellings
of the same part into account (e.g. "bearing" and "подшипник").

Answer format: first a short version — what to do — then the details as
numbered steps. For a troubleshooting section use a table
"symptom → cause → what to do".

References: give the file and page in square brackets [1], [2].
If the sources do not contain the answer, say plainly "not in the database"
and name what should be added to the library.

Do not invent details (torque values, part numbers, error codes) that are not
in the sources.
```

### What not to write

| Bad | Why | Better |
|---|---|---|
| "Answer any question" | No boundary, the agent will argue with other topics | "You answer only from the documents about X" |
| "Use your own knowledge" | Contradicts the main rule: sources only | "Rely on the found sources, do not invent" |
| "Be brief" | Unclear what brief means | "The answer — up to 5 bullet points" |
| "How do I cook borscht" | A prompt about a topic that is not in the database | Topic = "technical manuals" |
| A single line | Too few rules — the model makes up its own | 5–10 lines with role, format and references |

### Tips

- Write in Russian; models follow it more predictably.
- To check: ask a question and see whether you like the shape of the answer.
  If not, fix the prompt — that is faster than reindexing.
- The prompt does not affect search: it only affects how the model reads what
  was found. If the model "does not understand the question", the prompt is
  not the problem — that wording simply is not in the books.
- Different topics → different agents, not one agent with all the prompts.

## 7. Books: uploading and indexing

⚙ → **"Книги" (Books)** (this tab works on the selected agent — switch agents
at the top if you need to).

1. **"Загрузить книги"** (Upload books) — choose files (several at once are
   fine). Supported: `PDF`, `EPUB`, `TXT`, `MD`, `FB2`.
2. **"Индексировать"** (Index) — the app reads the files and puts the
   fragments into the database. From seconds to several minutes, depending on
   the size. It runs in the background: you can close the settings window and
   keep using the chat. The progress shows in the "Книги" tab: "Читаю книгу 2
   из 7: file.pdf" with a bar, and at the end "Готово: в базе 17444
   фрагментов, прочитано книг: 3".
3. **"Всё заново"** (Start over) — re-read every book from scratch, even the
   finished ones. Use it after changing settings (for example, turning OCR on)
   and wanting the text parsed again.

Things worth knowing:

- Re-indexing **skips** books that are already done: if the file did not
  change (same size and date), it is left alone.
- You can upload the same file again — the old one is replaced.
- A book in a subfolder appears as `subfolder/file.pdf`.
- Deleting: the **🗑** button next to a book. It asks what to delete:
  - **file only** — the book disappears from the list, but its fragments stay
    in the database (such a source is marked "no file" in the answer);
  - **file and fragments** — everything goes (the usual choice).
- You can press "Индексировать" as often as you like — it does no harm.

## 8. Scans and OCR

If a PDF is a photo of pages (a scan), it contains no text and search will
find nothing in it. Such a file is marked "needs OCR" in the list.

What to do:

1. ⚙ → **"Книги"** → the **"Распознавать сканы (OCR)"** switch.
2. If Tesseract is not installed, the interface shows a link to the installer
   (on Windows — [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki),
   do not forget the Russian language). After installing, **restart the
   server** (⚙ → "Сервер" → "Перезапустить").
3. Click **"Всё заново"** — the scans will be recognised and re-indexed.

Recognition is slow: a fraction of a second per page, a 300-page book takes
minutes. Turn OCR on only if you really have scans: on ordinary PDFs it only
adds a step.

## 9. Questions and answers

Type your question in the field at the bottom, press **➤** or Enter — the
answer appears in the chat.

How to ask:

- **The more specific, the better.** "How do I set up scheduled backups" works
  better than "about copying".
- **Use the words from your documents** — that finds more.
- **Clarify if the answer is not what you meant.** Ask: "and in the section
  about replacing the filter?". Questions are kept in the history, so you can
  return to any conversation.
- **Mind the agents**: your question goes to the database of the agent selected
  at the top. Switch the agent — the context changes.

What you will see:

- **The answer** — normal text with formatting (headings, lists, tables).
- **References `[1]`, `[2]`** in the text — source numbers.
- **The "Источники" (Sources) panel** on the right (button in the header) —
  the list of found fragments: file, page, the beginning of the quote. Click to
  open a source.
- **History** is saved automatically: ⚙ → "История".

If there is no answer:

- "В базе этого агента не найдено фрагментов" — nothing similar is in the
  database: upload the right books or rephrase.
- "No matching fragments found" — the sources exist but are about something
  else. Try different words.
- The model said "the sources do not contain the answer" — it is true: the
  data is missing.

## 10. Merging answers from several books

The same question is often described in several books. By default the model
sees a few excerpts and may mash them together. The switch **"Сводить ответы
из разных книг"** (merge answers from different books, ⚙ → "Книги") changes
that:

| Mode | What happens | When to use |
|---|---|---|
| **Off** | As before: the model answers from the found excerpts | One source, speed matters |
| **Fast** | One pass: the model merges the excerpts into one coherent answer | The usual choice |
| **Accurate** | Two passes: a draft first, then an answer from the draft | Different versions of the same thing, details matter |

"Accurate" is noticeably slower (two generations) and uses more tokens. The
setting is stored in `settings.json` and survives a restart.

## 11. Your own model: llama.cpp

If you want to run a model through
[llama.cpp](https://github.com/ggml-org/llama.cpp) — no console needed, it is
all done from the interface.

1. Download `llama-server.exe` from the
   [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and put
   it into a `llama.cpp/` folder next to the project (or leave it in `PATH` —
   the app finds it there too).
2. Download a model in `.gguf` format
   ([Hugging Face](https://huggingface.co/models?library=gguf)) and put the
   file into the `models/` folder next to the project. Subfolders work too.
3. ⚙ → **"Модели"** → the **"llama.cpp — своя модель, без консоли"** block:
   choose the model in the list and press **"Запустить"** (Start).

What happens:

- the server comes up on `127.0.0.1:8080` (while the model loads it says
  "загружает модель…");
- the `llama.cpp` provider becomes active and the chosen model becomes the
  current one;
- you can ask right away.

**"Остановить"** (Stop) shuts the server down, **"Лог llama-server"** shows
why a model failed to start. The server is not tied to the app: restarting
CatLuBooks does not kill it.

Settings — `config.yaml` → `llamacpp`:

```yaml
llamacpp:
  binary: ""        # empty — we look next to the project and in PATH
  models_dir: "./models"
  port: 8080        # port of the server; the provider points there too
  args: ["-c", "8192"]   # llama-server flags, e.g. "-ngl", "99" for GPU
```

If port 8080 is taken by another program, the app says so; then pick another
port in `llamacpp.port` and restart the server.

## 12. Phone: PWA

The server listens on `0.0.0.0`, so it is reachable over the local network.

1. On the PC open **⚙ → "Сервер"** — there is the address for the phone, for
   example `http://192.168.4.76:8000`.
2. On the phone (same Wi-Fi) open that address.
3. In the browser menu choose "Install app" / "Add to Home screen". You get an
   icon, and the app opens without the address bar.

If the phone cannot connect:

- the phone and the PC must be on the same network (not a "guest" Wi-Fi);
- Windows may block Python in the firewall — allow private networks on the
  first run;
- the address may change when you reconnect to the network — check the new one
  in ⚙ → "Сервер".

## 13. Server: addresses, restart, stop

⚙ → **"Сервер"** (Server):

- **Address for the phone** — the link to open from the phone;
- **"Перезапустить"** (Restart) — after editing `config.yaml`, after `git pull`
  or after installing Tesseract. The page reloads by itself when the server is
  back;
- **"Остановить"** (Stop) — shut the server down for the night. You cannot
  start it again from the browser: that needs a command in the console.

A restart takes 15–30 seconds (the embedding model is loaded again) — that is
normal.

## 14. Question history

⚙ → **"История"** (History): every question and answer is saved automatically
(up to 200 entries). Click an entry to expand the full answer with its
sources. The **🗑** button removes one entry, **"Очистить историю"** removes
all of them.

The history lives in `agents/<agent>/history.json` — copy it into your backup
together with the agent.

## 15. Managing an agent

⚙ → **"База"** (Database):

- **"Переименовать"** (Rename) — a new name for the agent (the folder and the
  database stay the same);
- **"Очистить базу"** (Clear the database) — remove all fragments of the
  agent, the books on disk stay. The database is rebuilt at the next
  indexing;
- **"Удалить агента"** (Delete the agent) — wipes the folder: books,
  database, prompt, history. This cannot be undone.

In the agent list (⚙ → "Агенты") every agent has buttons: **✓** — make it
active, **✎** — rename, **🗑** — delete.

## 16. What lives where on disk

```
CatLuBooks/
├── config.yaml          # all settings (in git)
├── settings.json        # choices made in the interface: OCR, merge (not in git)
├── secrets.json         # provider API keys (not in git)
├── agents/              # agents (not in git)
│   └── <agent>/
│       ├── agent.json   # title, topic, prompt
│       ├── books/       # uploaded books
│       ├── db/          # database (ChromaDB)
│       └── history.json # question history
├── models/              # GGUF models for llama.cpp (not in git)
├── logs/                # server logs (not in git)
└── .venv/               # Python environment (not in git)
```

The `agents/` folder is your data. Everything else can be deleted and
downloaded again.

## 17. Backup and moving to another computer

To move everything to another computer, copy:

1. the `agents/` folder — agents, books, databases, prompts, history;
2. `secrets.json` — if you want to keep the keys (optional);
3. `settings.json` — the choices made in the interface (optional);
4. `config.yaml` — if you changed the settings (but check the paths).

On the new computer: install the dependencies, put the folders in place, run
`python app.py`. You do not need to reindex anything — the database is ready.

Details in the [README](../README.en.md#migration-to-another-computer).

## 18. `config.yaml` settings

The file is in the project root and is edited by hand only. After a change —
⚙ → "Сервер" → "Перезапустить".

| Section | What it configures |
|---|---|
| `prompt` | The default prompt, the task the model uses to write prompts, answer and merge templates |
| `store` | Agents folder, book formats, fragment size, OCR |
| `embeddings` | The embedding model (search) |
| `llm` | Provider, model, temperature, timeouts, token limits |
| `llamacpp` | Your own llama-server: path, GGUF folder, port, start flags |
| `search` | How many fragments to retrieve, context limit, search mode, merging |
| `server` | Address and port of the web server |

What people usually change:

- `llm.provider` and `llm.model` — the default model (easier in the
  interface, though);
- `search.n_results` and `search.max_context_chars` — how much context goes
  to the model. If the model complains "the context does not fit", lower
  `max_context_chars`;
- `llamacpp.args` — `-ngl 99` (GPU), `-c 16384` (long context).

## 19. When something does not work

**"The model returned an error 400" / "The server does not respond — check
whether it is running"**
The provider is not answering. Check: is Ollama running? Does `ollama list`
show the model? For the cloud — is there a key? Press "↻ Обновить список".

**"The context does not fit into the model"**
The model is small. Lower `search.max_context_chars` (say, to 6000) or take a
bigger model, or raise `-c` in `llamacpp.args`.

**There are answers, but not the right ones**
Check that the right agent is selected and rephrase the question with the
words from the books. If the topic is not in the database, the prompt will
not help — you need the files.

**A PDF finds nothing**
Most likely it is a scan. Turn OCR on (step 8) and click "Всё заново".

**Indexing takes forever**
Usually many pages, or scans with OCR. The progress shows in the "Книги" tab
("Читаю книгу 2 из 7: file.pdf" and a bar); wait for it and do not close the
app.

**Nothing changed after `git pull`**
Restart the server: ⚙ → "Сервер" → "Перезапустить". And press Ctrl+F5 in the
browser — otherwise you see the old interface from the cache.

**The phone cannot see the server**
See step 12: same network, firewall permission, fresh address from
⚙ → "Сервер".

**The interface looks broken (buttons do nothing, old look)**
Ctrl+F5 — hard reload.

## 20. Frequently asked questions

**Can I ask in another language?**
Yes, questions and answers work in any language. It is better to write the
prompt in Russian: the model behaves more predictably that way.

**Where did my uploaded books go?**
They are in `agents/<agent>/books/`. Check that the right agent is open —
each one has its own books.

**Can one agent cover two topics?**
It can, but better not: search mixes the topics and the answers get less
precise. Two agents are better than one.

**Do I have to reindex after changing the model?**
No. The database does not depend on the generation model: indexes and
embeddings are one thing, answers are another.

**What takes up space?**
The agent's database (`agents/<agent>/db/`), the uploaded books and the models
in `models/`. The size is shown in the agent list.

**How do I get the shared prompt back?**
Clear the "Промпт агента" field and press "Сохранить" — the agent will use
the prompt from `config.yaml` again.

**Can I work offline?**
Yes, after the dependencies and the embedding model are installed. A local
provider (Ollama or llama.cpp) works without the internet; a cloud one does
not.
