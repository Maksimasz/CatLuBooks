# CatLuBooks

[Русский](README.md) · **English**

A RAG assistant for your documents: it finds the answer in your books and files
and cites the source and page. The web interface works on a desktop and on a
phone (installable as a PWA).

> The interface itself is in Russian. UI labels are quoted below exactly as they
> appear on screen, so you can find them: ⚙ → Книги means the "Books" tab.

**Quick links:** [install](#install) ·
[migration to another computer](#migration-to-another-computer) ·
[building the database](#building-the-database) ·
[configuration](#configuration-configyaml)

## How it works

```
book (PDF, EPUB, TXT, FB2) ─► text ─► chunks ─► embeddings ─► database (ChromaDB)
                                                    │
question ─► search (vector + exact matches) ────────┤
                                                    ▼
                         context + question ─► LLM ─► answer with citations
```

- **Embeddings** are computed locally (the model is small, no internet needed).
- **Generation** goes through any OpenAI-compatible server: Ollama, LM Studio,
  llama.cpp, or a cloud provider (OpenAI, OpenRouter, Groq, DeepSeek).

## Features

- **Agents**: each one has its own folder, its own books and its own database,
  so different subjects never mix (for example "Документы" and "Заметки"
  side by side).
- Everything is managed from the interface (⚙ in the top right corner): create
  an agent, upload and delete books, start indexing, enter a key and choose a
  model. No console or file manager required.
- Books are indexed in batches without loading a whole book into memory; the
  process runs in the background and shows its progress.
- Scans: the server checks whether a PDF has a text layer and offers to enable
  OCR (requires Tesseract with the Russian language) right in the settings.
- Hybrid search: vector search plus a boost for exact word matches; neighbouring
  chunks and the page (or section) number are added to the answer.
- Merging: the same material found in several books is combined into one
  coherent answer (fast and accurate modes, switchable).
- Before answering, the model analyses what it found: what matters, whether
  there are contradictions or gaps — and only then replies.
- Streaming answer (the text is typed out as it is generated).
- Web chat + PWA: installs on a phone like an app.
- Server control from the admin panel (⚙ → Сервер): the LAN address for your
  phone, restart (after editing `config.yaml` or `git pull`) and stop. A stopped
  server can only be started again on the computer itself.
- Everything adjustable lives in a single `config.yaml` (theme, model, prompts,
  chunk sizes).

## Agents

```
agents/
└── dokumenty/          # agent: a folder with a Latin name
    ├── agent.json      # title and description for the interface
    ├── books/          # the agent's books (uploaded from the interface)
    └── db/             # this agent's database (ChromaDB)
```

An agent folder is self-contained: copy the folder and you have moved the agent
together with its books and database; delete it in the interface and the whole
agent is gone. The folder name is transliterated from the title ("Документы" →
`dokumenty`); if the title repeats, a number is appended (`dokumenty-2`).

The `agents/` folder is not tracked by git: it holds your books and databases.

## Project layout

```
CatLuBooks/
├── app.py              # FastAPI: API, chat and the agent admin panel
├── config.py           # loads config.yaml
├── config.yaml         # all project settings
├── core/
│   ├── database.py     # indexing books and searching
│   ├── parsers.py      # reads PDF, EPUB, TXT, FB2 (+ OCR of scans)
│   ├── ocr.py          # checks tesseract: version and languages
│   ├── agents.py       # agents: folders, books, names
│   ├── jobs.py         # background indexing progress for the interface
│   ├── llm.py          # generative model client
│   ├── providers.py    # provider registry and API keys
│   ├── models.py       # model list for the interface
│   ├── server.py       # network addresses, server restart and stop
│   └── store.py        # per-agent stores for the web layer
├── web/                # chat interface, settings, PWA
├── tools/drop.py       # removing data from the console
├── agents/             # agents: books and databases (not in git)
├── settings.json       # choices made in the interface, e.g. OCR (not in git)
├── secrets.json        # cloud provider keys (not in git)
├── tests/              # self-contained tests (no Ollama, no models)
├── requirements.txt
└── LICENSE
```

## Install

### What you need

| Item | Purpose | Where to get it |
|---|---|---|
| Python 3.10+ | runs the server | <https://www.python.org/downloads/> — or skip it: uv can download Python for you |
| uv (optional) | downloads Python and installs dependencies | <https://docs.astral.sh/uv/> — install command in the steps below |
| Embedding model | searching the books (locally, never sent to the cloud) | downloaded automatically on first launch — [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| Ollama and a generation model | answers on this computer, no cloud | [ollama.com/download](https://ollama.com/download), models — [search](https://ollama.com/search) |
| Cloud provider key | if answers are generated in the cloud | table "Model providers" below |
| Tesseract with the Russian language | reading scans, if you have any | [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) |

The internet is needed only once — to download dependencies and models. After
that everything runs locally.

### Steps

```bash
git clone https://github.com/Maksimasz/CatLuBooks.git
cd CatLuBooks
ollama pull yandex/YandexGPT-5-Lite-8B-instruct-GGUF   # generation model, or another one
```

Then pick whichever you prefer: **uv** (it downloads the right Python, installs
dependencies and runs the server; no environment activation needed) or plain
`venv` + `pip`.

**With uv** — [uv](https://docs.astral.sh/uv/) is installed with one line:

```bash
# if uv is not installed yet (Windows):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12             # Python is downloaded automatically if missing
uv pip install -r requirements.txt
uv run python app.py
```

Later launches are the same: `uv run python app.py` — uv finds `.venv` in the
project folder by itself. To update dependencies:
`uv pip install -r requirements.txt`.

**With venv and pip**:

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # about a gigabyte: torch and the rest
python app.py
```

Open <http://localhost:8000> — everything else is done in the interface (⚙).

What happens on the first launch:

- the embedding model (~120 MB) is downloaded into the user's HuggingFace cache —
  this is the only moment search needs the internet. The ready cache folder
  (`%USERPROFILE%\.cache\huggingface`) can be copied to another PC and used
  offline;
- if the model from `llm.model` is not present on this computer, no answers will
  be generated: run `ollama pull <model>` (or pick another one in ⚙ → Модели);
- to check that everything is in place:
  `python -m unittest discover -s tests -t .` — the tests are self-contained and
  need neither models nor Ollama.

### Updating

```bash
git pull        # then restart the server: ⚙ → Сервер → Перезапустить,
                # or press Ctrl+C and run python app.py
```

Updating does not touch your data: books and databases live in `agents/`, keys in
`secrets.json`, interface settings in `settings.json`. If you have edited
`config.yaml`, the update may conflict — keep a copy of your version beforehand.

## Migration to another computer

Moving is just copying a few folders. You do not need to reinstall dependencies
or re-index books if you move the database as well.

### What to copy

| Folder or file | What it is | Needed? |
|---|---|---|
| `agents/` | books and ready-made agent databases — the most valuable part | yes, if you want your books |
| `secrets.json` | cloud provider keys (Groq, OpenRouter…) | if you use the cloud |
| `settings.json` | interface choices: OCR, merge mode | no, it can be set up again |
| `config.yaml` | your edited settings | if you changed them |
| HuggingFace cache | the embedding model (~120 MB) | no, it is downloaded automatically |

`agents/` grows with your library: an agent with a few dozen books can take
gigabytes. These files are not tracked by git, so they must be copied manually —
a USB drive, the local network or an archive.

### Step by step

1. On the new computer, follow the "Install" section.
2. Copy `agents/` (and, if needed, `secrets.json`, `settings.json`,
   `config.yaml`) into the project root — next to `app.py`.
3. Start the server: `python app.py` (or `uv run python app.py`).
4. For local answers, download the model: `ollama pull <model>` — see
   [Choosing a model for weak hardware](#choosing-a-model-for-weak-hardware) to
   pick one that fits the new machine.
5. In the browser, choose the provider and model: ⚙ → Модели. This choice is
   stored in the browser, not in the project files, so it has to be made again
   on the new computer. A cloud key is best entered there too, with the
   "🔑 Ввести API-ключ" button — then copying `secrets.json` is optional.
6. Open the agent: books, chunks and search history are already in place.

There is no need to press «Индексировать» after moving — the database is in
`agents/<agent>/db` and ready to search.

### If you are moving only the books

Copy `agents/<agent>/books` (or create an agent and upload the books through the
interface) and press ⚙ → Книги → «Индексировать». This takes a while: indexing
dozens of books takes hours. If some books are scans, the new machine will also
need Tesseract with the Russian language.

Re-indexing after copying will read the books again: their modification time
changes, and indexing uses it to decide whether a book needs to be re-read. This
does not damage a ready database — just do not start indexing if the database is
already there.

### What not to change

The embedding model (`embeddings.model` in `config.yaml`) must stay the same as
the one the database was built with: vectors from different models are not
comparable, and search will start returning random chunks. If you did change it,
rebuild the database completely: ⚙ → Книги → «Всё заново».

Paths inside the database are relative (`instructions/guide.pdf`), so moving does
not depend on where the project is located and works between Windows and Linux.

## Building the database

Agents and books are created right in the interface — open
<http://localhost:8000> and click ⚙ in the top right corner:

1. **Агенты** — create an agent (for example "Документы"). A folder
   `agents/dokumenty/` appears with its own `books/` and `db/`.
2. **Книги** — upload files (PDF, EPUB, TXT/MD, FB2; subfolders inside `books/`
   are read too) and press **Индексировать**. Progress is shown in place: which
   book is being read and how many chunks have been saved. Indexing runs in the
   background — you can close the settings and keep chatting.
3. **Модели** — choose a provider and a model, and enter a key if needed.

A book can be deleted in the same window: the file and its chunks are removed
together. Re-indexing updates books that are already there instead of
duplicating them.

Books that have already been read are skipped when you press «Индексировать»
again: only new and changed books are read. So adding one book takes minutes
instead of hours: the list shows "already in the database, no need to read
again". To read everything from scratch (for example, to re-run OCR on scans),
use the **Всё заново** button — it asks for confirmation first.

If a file was removed from the folder outside the interface, its chunks stay in
the database. Such a row is shown in the book list marked "нет файла" (no file),
and it can be removed with the same 🗑 — that deletes the remaining chunks.

Indexing can also be started from the console (for example, when the database is
built on a server without a GUI):

```bash
python -m core.database              # index all agents
python -m core.database dokumenty    # index a single agent
```

PDF pages are read through their text layer; EPUB, TXT/MD and FB2 are textual by
nature. Scans (image-only pages) are recognised with Tesseract — see below.
`.djvu` and `.mobi` files are not supported: a converter is required.

### Scans and OCR

If a book is a set of page photos, it has no text and there will be nothing to
search. The server checks books by itself: such a PDF is marked "скан" (scan) in
⚙ → Книги, and a hint below the switch suggests enabling OCR.

Recognition is done by [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki),
a separate free program. The order is:

1. Install Tesseract for Windows (the UB-Mannheim installer). During setup, open
   "Additional language data" and tick **Russian** — without it Russian text is
   not recognised.
2. Open ⚙ → Книги and press «Проверить снова» (check again): the server finds the
   program itself (it looks both in PATH and in `C:\Program Files\Tesseract-OCR`),
   no restart needed.
3. Enable «Распознавать сканы» (recognise scans) and press «Индексировать».

Worth knowing:

- OCR is only needed for PDF scans; EPUB/FB2/TXT already contain text.
- Speed is about 1–2 seconds per page: a 300-page book takes 5–10 minutes. It runs
  in the background, so you can close the settings.
- Quality depends on the scan: clean text is recognised almost without errors,
  two-column tables and small print are worse.
- The setting is stored locally in `settings.json` (not in git) and takes
  precedence over `store.ocr` in `config.yaml`.
- Books that were already in the database without OCR are re-read automatically:
  after enabling recognition press «Индексировать» — the server notices that a
  scan was read without OCR and reads only that book, leaving normal books alone.

### Removing data from the database

This is normally done in the interface (⚙ → Книги, ⚙ → База). From the console:

```bash
python tools/drop.py --list                   # agents, books and chunks
python tools/drop.py dokumenty                # what is inside one agent
python tools/drop.py dokumenty "Guide.pdf"    # remove a book together with its chunks
python tools/drop.py dokumenty --clear        # clear the agent's database
python tools/drop.py dokumenty --delete       # delete the whole agent
```

Stop the server first (`Ctrl+C`): it keeps the database open. Reading (`--list`)
works while the server is running.

From the interface you do not need to stop the server: deletion in ⚙ → Книги (🗑)
and clearing the database work on the fly — the server closes the database
itself. Stopping (`Ctrl+C` or ⚙ → Сервер → Остановить) is only needed for the
console utility `tools/drop.py` and to shut the server down completely. The
"нет файла" row means the chunks of a book whose file was deleted outside the
interface: 🗑 on that row removes only the chunks from the database.

Deleting a book (🗑) comes in two flavours, so knowledge is not lost by mistake:
- 📄 **«Удалить только файл»** (delete the file only) — the file is removed from
  disk, but the chunks stay in the database: the knowledge is still searchable,
  the source just can no longer be opened. Handy for freeing space taken by large
  PDFs without losing what was in them.
- 🗑 **«Удалить книгу и базу»** (delete the book and its database entries) — the
  file and the chunks are removed together.

### How indexing works

Text is extracted from the file in parts (a PDF page, an EPUB chapter, a piece of
a TXT file), then running headers, soft hyphens and broken words are removed.
Numbers are kept: only a lone number at the very top or bottom line of a page is
treated as a page number. Tables of contents, back-of-book indexes, pages without
text and chunks with broken glyphs are not written to the database: they are of
no use, while a dish name in an index occurs dozens of times next to page numbers
and used to outrank the recipe itself.

A book counts as already read if its chunks are in the database and the file has
not changed since (size and modification time match). Change the file and it is
read again; a scan read without OCR is re-read once OCR is enabled. All of this is
recorded inside the book's chunks rather than in a separate list, so moving the
`agents/` folder to another computer loses nothing.

Service text is recognised by its features, not by specific books:

| Feature | What it finds |
|---|---|
| dot leaders ("Борщ . . . . . 27") | table of contents |
| "name + page number" in a row ("177. Борщ", "Клафути 118") | back-of-book index |
| a list of names without numbers: almost no commas or periods, many capitalised words, no preparation verbs | a table of contents in books where numbers were not recognised |
| broken glyphs (`/g299/g301`) | OCR garbage |

Ingredient lines ("мука 200 г") do not fall under the last feature: they start
with a lowercase letter. Neither do technological cards: their quantities are
printed on separate lines ("Картофель", "80", "60"), and these lines used to be
dropped along with page numbers, so the recipe reached the model without
quantities. Such pages are now recognised by their "брутто/нетто" (gross/net)
columns, the recipe number and the "Расход сырья" (ingredient consumption) line,
and are kept in the database in full.

## Running

```bash
python app.py
```

- On the desktop: <http://localhost:8000>
- From a phone (same Wi-Fi): `http://<PC-IP>:8000` → "Add to Home Screen".
- Settings: ⚙ in the top right corner (agents, books, models, database).
- Your IP: `ipconfig` (Windows). On the first launch, allow access in the
  firewall for **private** networks.
- The address for your phone, restart and stop are in the same place, on the
  "Сервер" tab (⚙ → Сервер): it detects the computer's LAN IP by itself.

## Configuration (`config.yaml`)

| Block | What it sets |
|---|---|
| `prompt` | the assistant's subject and behaviour (system prompt and templates) |
| `store` | the agents folder, book formats, chunk sizes, OCR |
| `embeddings` | the local embedding model |
| `llm` | provider, model, generation parameters |
| `search` | how many chunks to retrieve, context limit, history, search mode, source merging |
| `server` | address and port |

**This computer only, no network access** — `server.host: 127.0.0.1`.

**Changing the subject** — the `prompt` block only. **Changing the model** —
`llm.model`. **Changing the provider** — `llm.provider`.

## Model providers

Generation works through any OpenAI-compatible server. Ready-made providers are
built in (choose them in ⚙ → Модели, enter the key there as well):

| Provider | Key | Note |
|---|---|---|
| `ollama` | not needed | local, the default; models — [ollama.com/search](https://ollama.com/search) |
| `groq` | `GROQ_API_KEY` | free, very fast |
| `gemini` | `GEMINI_API_KEY` | Google's free tier |
| `openrouter` | `OPENROUTER_API_KEY` | free models available |
| `deepseek` | `DEEPSEEK_API_KEY` | cheap API |
| `mistral` | `MISTRAL_API_KEY` | free tier available |
| `openai` | `OPENAI_API_KEY` | paid |

A key can be provided in two ways:

1. **In the interface** — ⚙ → Модели, the "Ввести API-ключ" button: choose a
   provider and paste the key. It is saved to the local `secrets.json` file
   (in `.gitignore`).
2. **As an environment variable** — for example `set GROQ_API_KEY=...` (Windows).

### Model list

The model list is fetched from the provider itself (`/models`), so it is always
up to date: providers add new models and retire old ones over time. The model
field shows the models of the selected provider, and below them free models from
other providers (marked "бесплатно" — free). Selecting an entry like
`[groq] openai/gpt-oss-120b` switches both the provider and the model.

Free models are detected from the price in the provider's response (OpenRouter),
the `:free` suffix, a provider flag (all of Groq's API is free) and the model name
(for example, `flash` at Gemini). Service models — speech synthesis, audio
transcription, embeddings — are excluded from the list. The list is cached for 5
minutes and refreshed when a key is entered.

Any server of your own will work too: put the full URL of an OpenAI-compatible
endpoint into `llm.provider` (`http://192.168.1.5:1234/v1`).

### Choosing a model for weak hardware

If the GPU is weak (little VRAM), 8B models run on the CPU and answer slowly.
Options:

- light local models: `qwen2.5:3b`, `gemma2:2b`, `llama3.2:3b`;
- or generate in the cloud (Groq, Gemini, OpenRouter) — search and embeddings
  stay local, only the retrieved context is sent to the cloud;
- reduce `search.n_results` and `search.max_context_chars`; `max_tokens` can be
  lowered too — answers get shorter, and if the model runs out of budget the
  server asks for a continuation;
- `search.lexical` (default `true`) gives priority to exact word matches; search
  goes through the Chroma index and is therefore fast. Turning it off speeds
  things up a little more but loses exact matches.

## Search quality

Retrieved chunks are sent to the model whole: a chunk cut in half is useless, so
the `search.max_context_chars` limit drops entire chunks instead of truncating
text in the middle.

The answer length is limited by `llm.max_tokens` (4096 by default): a detailed
answer in Russian takes about 2600 tokens, so at 2000 it used to stop mid-word.
If the model still hits the limit, the server notices it via `finish_reason` and
asks for a continuation itself — up to `llm.max_continuations` times (2 by
default). The continuation is visible in the chat as "⏳ Ответ длинный —
дописываю продолжение…".

## Free-tier limits

Free cloud models limit not only the speed but also the volume: for Groq it is
**8000 tokens per minute**, for example. One question with merging (a book
context of ~6000 tokens plus the answer) uses up almost the whole minute, so a
second question in a row may get "Rate limit reached". That is a provider limit,
not a server error: wait a minute (the message says how long) or pick a model
with a larger limit.

If you hit the limit constantly, reduce `search.merge_context_chars` (how much
text is sent to merging) or turn merging off.

### Merging answers from several books

Usually the same material is described in several books, with different details.
In the normal mode the model sees only a few of the best chunks, so it may produce
a fragment or general words instead of a complete answer.

It is enabled in ⚙ → Книги → «Сводить ответы из разных книг» (or via
`search.merge_mode` in `config.yaml`):

| Mode | What happens | When to enable |
|---|---|---|
| **Off** | as before: whatever was found is in the answer | by default, the fastest |
| **Fast** | search takes the best chunk from each of several books, and the model merges them into one coherent answer | when the answer is spread across different books |
| **Accurate** | two passes: first a draft merging all variants, then the answer based on it | when you need a complete answer and waiting is fine |

The "Accurate" mode answers roughly twice as long: two model calls instead of
one. How many books to query — `search.merge_sources` (6 by default), how much
text to send to merging — `search.merge_context_chars`.

A local model on the CPU needs time: a normal answer fits into `llm.timeout`
(300 seconds), while merging needs `llm.merge_timeout` (900). If instead of an
answer you get "the model did not respond in the allotted time", raise that
number in `config.yaml` or reduce `search.merge_context_chars`.

Merging prompts live in `config.yaml` (`prompt.merge_template`,
`prompt.synthesis_template`, `prompt.answer_from_draft_template`) — you can
rewrite them for your own subject.

Tables of contents, back-of-book indexes and OCR garbage no longer reach the
database at all — `core/parsers.py` filters them out during indexing. Just in
case, the same rules apply at search time: if old chunks are still in the
database, they are pushed to the end of the results and only make it into the
answer when nothing better was found.

A recipe table ("брутто/нетто" columns, "Расход сырья") is not treated as service
text, neither during indexing nor during search: its numbers are quantities, and
this is exactly the chunk that answers "how much to put in". Matching on all the
words of the question is rewarded separately: for "московский борщ" the
technological card that contains both "борщ" and "московский" comes before a
chef's biography that contains only "московский", even if the biography ranked
higher in the vector list.

If a book was recognised poorly, two steps help: enable `store.ocr: auto`
(recognises image-only pages with tesseract) and re-index the database from
scratch. Search alone will not fix broken text.

## Tests

```bash
python -m unittest discover -s tests -t .     # one file: python -m unittest tests.test_jobs
```

The tests are self-contained: they need no Ollama, no models and no working
database.

## Based on

The project grew out of the idea and code of
[kpshinnik/rag_ollama](https://github.com/kpshinnik/rag_ollama) (MIT). The
indexing and search core is reused; the rest has been reworked.

## License

MIT — see `LICENSE`.
