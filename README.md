# Hakim AI System

A local AI agent built on llama.cpp. Everything runs on your machine by
default: no API keys, no network calls, nothing leaves the box.

Hosted models are the one exception, and an opt-in one. Selecting Gemini or
Cerebras trades that guarantee for speed — a turn that costs minutes here costs
seconds there — and the UI says so wherever it is true rather than claiming
otherwise. See [section 7](#7-models-and-switching).

Two front ends over one engine: a React web app talking to a FastAPI layer, and
a terminal CLI. Both use the same agent, the same tools and the same model
manager — the API adds transport, not a second implementation.

---

## Contents

1. [What it does](#1-what-it-does)
2. [Hardware reality](#2-hardware-reality)
2a. [Why a turn takes as long as it does](#2a-why-a-turn-takes-as-long-as-it-does)
3. [Setup](#3-setup)
4. [Running it](#4-running-it)
5. [Project layout](#5-project-layout)
6. [Architecture](#6-architecture)
7. [Models and switching](#7-models-and-switching)
7a. [Adding your own models](#7a-adding-your-own-models)
8. [Auto-routing](#8-auto-routing)
9. [Tools](#9-tools)
10. [Chat history](#10-chat-history)
11. [The web UI](#11-the-web-ui)
12. [Commands](#12-commands)
13. [Configuration](#13-configuration)
14. [Tests](#14-tests)
15. [Security posture](#15-security-posture)
16. [What is verified and what is not](#16-what-is-verified-and-what-is-not)
17. [Troubleshooting](#17-troubleshooting)
18. [Roadmap](#18-roadmap)

---

## 1. What it does

- Chats with a local model served by `llama-server`
- Calls tools when they help: exact arithmetic, reading files in a workspace
- Finds any model you drop in `weights/`, sizing it from its own GGUF header
- Holds only one model in RAM at a time
- Optionally picks the model itself based on how hard the prompt looks
- Streams tokens, and the model's reasoning, as they generate
- Shows what every tool call sent and received, so you can check its work
- Reads images you drop in, via OCR
- Searches your own notes and PDFs by meaning, and answers from them
- Remembers your preferences and decisions across conversations, and
  retrieves only the ones relevant to what you just asked
- Can send a turn to a hosted model instead, when speed matters more than
  keeping it local — and asks first when the router chooses that itself
- Saves every conversation to a local SQLite file

Deliberately **not** used: LangChain, LangGraph, AutoGen, CrewAI. The agent loop
is about eighty lines you can read.

---

## 2. Hardware reality

This matters more than anything else in the guide.

**Machine:** Intel i5-6300U (2 cores / 4 threads, 2015 ultrabook), 8 GB RAM, no GPU.

**Measured on this machine, from the llama-server logs:**

| Model | Load time | Prompt eval | Generation |
|---|---|---|---|
| Qwen3 8B Q4_K_M | ~130 s | 3.7–5.5 tok/s | 0.23–0.49 tok/s |

A calculator turn end-to-end took **285.7 s**: 252 s for the tool-call round
(714 prompt tokens + 30 generated), then 32 s for the answer — the second round
is fast because llama.cpp's prefix cache only had to process 69 new tokens.

**Consequences that shape the whole design:**

- The 8B model is 4.7 GB on an 8 GB machine. It cannot stay fully resident, so
  it pages from disk. This is the main reason generation is slow.
- One model at a time. Two would thrash.
- Switching models costs ~130 s to load, plus the new process starts with a cold
  prefix cache, so the conversation is re-processed — roughly **5 minutes total**.
- Streaming is not a nicety. Without it you stare at a blank screen for minutes.
- Thinking mode roughly doubles the wait. Off by default in practice.

Use `tiny` or `fast` for day-to-day work. Reach for `reasoning` deliberately.

---

## 2a. Why a turn takes as long as it does

Measured on this machine (i5-6300U, 2 physical cores, 8 GB) with
`Qwen3.5-2B-XS-Q3_K_S`, using llama.cpp's own `llama-bench` and the `timings`
block llama-server returns. Worth reading before trying to optimise anything,
because the obvious suspects turned out not to be the problem.

### Generation is ~2 tok/s, and threads do not change that

```
threads   prompt (pp128)      generation (tg32)
   1       7.81 ± 0.57          1.58 ± 0.14
   2      13.57 ± 1.90          1.88 ± 0.49
   3      12.66 ± 3.77          2.15 ± 0.19
   4      16.70 ± 3.69          2.05 ± 0.66
```

The generation figures overlap inside their error bars. Two threads, three or
four make no difference to how fast tokens come out — the machine is bound by
memory bandwidth, not by cores. `-t 4` is kept because it is the best of them
for prompt processing, which is the part that *is* compute-bound.

> An earlier run of this suggested `-t 3` was 81% faster. It was noise: the
> dev servers were running during part of it. The numbers above come from
> `llama-bench` with repetitions and standard deviations, on an idle machine.
> If you re-measure, do it that way — single timed requests on this hardware
> vary by more than the effect being looked for.

### The prefix is what makes the agent slower than raw llama-server

Same server, same model, four shapes of request:

| request | prompt tokens | prompt time | total |
|---|---|---|---|
| a bare question | 22 | 2.1 s | 19.2 s |
| + the system prompt | 339 | 30.5 s | 46.3 s |
| + tool definitions (**a real turn**) | **906** | **62.4 s** | 77.1 s |
| the same request again | 4 | 0.6 s | 17.0 s |

Prompt processing runs at about 14.5 tok/s. A turn carries 906 tokens of
prefix before the user's question, so the first turn of a conversation spends
**a minute** on prompt evaluation. That is the whole difference between this
and typing into llama-server's own UI — not tokens per second, which is
identical, but how much prompt there is.

**The cache works.** The fourth row is the same request repeated: 4 tokens
processed instead of 906, 61.8 s saved. So the prefix is paid once per
conversation, not once per turn.

`--cache-reuse` is deliberately **not** passed. The theory was that replaying
history without tool calls would diverge from the cached tokens and force a
reprocess; measured, turn two reprocesses 50 tokens of 982 either way, because
the divergence is near the end of the prompt where little is left to redo.

### Tools are the prefix

| enabled | tools | prompt tokens | first turn |
|---|---|---|---|
| default | 3 | 894 | ~62 s |
| + documents | 5 | 1,167 | ~80 s |
| + memory | 8 | 1,740 | ~120 s |
| everything | 21 | 3,646 | ~251 s |

Every tool switched on is prompt tokens paid on the first turn of every
conversation. Turning on all of them costs **four minutes** before the model
writes anything. This is the reason tool descriptions here are terse and the
reason tools are off by default — it is not caution about capability, it is
that each one has a measurable price.

### So, to make it faster

1. **Turn off tools you are not using.** Biggest single lever, and it is a
   switch in the sidebar.
2. **Keep the conversation going** rather than starting a new one. The prefix
   is cached; a new conversation pays it again.
3. **Turn off Thinking** unless the question needs it. It does not change
   tok/s, it changes how many tokens are generated — hundreds of them, at
   2 tok/s.
4. **Mind the idle timeout.** `idle_timeout_seconds` in `models.json` is 300.
   When the model unloads, its KV cache goes with it, so the next message pays
   both the reload and the full prefix again. Raise it if you have RAM to
   spare and think in long pauses.
5. **Use a smaller model.** At 2 tok/s the model is the ceiling, and the 8B is
   far slower still.

---

## 3. Setup

### Prerequisites

- Python 3.11+
- Node 20+ for the front end (verified on 26.7.0)
- `llama-server.exe` from llama.cpp — this project was verified against
  **build 10373**
- GGUF model files

### Install

```bash
cd "C:\Users\SHAMI\HAKIM\AI\Hakim Local Agent"
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Python dependencies are only `requests`, `fastapi` and `uvicorn`. Everything
else is standard library.

The front end needs Node (verified on 26.7.0):

```bash
npm --prefix web install
```

### Point it at your files

Edit [`models.json`](models.json):

```json
{
  "server_exe": "../LLAMA CP/llama-server.exe",
  "models_dir": "weights"
}
```

No model paths are hardcoded anywhere in the Python.

**Paths here are relative to `models.json` itself**, not to the working
directory — so the project folder can be renamed or moved and nothing needs
editing. That is not a style choice: these were absolute, the folder was
renamed, and the hardcoded path was the one thing that broke. An absolute path
still works if the weights live on another drive.

---

## 4. Running it

### Web UI

Two processes: the API, and the front end that talks to it.

```bash
cd "C:\Users\SHAMI\HAKIM\AI\Hakim Local Agent" && .venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```bash
cd "C:\Users\SHAMI\HAKIM\AI\Hakim Local Agent" && npm --prefix web run dev
```

Opens on <http://127.0.0.1:5173>. Vite proxies `/api` to the API, so the
browser sees one origin. Pick a model in the sidebar, click **Load**, then
chat — or just send a message and the model starts on demand.

Both bind to **127.0.0.1 on purpose**. With the tool flags on, this API can
write files and run commands; that is fine as a local tool and unacceptable on
a network interface. See [section 15](#15-security-posture).

Two flags that matter:

- **`--workers 1`** is the only supported setting, and is the default. The
  model manager owns the `llama-server.exe` child processes, so a second
  worker would mean a second manager fighting it for the same ports.
- **Do not use `--reload`.** The reloader kills the worker in a way that does
  not reliably reach the shutdown handler, and every restart then leaks a
  `llama-server` holding gigabytes.

For a single process serving both, build the front end first — FastAPI then
serves it from the same origin and Vite is not involved:

```bash
npm --prefix web run build
```

### Terminal

```bash
cd "C:\Users\SHAMI\HAKIM\AI\Hakim Local Agent" && .venv\Scripts\python main.py
```

### Starting a model server by hand

You never have to — the manager does it — but if you want to:

```bash
"C:\Users\SHAMI\HAKIM\AI\LLAMA CP\llama-server.exe" -m "C:\Users\SHAMI\HAKIM\AI\Hakim Local Agent\weights\Ministral-3-3B-Instruct-2512-Q4_K_M.gguf" --jinja -c 4096 -t 4 -np 1 --port 8084
```

If a healthy server is already on a model's port, the manager **adopts** it
rather than starting a rival.

> **Note on `-t 4`:** your CPU has 4 logical processors. The original
> `runner.bat` passed `-t 8`, which oversubscribes and slows things down.

---

## 5. Project layout

```
Hakim Local Agent/
├── main.py              terminal CLI
├── config.py            all settings, environment-driven
├── models.json          model registry (paths, ports, RAM thresholds)
├── chat_store.py        SQLite conversation history
├── requirements.txt
├── .gitignore           keeps weights/ and data/ out of any repo
│
├── weights/             the GGUF model files (~9.5 GB, git-ignored)
├── data/                generated state: the SQLite database, and rag/ index
├── uploads/             images attached in the UI (git-ignored)
├── samples/             example images for the OCR tool
├── .env                 API keys for hosted models (git-ignored)
│
├── api/                 the HTTP layer the front end talks to
│   ├── main.py          app, lifespan, static serving
│   ├── runtime.py       process-wide objects; runs one turn
│   ├── turns.py         the queue: one turn at a time, with positions
│   ├── schemas.py       request and response bodies
│   └── routes/          chat (SSE), conversations, models, meta, uploads,
│                        rag, memory
│
├── web/                 React + TypeScript + Vite + Tailwind
│   ├── src/lib/         api client, SSE reader, markdown, commands
│   ├── src/hooks/       the turn state machine and data hooks
│   ├── src/components/  sidebar, transcript, composer, palette
│   └── dist/            build output (git-ignored)
│
├── agent/
│   ├── loop.py          the reasoning/action cycle
│   ├── parser.py        assistant message → tool calls
│   ├── prompts.py       system prompt
│   └── router.py        picks a model from the prompt
│
├── models/
│   ├── gguf.py          reads a GGUF header without loading the model
│   ├── discovery.py     turns a folder of .gguf files into model entries
│   ├── preferences.py   your choices, in data/models.local.json
│   ├── qwen.py          llama-server client (the only module that speaks HTTP)
│   ├── remote.py        hosted providers (the only module that leaves the box)
│   ├── connectivity.py  cached "is there internet"
│   └── manager.py       starts/stops/switches llama-server processes
│
├── memory/              persistent memory (disabled by default)
│   ├── types.py         memory kinds, statuses, decay half-lives
│   ├── store.py         memories, links, jobs, summaries (SQLite)
│   ├── vectors.py       memory embeddings, reusing the document index
│   ├── retrieval.py     scoring and decay - pure arithmetic
│   ├── extraction.py    what is worth remembering, without a model
│   ├── consolidation.py duplicates and contradictions
│   ├── processor.py     the one place a model is switched
│   ├── context.py       assembling a turn under a token budget
│   └── manager.py       the object everything else talks to
│
├── rag/                 document search (disabled by default)
│   ├── __main__.py      `python -m rag` - index without a model server
│   ├── extractor.py     file to text, page by page for PDFs
│   ├── chunker.py       overlapping chunks on paragraph boundaries
│   ├── embeddings.py    owns the worker process; starts late, stops early
│   ├── worker.py        BGE in its own process, so its RAM comes back
│   ├── index.py         flat float32 vectors, searched with numpy
│   ├── metadata.py      chunk text and documents (SQLite)
│   └── manager.py       the order everything happens in
│
├── tools/
│   ├── base.py          Tool, ToolRegistry, argument validation
│   ├── registry.py      builds the default tool set
│   ├── calculator.py    safe expression evaluator
│   ├── filesystem.py    workspace-jailed list + read
│   ├── python_tool.py   restricted Python (disabled by default)
│   ├── shell_tool.py    allowlisted terminal commands (disabled by default)
│   ├── http_tool.py     allowlisted HTTP requests (disabled by default)
│   ├── git_tool.py      structured git, commits behind a second flag
│   ├── memory_tool.py   the five memory tools
│   ├── ocr_tool.py      OCR, dispatching to one of two backends
│   ├── tesseract.py     the Tesseract backend
│   ├── document_search.py  semantic search over indexed files
│   └── web.py           placeholder
│
└── tests/               418 tests, no server required
```

---

## 6. Architecture

```
Browser            (web/ — React, talks only to /api)
 → FastAPI         (api/routes/chat.py — one POST, one SSE stream)
 → Turn queue      (api/turns.py — one turn at a time, the rest wait)
 → Agent           (agent/loop.py)
 → Qwen            (models/qwen.py → llama-server)
 → tool call
 → Tool Registry   (tools/base.py)
 → tool execution
 → tool result
 → Qwen
 → final response  (streamed back out as events the whole way)
```

The CLI enters at **Agent** and skips the first three rows entirely.

`Agent.send()` loops up to `max_iterations`:

1. Send the conversation plus tool definitions to the model.
2. If the reply has no `tool_calls`, that is the final answer — return it.
3. Otherwise run each call through the registry, append each result as a `tool`
   message, and go round again.
4. Falling out of the loop raises `IterationLimitError`.

**The registry is the trust boundary.** Unknown tool, missing argument, wrong
type, or a tool throwing — all come back as `ToolResult(ok=False)` whose text
goes to the model so it can correct itself. Only genuinely broken server output
becomes an error the user sees.

**Reasoning is displayed but never replayed.** llama.cpp returns thinking in
`reasoning_content`, separate from `content`. It streams to the UI on its own
callback, and the agent still stores only `content` and `tool_calls` — so it is
never sent back to the model and never written to the database. Showing it and
feeding it back are different questions; only the first is the caller's to
decide. See [section 11](#11-the-web-ui).

**History trimming** only cuts immediately before a `user` message, so a tool
result is never orphaned from the call it belongs to — which the chat template
would reject.

### Verified llama.cpp behaviour (build 10373)

- `--jinja` is **on by default**. The server applies Qwen3's chat template,
  parses the model's native tool syntax, and returns standard OpenAI
  `tool_calls`. This is why `parser.py` invents no protocol — it reads what the
  server already produces.
- Reasoning goes to `message.reasoning_content` under the default
  `--reasoning-format deepseek`.
- `chat_template_kwargs` is supported, which is how `enable_thinking` is sent.
- `--alias` is unset, so the server ignores the `model` field in requests.

---

## 7. Models and switching

Defined in [`models.json`](models.json):

| Key | Model | File size | Port | Min free RAM |
|---|---|---|---|---|
| `mistral` | Ministral 3B Q4_K_M | 2047 MB | 8084 | 1900 MB |
| `fast` | Qwen3.5 2B (M) Q4_K_M | 1023 MB | 8080 | 1150 MB |
| `tiny` | Qwen3.5 2B (XS) Q3_K_S | 704 MB | 8083 | 900 MB |
| `reasoning` | Qwen3 8B Q4_K_M | 4794 MB | 8082 | 6200 MB |

**`mistral` is the default and the router's "fast" model.** Ministral 3B is
Mistral AI's edge model, not Mistral 7B. Measured: loads in **46 s**, and
answered a tool-call round in **35 s** against 252 s for the Qwen 8B. Its chat
template carries Mistral's native function-calling format (`AVAILABLE_TOOLS`,
`TOOL_CALLS`) and llama.cpp parses it into standard OpenAI `tool_calls`.

Its native context is **262144** with YARN scaling; it is capped at 4096 here
because the KV cache for anything larger will not fit on this machine.

Port 8081 is deliberately skipped — it is reserved for GLM-OCR.

> `tiny` is **Q3_K_S**, a harsher quantisation than the others. It is the
> quickest but the weakest at following instructions.

### How the manager behaves

- **One model at a time** (`max_active: 1`). Switching stops the other first.
- **Adopts** a healthy server already on a port instead of starting a rival.
  This keeps a CLI and a browser session from fighting, and means a
  hand-started server still works.
- **Reclaims its own ports.** If a model's port is held by a `llama-server`
  this manager did not start, switching stops it. Refusing used to deadlock the
  common case: restart the UI and every server the old process started looks
  foreign, so switching became impossible until you killed things by hand. The
  identity check is what keeps this safe — a port held by anything that is not
  a llama-server is left alone, and the switch fails with that reason.
- **Checks RAM before starting**, via `GlobalMemoryStatusEx` through `ctypes`
  — no `psutil` dependency. It accounts for mmap: see below.
- **Waits for `/health`** before reporting ready. Process started ≠ model loaded.
- **Detects crashes** — a server that dies on its own shows as `FAILED` with its
  exit code, and can be restarted.
- **Unloads after idle** (`idle_timeout_seconds`, default 300) to give RAM back.

States: `stopped → starting → ready → stopping → stopped`, plus `failed`.

#### The RAM guard, and why it is not a size check

The obvious guard — "refuse unless free RAM exceeds the model size" — is wrong
for llama.cpp, and measurement on this machine showed why:

| Observation | Figure |
|---|---|
| Available RAM before starting Qwen3.5 2B XS (704 MB file) | 503 MB |
| Time to load | 9 s |
| Available RAM immediately after | 504 MB |
| Private (committed) bytes of that process after a real turn | 738 MB |
| The agent turn it then completed | 89 s, correct answer |

llama.cpp maps the weights, so they are file-backed pages rather than committed
memory, and they fault in lazily. Committed memory climbs towards the model's
size only as inference touches the weights. The old guard measured the right
quantity at the wrong moment and refused a load that works.

So the guard now distinguishes two cases:

- **Below `HARD_FLOOR_MB` (250 MB): refuse.** There is not enough room for the
  process itself, whatever the model.
- **Below the model's own `min_free_mb`: start, and warn.** The pages that
  cannot stay resident get re-read from disk on every token, so this predicts
  *slow*, not *broken*. The warning appears in the sidebar and after `/model`.

`min_free_mb` therefore means "runs comfortably above this", not "refuses below
this". The figures are derived from measurement — GLM-OCR's 906 MB file holds
683 MB resident, a ratio near 0.75 — except `reasoning`, which is left
deliberately high because a 4.8 GB model on an 8 GB machine genuinely thrashes.


### Hosted models

Two are registered alongside the local ones: **Gemini** and **GPT-OSS 120B on
Cerebras**. Both speak the OpenAI chat-completions shape, so one client covers
them and the agent loop is identical — same tools, same history, same events.

They are for speed. Measured on the same prompt, same tool, through the whole
stack: **127.1 s** on local Ministral, **3.7 s** on Gemini 3.5 Flash.

What is different about them:

- **They leave the machine.** The prompt, the conversation history and every
  tool result go to the provider — which, with filesystem reading on, includes
  the contents of files the agent reads.
- **They hold no RAM,** so they are deliberately outside the one-at-a-time
  rotation. Selecting one leaves your local model resident, and switching back
  costs nothing instead of another cold load.
- **Nothing to load.** No file, no port, no RAM threshold, so no Load button.
  They are usable when a key exists *and* there is internet, and the sidebar
  distinguishes those two failures because they have different fixes.
- **The auto-router asks first.** Choosing one yourself is already deliberate;
  being moved onto one is not. See [section 8](#8-auto-routing).
- **No internet falls back to local** and says why, checked again at send time
  since the network can drop between loading the page and pressing enter.

The `model` field in `models.json` is the provider's own id and is the thing
most likely to go stale. Confirm it against the key rather than trusting it:

```bash
.venv\Scripts\python -c "from config import load_env_file; load_env_file(); from models.manager import ModelManager; from models.remote import RemoteClient; print(RemoteClient(ModelManager().get_spec('gemini')).list_models())"
```

That is not pedantry — the id shipped here was wrong until it was checked, and
a wrong one fails as a 404 that reads like an outage.

### Why it is now behind an API, having argued it should not be

An earlier version of this document rejected a FastAPI service, for two
reasons: every caller was already Python, and the venv carried a
`fastapi` ↔ `starlette` conflict. Both have since changed.

The first reason ended when the front end became a browser application. A
React app cannot import a Python class, so something has to speak HTTP, and
the choice is only *where* the boundary sits — not whether there is one.

The second was real and had to be cleared rather than worked around: FastAPI
0.115 pinned `starlette<0.46` while Streamlit had already installed 1.3.1, so
the venv could not have imported FastAPI at all. Upgrading FastAPI resolved
towards the version already present, and deleting Streamlit removed the other
side of the conflict entirely.

What the old note got right is that the manager stayed a plain class you
import. `api/` calls the same objects the CLI does and adds no logic of its
own, so the CLI still runs with no API and no Node anywhere in sight.

---

## 7a. Adding your own models

Copy a `.gguf` into `weights/` and the app finds it. There is nothing to edit
and no restart: press **Rescan folder** in Settings, or run `/rescan` in the
CLI.

```
weights/
├── Ministral-3-3B-Instruct-2512-Q4_K_M.gguf     declared in models.json
├── Qwen3-8B-Q4_K_M.gguf                         declared in models.json
└── Whatever-You-Dropped-In-Q4_K_M.gguf          found automatically
```

> It is `weights/`, not `models/`, for a boring reason: `models/` is the Python
> package (`models/manager.py`, `models/qwen.py`). A data directory with that
> name would shadow it on the import path. `models_dir` in `models.json` moves
> it anywhere you like, including another drive.

### What it works out for itself

A hand-written registry entry supplies three things a dropped-in file does not,
and all three are read from the model's own **GGUF header** rather than guessed:

| | Where it comes from |
|---|---|
| **context** | the largest of 2048/4096/8192 whose KV cache fits a 420 MB budget |
| **RAM threshold** | weights (file × 0.8, measured) + KV cache + 250 MB headroom |
| **port** | the next free one from 8090, skipping anything already claimed |

**Context is the one that matters.** Ministral is trained for 262,144 tokens
and costs 79,872 bytes of KV cache per token — running it as trained would ask
for **19.5 GB** of cache on an 8 GB machine. llama.cpp would try. So the
context is a RAM decision, made from the header, and the settings panel says so
rather than showing a number that looks arbitrary:

```
Running at 4,096 of the 262,144 tokens it was trained for;
the full context would need 19,968 MB of KV cache.
```

The sizing is checked against the entries that were measured by hand. It
reproduces GLM-OCR's 52,224 bytes/token exactly, puts GLM-OCR on 8192 and
Ministral on 4096 — the contexts those entries use — and estimates the 8B at
6,290 MB against the 6,200 MB someone measured for it.

An `mmproj-*.gguf` is recognised as a vision projector, paired with its model,
and never offered as something to talk to.

### Three layers, in this order

```
models.json            curated, hand-tuned, in version control
      ↓
weights/               everything the folder holds that is not already claimed
      ↓
data/models.local.json your choices from the settings panel
```

Layering rather than replacing is the point. A measured `min_free_mb` in
`models.json` beats anything inferred, so **a curated entry is never
overwritten by a scan of the same file**. And `models.json` is now optional: a
fresh clone with one `.gguf` and no registry works.

### Choosing the primary

On first launch, when there is more than one usable chat model and none has
been chosen, you are asked once:

```
Which model should be the primary?

  1. Ministral 3B      (2,047 MB, needs 1,900 MB free)
  2. Qwen3.5 2B (M)    (1,023 MB, needs 1,150 MB free)
  3. Qwen3.5 2B (XS)   (704 MB, needs 900 MB free)

  This machine has 995 MB free right now.
```

One model is not a choice, so a single-model install is never interrupted.

The answer goes in `data/models.local.json`, which is generated and
git-ignored. **Delete it to return to first-launch state** — it is not
load-bearing. Change it later in Settings, or with `/primary <key>`.

Choosing a primary deliberately does **not** load it. Choosing costs nothing;
loading costs minutes on this hardware, so they are two buttons.

### What Settings can change

Primary model, the router's two ends, and per-model `label`, `context`,
`threads` and `min_free_mb`. Retuning applies the next time that model starts,
because llama-server is given those on the command line.

Deliberately **not** editable: `file`, `port` and `role`. Those decide what a
model *is* and where it runs, and getting them wrong from a settings panel
produces a model that will not start for reasons the panel cannot explain.

Models you do not want in the picker can be hidden; the file stays where it is,
and the primary cannot be hidden.

```
POST   /api/models/rescan          re-read the folder
POST   /api/models/primary         {"key": "..."}
POST   /api/models/router          {"fast": "...", "strong": "..."}
PATCH  /api/models/{key}           retune one model
DELETE /api/models/{key}/override  back to registry values
POST   /api/models/{key}/hidden    {"hidden": true}
```

---

## 8. Auto-routing

Off by default. Turn on with the sidebar toggle or `/auto`.

When on, [`agent/router.py`](agent/router.py) scores each prompt with cheap
heuristics — **no extra model call** — and picks `fast` or `reasoning`.

Signals: prompt length, code fences, line count, demanding verbs (*debug*,
*refactor*, *trace*, *review*…), multiple questions, multiple filenames.
Everyday openers on a short prompt subtract. Score ≥ 3 routes to the strong
model.

Whole-word matching means *plan* does not fire inside *explanation*.

Live examples:

```
fast       score=0   hello
fast       score=0   what is 2+2?
fast       score=0   List the files in the workspace root
reasoning  score=4   Debug why the tool loop stalls and trace the root cause
reasoning  score=4   Refactor the registry and review the design decisions
```

### Two rules it always follows

**It never routes down.** Once a conversation has needed the strong model, it
stays there. Switching back would pay the ~5 minute cost twice to reclaim RAM
that is already spent.

**Every switch is announced** in the chat with the reason.

### Escalation on failure

Hitting the iteration limit is the clearest signal the small model is out of its
depth, so the turn is retried once on the strong model. This is cheaper than
asking a model to classify every prompt: a classification round would cost
10–20 s on *every* turn, including the trivial ones.

The heuristics lean small on purpose. Guessing small and escalating wastes one
turn; guessing big wastes minutes on every simple question.

---

## 9. Tools

| Tool | Category | Status |
|---|---|---|
| `calculate` | calculator | **on** |
| `list_directory` | filesystem | **on** |
| `read_text_file` | filesystem | **on** |
| `run_python` | python | **off by default** |
| `run_command` | terminal | **off by default** |
| `run_python_file` | python | with the python tool; restricted unless `AGENT_PYTHON_UNRESTRICTED=1` |
| `write_text_file`, `create_directory` | filesystem | **off by default** |
| `git_status`, `git_log`, `git_diff`, `git_branches` | git | **off by default** |
| `git_commit`, `git_create_branch` | git | needs a second flag |
| `remember`, `recall`, `search_memory`, `update_memory`, `forget_memory` | memory | **off by default** |
| memory background processing | memory | needs a second flag |
| `http_request` | http | **off by default** |
| `ocr_image` | ocr | **off by default** — Tesseract or the GLM-OCR model |
| `search_documents`, `list_documents` | documents | **off by default** |

Disabled tools are not registered at all — sending the model a definition it can
only fail on wastes context and a whole round-trip.

**Turning them on.** Each of these has an environment variable, listed in
[section 13](#13-configuration), and each can also be switched from the
sidebar. The switches apply from the next turn and last until the API
restarts; the environment variables are what make a choice permanent. The
second flags — unrestricted Python, git commits, state-changing HTTP — are
separate switches nested under their tool, and go off with it.

Read the section for a tool before enabling it. What each one can and cannot
protect you from is written there, and the same text appears next to its
switch.

### calculator

Exact arithmetic so the model never does sums in its head. Parsed with `ast` and
walked against a whitelist — **no `eval`**. Supports `+ - * / // % **`, the
constants `pi e tau`, and `sqrt cbrt exp log log2 log10 sin cos tan asin acos
atan atan2 sinh cosh tanh degrees radians hypot abs round floor ceil trunc min
max gcd lcm factorial comb perm isqrt`.

Rejects imports, function definitions, attribute access (so
`().__class__.__bases__` fails), assignments, strings, and any name outside the
constants. Exponent and factorial are capped.

```
{"success": true, "result": 637.0, "formatted": "637"}
```

### filesystem

Read-only: `list_directory(path)` and `read_text_file(path)`. No write, rename,
delete, chmod or execute exists anywhere in the module.

Paths are resolved **first** — collapsing `..` and following symlinks — and the
result must sit under the workspace root. That is what makes the check hold
against `../`, absolute paths and symlink tricks. Oversized files are refused.

Default workspace is the project directory; set `AGENT_WORKSPACE` to move it.

### python  — read this before enabling

Set `AGENT_ENABLE_PYTHON_TOOL=1` only after understanding the limitation.

What it does: screens code with `ast` (rejecting imports, dunder names and
attributes, and `open`/`eval`/`exec`/`compile`/`getattr` and friends), then runs
it in a **separate process** (`python -I -S`) with stripped `__builtins__`, a
temporary working directory, a wall-clock timeout and an output cap.
`math statistics random itertools functools decimal fractions re json datetime
collections` are pre-imported.

**What it is not: a sandbox.** CPython was never designed to contain hostile
code in-process, and escaping a restricted-builtins namespace is a known class
of trick. The separate process and the AST screen raise the cost and stop the
obvious attempts. They do not make it safe to run code from an untrusted source.

Enabling it means trusting model output roughly as much as code you paste into a
terminal yourself. Real isolation means a container, a VM or a seccomp jail —
a deliberate follow-up rather than something faked here.

### terminal — read this before enabling

Set `AGENT_ENABLE_SHELL_TOOL=1` only after understanding the boundary.

Handing a language model a shell is the most dangerous thing in this project,
so it is an **allowlist of specific commands**, not a filter over arbitrary
ones. Denylists leak; allowlists fail closed.

**The single most important property: there is no shell.** The command is
tokenised and handed straight to `CreateProcess`. Nothing reaches `cmd.exe`, so
`;`, `&&`, `||`, `|`, `>`, backticks and `$(...)` are not metacharacters — they
are literal argument text. Chaining is removed as a category rather than
filtered.

Demonstrated in a real repository:

```
$ git status --short          exit 0    ?? a.txt
$ git status && whoami        exit 0    On branch main        ← username never appears
$ git log --oneline ; whoami  exit 128  fatal: ambiguous argument ';'
```

That last line is the proof: git received `;` as an argument, so there was
never a second command to run.

**The rest of the boundary, in order of how much it protects you:**

| Layer | What it stops |
|---|---|
| Executable allowlist | Only `git`, `pip`, `python`, `where`. A path separator in the command name is refused, so `./evil.exe` and `C:/Windows/System32/cmd.exe` never resolve |
| Sub-command allowlist | git is limited to read-only verbs. No commit, push, pull, fetch, reset, checkout, clean, rebase, merge, stash — and no `config`, which writes as readily as it reads |
| Dangerous option screening | `git -c core.pager='sh -c …' log` executes anything; `--exec-path` relocates git's own binaries. Both refused, along with `-C`, `--git-dir`, `--work-tree`, `--upload-pack` |
| Interpreter pinning | `python` accepts only `--version`/`-V`. `python -c …` and `pip install` are refused — an interpreter would undo everything above |
| Workspace confinement | cwd is the workspace; absolute paths and `..` segments are refused in arguments |
| Scrubbed environment | The child gets a minimal PATH-and-essentials env, so API keys in your shell are never handed to a subprocess |
| Timeout and output cap | 30 s, 4000 characters |

A non-zero exit is reported to the model rather than raised — `git diff --quiet`
answers questions through its exit code.

**What this is not: a sandbox.** Allowed commands run with your account's
privileges. The protection is that the reachable set is small, read-only and
chosen deliberately — not that a hostile command would be contained if one got
through.

`AGENT_SHELL_EXTRA` adds executables, with **no** sub-command restrictions
invented for them. That is exactly as safe as the programs you add; adding an
interpreter (`node`, `powershell`, `bash`) hands over arbitrary execution.

> **Paths use forward slashes.** POSIX tokenising is used so quoted arguments
> with spaces survive, and that treats `\` as an escape — which would silently
> turn `sub\file.txt` into `subfile.txt`. Backslashes are refused with a
> pointer to forward slashes, which Windows accepts anyway.

### http — read this before widening it

Set `AGENT_ENABLE_HTTP_TOOL=1` to enable. It defaults to **loopback only**, so
the agent can inspect your own services — the llama servers included — without
being able to reach the internet.

Adding a public host to `AGENT_HTTP_HOSTS` is what turns this from a
local-service inspector into a web client, and it is the point at which the
project stops being local-only. That is a deliberate decision, not a default.

**Redirects are refused, not followed.** This is the layer that matters most
and the one that is easy to get wrong: an allowed host answering `302` with a
`Location` pointing anywhere would walk straight out of the allowlist, and
following it would mean validating one URL while fetching another. The redirect
is reported to the model instead, which can request the new URL and have it
checked properly.

| Layer | Stops |
|---|---|
| Host allowlist | Checked against the parsed hostname, so `localhost.evil.com` does not pass by prefix |
| Scheme allowlist | `http`/`https` only. Without it, `file://` would be an unrestricted file reader that ignores the workspace jail |
| No redirect following | A permitted host bouncing the request to an unpermitted one |
| Read-only by default | `POST`/`PUT`/`PATCH`/`DELETE` need `AGENT_HTTP_ALLOW_WRITES=1` |
| `trust_env = False` | Proxy settings and `.netrc` credentials riding along on a request the model composed |
| URL credential check | `user:pass@host` being quietly forwarded |
| Size cap, timeout | 100 KB, 20 s. Binary bodies are described, not dumped into the conversation |

A non-2xx status is reported rather than raised — `404` is an answer to a
question, and the model should see it.

Verified against the running servers:

```
200  http://127.0.0.1:8081/health    19ms  {"status":"ok"}
200  http://127.0.0.1:8083/health     0ms  {"status":"ok"}
200  http://127.0.0.1:8081/props      8ms  {"default_generation_settings":...
ERR  http://127.0.0.1:9999/nothing         Could not connect. Is the service running?
ERR  https://example.com/                  'example.com' is not an allowed host
ERR  file:///C:/Windows/win.ini            Only http and https urls are allowed
```

It is not a browser: no cookie jar, no session state, no JavaScript.

### ocr — reading text out of images

Reads text from images through a second llama-server running GLM-OCR.

**Verified end-to-end.** A generated delivery note was transcribed with every
field correct — reference `HK-4127-B`, recipient, crate count and signature — in
**37 s** first run, ~28 s after. A four-row table came back with all figures
intact.

### Two backends, and how to choose

`ocr_image` has two readers behind it. They are different trades rather than
more and less of one thing, which is why this is a chooser in the Tools pane
and not a quality setting:

| | RAM | One page | Layout |
|---|---|---|---|
| **Tesseract** | ~50 MB | **0.5 s** (measured) | no — lines of text, in order |
| **GLM-OCR** | ~1.4 GB | ~30 s | yes — tables, columns, handwriting |

Measured here against Tesseract 5.5.3 and the images in `samples/`:
`note.png` in 0.56 s and `table.png` in 0.44 s, both transcribed correctly.
The table came back as plain rows with no structure, exactly as the row above
says it will.

On an 8 GB machine that gap decides most cases. "Read the text off this
screenshot" does not need a vision model, and paying 1.4 GB and half a minute
for it is a poor trade. A scanned page whose table you actually need is exactly
what the model is for.

`OCR_BACKEND` picks, and the Tools pane has a chooser. The model's own tool
description changes with it, because the two behave differently enough that one
description would be a lie for whichever is running:

> **Tesseract** — "…it transcribes text line by line. It does not follow
> instructions and does not preserve tables or columns, so do not ask it to
> extract a particular field."
>
> **GLM-OCR** — "…it understands layout. Use for photos, scans, tables and
> handwriting, and say what to extract."

A `prompt` sent to Tesseract comes back with a `note` saying it was ignored,
rather than being silently dropped.

Everything above the backend is shared: the same workspace jail, the same
extension allowlist, the same size cap.

#### Installing Tesseract

```
scoop install tesseract
```

or `winget install UB-Mannheim.TesseractOCR`, or the installer from
[UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). If it ends up
somewhere unusual, `TESSERACT_CMD` takes a full path — and the code already
checks `C:\Program Files\Tesseract-OCR\` before giving up, because that is
where the installer puts it and it does not touch PATH.

> **Scoop's package ships no language data.** The binary installs, but
> `--list-langs` returns nothing and every read fails with "Could not
> initialize tesseract". The full `tesseract-languages` package is over a
> gigabyte; one file is enough:
>
> ```bash
> curl -L -o "$env:USERPROFILE\scooppps	esseract\current	essdata\eng.traineddata" https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata
> ```
>
> That is 4 MB, and the error message names the missing pack rather than
> failing vaguely. The UB-Mannheim installer bundles English already.

`TESSERACT_LANG` selects a language pack (`eng` by default) and a missing one
is reported as a missing pack rather than a generic failure. `TESSERACT_PSM`
is page segmentation: 3 is automatic, and 6 — "a single uniform block" — is
what to try when 3 scrambles a simple image.

Tesseract is called as a subprocess rather than through `pytesseract`. The
wrapper's whole job is to build an argument list and read stdout, and this
project already starts and supervises `llama-server` the same way; a dependency
to avoid twenty lines would not pay for itself.

#### The two files it needs

GLM-OCR ships as a pair, and the language half alone is not enough:

| File | Size | Contains |
|---|---|---|
| `GLM-OCR-Q8_0.gguf` | 906 MB | Language model. `general.architecture = glm4`, 179 tensors, all `blk.N.*` |
| `mmproj-GLM-OCR-Q8_0.gguf` | 461 MB | Vision encoder. `general.type = mmproj`, `clip.projector_type = glm4v`, 340 `v.blk.*` tensors |

They pair correctly because `clip.vision.projection_dim = 1536` matches
`glm4.embedding_length = 1536`. If you ever swap either file, that is the
number to check.

> If you go looking in the GGUF yourself: grepping the language file for
> "vision" returns two dozen hits and every one is tokenizer vocabulary —
> *television*, *supervision*, *division*. The tensor names are what matter.

#### Running it

```bash
"C:\Users\SHAMI\HAKIM\AI\LLAMA CP\llama-server.exe" -m "C:\Users\SHAMI\HAKIM\AI\Hakim Local Agent\weights\GLM-OCR-Q8_0.gguf" --mmproj "C:\Users\SHAMI\HAKIM\AI\Hakim Local Agent\weights\mmproj-GLM-OCR-Q8_0.gguf" -c 4096 -t 4 -np 1 --port 8081
```

Then set `OCR_ENABLED=1` and the tool registers itself. Loading takes about
10 seconds — far quicker than the chat models, because it is small.

The server confirms itself on `GET /props`:

```json
"modalities": {"vision": true, "video": true, "audio": false}
```

The tool reads exactly that before sending, and refuses early with a clear
message if vision is absent.

#### RAM

It runs **alongside** the chat model rather than in the manager's one-at-a-time
rotation, so budget for both: roughly 1.4 GB of weights for GLM-OCR on top of
whichever chat model is loaded. Pair it with `tiny` or `fast`, not `reasoning`.

#### Two behaviours worth knowing

**Tables always come back as HTML.** Asking for markdown makes no difference —
GLM-OCR is trained to emit `<table><tr><td>` and ignores the instruction. The
data is accurate; the markup is not negotiable.

**Mentioning tables at all changes plain output.** An earlier default prompt
said "render tables as markdown", and the model then wrapped even five plain
lines in table markup. The default now asks for plain text and never mentions
tables, which produces clean line-by-line output. Pass `prompt` yourself when
you want something else.

Sample images to try are in `samples/`.

### memory — what the agent remembers about you

Off by default. `AGENT_ENABLE_MEMORY=1`, or the **Memory** switch in the
sidebar.

Not a chat-history buffer. Four kinds of memory, stored separately and
retrieved by meaning:

| | What it holds | Example |
|---|---|---|
| **working** | the current task, in the prompt only, never persisted | "fixing the PDF importer" |
| **episodic** | events worth recalling later | "User added Biology.pdf" |
| **semantic** | durable facts and preferences | "User prefers lightweight local solutions" |
| **summaries** | what was dropped from a long conversation | one paragraph per conversation |

#### The rule everything is built around

**Only one chat model is ever loaded.** Ministral is the primary reasoner;
Qwen3.5 2B is an optional memory worker. They are never resident together, and
that is not enforced by new code — it is enforced by `ModelManager.ensure()`,
which already stops every other chat model when `max_active` is 1. The memory
processor drives that same manager rather than loading anything itself.

```
 normal turn      Ministral loaded -> answer -> done
 retrieval        no model switch: SQLite + embeddings + arithmetic
 processing       Ministral stopped -> Qwen loaded -> whole batch -> Qwen stopped
```

#### Retrieval never loads a language model

This is the part that makes memory cheap enough to use on every turn. Storing,
ranking, decay, deduplication and forgetting are all ordinary code:

```
question -> embed (BGE, the same worker document search uses)
         -> cosine against the memory index
         -> similarity gate
         -> score = similarity x importance x confidence x recency x type x usage
         -> top k -> context builder -> Ministral
```

A **product**, not a weighted sum, so every factor has a veto: a vaguely
similar, stale, low-confidence guess cannot out-rank an exact match by being
strong on one axis.

**The similarity gate is the number that matters.** BGE-small has a high noise
floor — two unrelated English sentences score 0.4–0.55, not zero. Measured on
this machine against five stored memories:

| Query | Best hit |
|---|---|
| "what editor do I use?" | 0.664 |
| "what setup do I normally use?" | 0.575 |
| **"what is photosynthesis?"** | **0.549** |
| "write a python function to reverse a string" | 0.479 |
| "explain the french revolution" | 0.395 |

So `MEMORY_MIN_SIMILARITY` defaults to **0.55**, and unrelated questions
retrieve nothing at all. Set it lower and every question drags the whole store
into the prompt — that was a real bug here, found only by running it against
the real model, because a bag-of-words test fake scores unrelated text at ~0
and hides the problem entirely.

> The honest cost: a genuinely relevant question can land just under the gate.
> "What kind of tooling does this person like?" scores 0.506 and is cut. The
> trade is deliberate — precision over recall, because a wrong memory asserted
> confidently is worse than a missing one — but if you change `RAG_MODEL`, this
> number has to be re-measured.

#### Decay, not deletion

Memories carry `importance` and `confidence` (0–1) and decay on an exponential
half-life set by their type — 720 days for a preference, 5 for a temporary
note. Nothing is deleted by age; it just stops out-ranking fresher things.

Statuses are `active`, `archived`, `superseded` and `deleted`. **Superseded is
how contradictions are handled.** Told "I use Qwen" and later "I've switched to
Mistral", the newer memory wins and the older one is kept, linked, and still
answerable:

```
CURRENT   User uses Mistral        (active)
HISTORY   User uses Qwen           (superseded_by -> the above)
```

That resolution is deterministic — same type, same subject, both active, newer
wins — and needs no model.

#### What gets remembered, and what does not

After each turn, `observe_turn` runs a couple of regexes and usually does
nothing. In order:

1. **"Remember that ..."** is stored immediately — no queue, no model. An
   explicit instruction that sat in a queue behind a model switch would make
   "I'll remember that" a lie.
2. **Greetings, thanks, "ok"** are rejected outright and never reach the queue.
3. **Clear patterns** — "I always use X", "I've switched to Y" — are stored
   directly, with no model.
4. **Anything else** is queued, and only every fourth message.

Hedging and time-boxing are checked *first*, so "I might always use X" becomes
an `intention` at low confidence and "I'm using X today" becomes `temporary`.
Neither becomes a permanent preference. An uncertain intention must never
harden into a fact.

#### The queue, and why it batches

Jobs live in SQLite and survive a restart; anything left `running` by a crash
is returned to the queue at startup. A batch runs only when the agent is idle
**and** enough work has piled up (`MEMORY_QUEUE_HIGH_WATER`, default 6) —
switching models after every message would cost minutes for nothing.

When it does run, one model session handles the whole batch: extraction,
classification, consolidation and summarisation together. Six jobs cost one
load and one stop, not six of each. If a turn arrives mid-batch the remaining
jobs go back on the queue and the auxiliary model is stopped immediately —
responsiveness wins.

The auxiliary model is stopped on **every** exit path, including a crashing
job. Leaving it resident is the one outcome that would break the memory ceiling
for the next turn.

#### It works without the auxiliary model

Background processing is a separate switch (`AGENT_ENABLE_MEMORY_PROCESSING=1`)
and is off by default, because a model swap is a visible pause. Without it you
still get explicit memories, semantic retrieval, ranking, decay, deduplication,
conflict resolution and forgetting. Qwen makes memory *smarter*; it is not a
dependency.

#### Talking to it

Five tools reach the model: `remember`, `recall`, `search_memory`,
`update_memory`, `forget_memory`. "Forget that I prefer X" actually calls
`forget_memory` rather than the model saying it will.

```
.venv\Scripts\python -m rag stats      # documents
curl 127.0.0.1:8000/api/memory/stats   # memories
```

`GET /api/memory`, `POST /api/memory`, `POST /api/memory/search`,
`PATCH /api/memory/{id}`, `DELETE /api/memory/{target}`,
`POST /api/memory/consolidate`, `POST /api/memory/process`, `GET /api/memory/stats`.
`process` and `consolidate` are refused mid-turn.

#### Context, under a budget

Every turn is assembled by a context builder rather than "system prompt plus
everything":

```
system instructions            always
working memory                 the current task, if any
conversation summary           <= 12% of the budget
retrieved memories             <= 15% of the budget
recent messages                <= 55%, and never fewer than 4
```

Budgets are characters converted from tokens at the same conservative ratio the
document chunker uses, so estimates run high and the real context comes out
under. Memories and summaries are capped *first*: a turn where retrieved
memories crowded out the question being asked would be worse than no memory at
all.

#### Where it lives

Same SQLite file as the chat history — `memory_items`, `memory_links`,
`memory_jobs`, `conversation_summaries` — plus a `data/memory/memory.f32`
vector file, the same flat-float32 format the document index uses.

The old keyed `memories(key, value)` table is imported on first open and then
**renamed** to `memories_legacy`, not dropped. An upgrade should never be the
thing that loses your data.

### documents — semantic search over your own files

Off by default. It also has dependencies the rest of the project does not,
kept in their own file so `pip install -r requirements.txt` never drags torch
in behind your back:

```
.venv\Scripts\pip install -r requirements-rag.txt
```

Then `AGENT_ENABLE_RAG=1`, or the **Document search** switch in the sidebar.
Nothing in `rag/` is imported until you do — the CLI and the API both start
without any of it installed, and the endpoints answer `501` with the install
line rather than a traceback.

Ask "according to my biology notes, explain the light-dependent stage" and the
agent searches your own indexed files, gets back the passages that actually
mean that, and answers from them with the document and page named. Ask "what is
25 × 17?" and it uses the calculator instead, because the tool description
tells it which is which.

#### The pipeline

```
document -> extract -> clean -> chunk -> embed -> vectors.f32
                                             \-> chunks.db  (text + metadata)
```

Each stage is one module in [`rag/`](rag/), and nothing in it reaches the
network once the embedding model is on disk.

| File | What it does |
|---|---|
| [`rag/extractor.py`](rag/extractor.py) | file → text, page by page for PDFs |
| [`rag/chunker.py`](rag/chunker.py) | text → overlapping chunks on paragraph boundaries |
| [`rag/embeddings.py`](rag/embeddings.py) | owns the worker process; starts it late, stops it early |
| [`rag/worker.py`](rag/worker.py) | the model itself, in its own process |
| [`rag/index.py`](rag/index.py) | the vector file, searched with numpy |
| [`rag/metadata.py`](rag/metadata.py) | chunk text, documents, the free list (SQLite) |
| [`rag/manager.py`](rag/manager.py) | decides the order everything happens in |

#### The model

**BAAI/bge-small-en-v1.5** — 384 dimensions, a 512-token window, 134 MB on
disk. It downloads itself the first time anything needs it, into the standard
Hugging Face cache; `RAG_MODEL_DIR` moves it. To fetch it ahead of time:

```
.venv\Scripts\python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

Queries are embedded with BGE's own retrieval instruction prefix and passages
without one — that asymmetry is what the model was trained for, and dropping it
costs recall. Vectors are normalised, so cosine similarity is a dot product.

> On Windows the first download fails with **WinError 1314: a required
> privilege is not held** unless Developer Mode is on, because the Hugging Face
> cache wants to create symlinks. The worker sets `HF_HUB_DISABLE_SYMLINKS=1`
> so it copies instead. Nothing to do about it; it is written down because the
> raw error is baffling.

#### Indexing

Ingestion has nothing to do with chatting, so it has its own entry point and
needs no model server running:

```
.venv\Scripts\python -m rag index "C:\Users\you\Documents\notes"
```

```
index <path>     a file, or every supported file in a folder
search "<text>"  the same search the agent's tool makes
list             what is indexed
remove <id>      drop one document
rebuild          re-read and re-embed everything from source
compact          reclaim rows left behind by deletions
stats            index size, model, and whether the model is loaded
```

Handles `.txt` `.md` `.pdf` `.py` and about fifty other text and source
formats. PDFs are read with **pypdf**, chunked one page at a time so `page` in
a result is true rather than approximate.

**Re-running `index` is cheap and safe.** A file is skipped when its size and
modification time are unchanged; if those moved, it is hashed, and skipped
again if the hash matches. Only genuinely changed files are re-embedded, and a
document is replaced by path in a single transaction — so running it twice
cannot produce duplicate chunks. Re-indexing an unchanged folder of two
documents takes 0.7 s and never loads the model at all.

The same operations are on the API: `POST /api/rag/index`, `POST /api/rag/search`,
`GET /api/rag/documents`, `DELETE /api/rag/documents/{id}`, `POST /api/rag/rebuild`,
`GET /api/rag/stats`, `POST /api/rag/unload`. Indexing is refused while a turn
is running — it would fight it for both cores — but searching is not, because
that is the agent's own tool call.

#### Why no FAISS, and no vector database

The index is a flat file of float32 read with `numpy.memmap` and searched in
6 MB blocks. FAISS starts earning its keep somewhere around a million vectors;
a personal document collection is three orders of magnitude short of that —
20,000 chunks is 29 MB and one matrix multiply. Against that, `faiss-cpu` is a
binary wheel to install, a second file format to keep in step with the
metadata, and an index that has to be rebuilt to delete anything. The
brute-force scan is also *exact*: no recall is lost to an approximation.

Chunk text and metadata live in SQLite beside it rather than in JSON, because a
JSON store has to be read and rewritten whole — re-indexing one file in a
collection of five hundred would rewrite every record and hold all of them in
memory to do it. A search reads the five rows it hit.

Deleting a document does not move any vectors, because moving one would
invalidate every row number in the metadata. Its rows go on a free list and are
overwritten by the next document indexed; `compact` squeezes them out when you
want the disk back.

#### RAM — the part that shaped the design

The embedding model **never runs in the API process**. It runs in a child
process, started on first use and stopped once idle by the same sweeper that
unloads idle llama-servers. Importing torch in-process would add a few hundred
megabytes that Python never gives back; a child process gives back every byte
the moment it exits.

Measured here, embedding 1,750-character chunks:

| | Worker RSS |
|---|---|
| model loaded, idle | ~417 MB |
| embedding, batch 4 | ~535 MB |
| embedding, batch 8 *(default)* | ~578 MB |
| embedding, batch 16 | ~674 MB |
| embedding, batch 32 | ~821 MB |
| **after unload** | **0 — the process is gone** |

Throughput is ~2 chunks/sec across that whole range, because two cores are the
bottleneck rather than the batch size. That is why the default is 8: the bigger
batches buy memory pressure and nothing else.

Cold start is the real cost — **50–110 s** for the first search, almost all of
it importing torch. The worker then stays up for `RAG_IDLE_SECONDS` (120 s by
default), so follow-up searches are ~0.3 s.

**Do not index a large folder while the 8B model is loaded.** 417 MB next to
~5 GB of Qwen on an 8 GB machine is how you find the swap file. Index first,
then chat — which is what the separate `python -m rag` entry point is for.

---

## 10. Chat history

[`chat_store.py`](chat_store.py) — SQLite, standard library, no new dependency.

Two tables. `conversations` holds title, model, timestamps. `messages` holds
role, content, tool calls as JSON, elapsed time. Tool calls are display metadata
rather than conversation state, so they do not earn their own table.

A connection is opened per operation rather than held open — turns run on a
worker thread while requests are served on others, and a shared `sqlite3`
connection is not safe across threads.

Timestamps are microsecond resolution. At second resolution several updates land
in the same tick and "most recently updated" ordering silently falls back to
insertion order.

In the sidebar: recent conversations newest-first, click to reload, ✕ to delete,
**New conversation** to start fresh. Titles come from the first message.
A conversation row is only created on the first message, so idle sessions do not
litter the database.

Loading a saved conversation restores the **agent's** transcript too — otherwise
the model would answer against a conversation you cannot see.

Stored at `data/chat_history.db`; `AGENT_DB_PATH` moves it.

---

## 11. The web UI

React 19, TypeScript, Vite 8 and Tailwind 4, in [`web/`](web/). About 70 kB
gzipped, because the dependency list stops at those four.

- User turns are right-aligned accent bubbles; replies sit flat and full-width
- **Every tool call expands** to the arguments it sent and the whole payload it
  got back - the same thing the model saw. A 60-character summary is enough to
  know `read_text_file` ran and useless for checking it read the right file
- Copy buttons on prompts, answers, the reasoning trace, and each half of a
  tool call
- **Attach an image** with the paperclip or by dropping it on the composer, and
  the agent reads it with OCR
- **Each stage of a turn is named** — queued with its position, model loading,
  generating — with a live elapsed timer. One spinner for all three says
  nothing on a machine where each takes minutes, and telling slow progress
  from a hang is the entire difficulty
- The model's **reasoning** streams into a collapsible panel — see below
- **Tool switches** in the sidebar, with each tool's own risk text
- Slash completion inline in the composer, and a ⌘K palette over commands,
  models and conversations
- Sidebar folds away; light and dark both supported

### No markdown library, and no innerHTML

[`web/src/lib/markdown.tsx`](web/src/lib/markdown.tsx) renders the subset that
actually appears in replies — fenced code, headings, lists, quotes, and inline
code, bold, italic and links — in about a hundred lines.

It builds React elements, never HTML, so there is no `dangerouslySetInnerHTML`
anywhere in the app and no sanitiser to get wrong. A model that emits
`<script>` emits eight harmless characters. Links are restricted to `http(s)`,
so a `javascript:` URL renders as text.

### Attachments and OCR

The OCR tool takes a path *inside the workspace* and resolves it through the
same jail as the filesystem tools. A browser upload is bytes, so `POST
/api/uploads` is the bridge: bytes in, a workspace-relative path out, which is
exactly what `ocr_image` wants.

Uploads therefore land in `config.workspace`, not beside the database — if
`AGENT_WORKSPACE` points elsewhere they follow it, because anywhere else and
the agent could not read its own attachment.

The path is then **named in the prompt itself**, not passed beside it. The
model has no other way to learn the file exists, and folding it in means the
stored message is exactly what the model was asked, so replaying the
conversation later stays faithful.

Two things must be true before it works, and the composer distinguishes them
because the fixes differ: the **OCR tool** must be switched on, and the
**GLM-OCR server** must be running — it is separate from the chat models, is
not in the manager's rotation, and needs both its model and its `mmproj` file:

```bash
"C:\Users\SHAMI\HAKIM\AI\LLAMA CP\llama-server.exe" -m weights\GLM-OCR-Q8_0.gguf --mmproj weights\mmproj-GLM-OCR-Q8_0.gguf -c 4096 -t 4 -np 1 --port 8081
```

Confirm it came up with vision, which is the check that actually matters:
`GET http://127.0.0.1:8081/props` should report `"modalities": {"vision": true`.

Nothing about an upload is trusted: the filename is rebuilt rather than
sanitised, the extension must be one the OCR tool accepts, and the size cap is
enforced *while* reading, so an oversized file is never held in memory and a
refused upload leaves nothing on disk.

### Reasoning

With `--reasoning-format deepseek`, llama.cpp returns thinking as
`reasoning_content` deltas. Those stream on their own channel — `on_reasoning`,
never mixed into the answer — as far as a `reasoning` event and a collapsed
panel.

**It is never sent back to the model, and never stored.** `_assistant_entry`
keeps the answer and the tool calls only, so no thinking trace is replayed
into a later prompt, and nothing writes it to the database. It lives until the
page reloads, and the panel says so.

Only models that think produce any: Qwen3 with **Extended thinking** on.
Ministral does not, and its panel simply never appears.

---

## 12. Commands

Type `/` in the composer and a dropdown appears. ↑/↓ to move, Enter or Tab to
accept, Esc to dismiss, or click. **⌘K / Ctrl+K** opens a palette over the same
commands plus every model and saved conversation.

| Command | Does |
|---|---|
| `/help` | Show the commands |
| `/models` | List models and which is loaded |
| `/model <key>` | Select a model; it loads on your next message |
| `/load <key>` | Load a model now |
| `/unload` | Unload the current model, free its RAM |
| `/tools` | List enabled tools and what is off |
| `/auto` | Toggle automatic routing |
| `/thinking` | Toggle extended thinking |
| `/new`, `/clear` | Start a new conversation |

**The server knows nothing about commands.** It has REST endpoints, and these
are the client's shorthand for them — unlike the Streamlit build, where
`/model` was parsed out of the message text server-side. Everything a command
does is also a control in the UI.

The CLI has its own smaller set plus `/quit`.

---

## 13. Configuration

Everything is environment-driven with local defaults. See [`config.py`](config.py).

### API keys

Hosted models need one. They go in `.env` beside `models.json`, which is
git-ignored, and are read at startup by a small loader in `config.py` — no
dependency, and real environment variables win over the file so you can
override one for a single run.

```
GEMINI_API_KEY=...
CEREBRAS_API_KEY=...
```

Restart the API after editing it. The key is never stored on a model spec,
never sent to the browser, and is scrubbed out of provider error bodies before
they can reach a log — some providers echo the request back in an error.

Which variable a model wants is `api_key_env` in [`models.json`](models.json).
A missing key is reported as exactly that, naming the variable, rather than as
a connection failure.

| Variable | Default | Meaning |
|---|---|---|
| `QWEN_SERVER_URL` | `http://127.0.0.1:8080` | Model server |
| `QWEN_MODEL` | `qwen3` | Cosmetic — server ignores it without `--alias` |
| `QWEN_TEMPERATURE` | `0.7` | Sampling |
| `QWEN_TOP_P` | `0.8` | Sampling |
| `QWEN_MAX_TOKENS` | `-1` | −1 lets the server decide |
| `QWEN_ENABLE_THINKING` | `1` | `0` is much faster on CPU |
| `OCR_SERVER_URL` | `http://127.0.0.1:8081` | GLM-OCR server |
| `OCR_MODEL` | `glm-ocr` | Cosmetic, as with `QWEN_MODEL` |
| `OCR_ENABLED` | `0` | Set to 1 once the OCR server is running |
| `OCR_MAX_IMAGE_BYTES` | `10000000` | Size limit |
| `AGENT_REQUEST_TIMEOUT` | `1200` | Seconds. Generous because CPU is slow |
| `AGENT_CONNECT_TIMEOUT` | `10` | Seconds |
| `AGENT_MAX_HISTORY` | `60` | Messages kept; 0 disables trimming |
| `AGENT_MAX_ITERATIONS` | `8` | Tool rounds per turn |
| `AGENT_WORKSPACE` | project dir | The only directory tools may read |
| `AGENT_MAX_READ_BYTES` | `200000` | Largest readable file |
| `AGENT_DB_PATH` | `data/chat_history.db` | History and memory database |
| `AGENT_ENABLE_PYTHON_TOOL` | `0` | See the security note |
| `AGENT_PYTHON_TIMEOUT` | `10` | Seconds |
| `AGENT_PYTHON_MAX_OUTPUT` | `4000` | Characters |
| `AGENT_ENABLE_SHELL_TOOL` | `0` | See the security note |
| `AGENT_SHELL_TIMEOUT` | `30` | Seconds |
| `AGENT_SHELL_MAX_OUTPUT` | `4000` | Characters |
| `AGENT_SHELL_EXTRA` | empty | Extra allowed executables, comma-separated |
| `AGENT_ENABLE_FILE_WRITES` | `0` | Create files and directories |
| `AGENT_MAX_WRITE_BYTES` | `200000` | Largest writable file |
| `AGENT_PYTHON_UNRESTRICTED` | `0` | Run scripts as plain CPython |
| `AGENT_ENABLE_GIT_TOOL` | `0` | Structured git reading |
| `AGENT_GIT_ALLOW_WRITES` | `0` | Permit commits and branch creation |
| `AGENT_GIT_TIMEOUT` | `30` | Seconds |
| `AGENT_ENABLE_MEMORY` | `0` | Durable facts across conversations |
| `AGENT_ENABLE_HTTP_TOOL` | `0` | See the security note |
| `AGENT_HTTP_HOSTS` | loopback | Allowed hosts, comma-separated |
| `AGENT_HTTP_TIMEOUT` | `20` | Seconds |
| `AGENT_HTTP_MAX_BYTES` | `100000` | Response size cap |
| `AGENT_HTTP_ALLOW_WRITES` | `0` | Permit POST/PUT/PATCH/DELETE |
| `OCR_BACKEND` | `model` | `tesseract` or `model` |
| `TESSERACT_CMD` | *(found)* | Full path to tesseract.exe when it is not on PATH |
| `TESSERACT_LANG` | `eng` | Language pack |
| `TESSERACT_PSM` | `3` | Page segmentation; 6 for a single block |
| `OCR_TIMEOUT` | `120` | Seconds, both backends |
| `AGENT_ENABLE_MEMORY_PROCESSING` | `0` | Let the agent swap models to process memory when idle |
| `MEMORY_AUX_MODEL` | `tiny` | The memory worker, from models.json |
| `MEMORY_STORE` | `data/memory` | Memory vector index |
| `MEMORY_TOP_K` | `5` | Memories per turn |
| `MEMORY_MIN_SIMILARITY` | `0.55` | Re-measure if RAG_MODEL changes |
| `MEMORY_SCORE_FLOOR` | `0.10` | Composite score floor |
| `MEMORY_CONTEXT_TOKENS` | `3000` | Whole context budget |
| `MEMORY_EXTRACT_EVERY` | `4` | Queue extraction every N messages |
| `MEMORY_QUEUE_HIGH_WATER` | `6` | Jobs before a switch is worth it |
| `MEMORY_SUMMARIZE_AFTER` | `24` | Messages before summarising |
| `MEMORY_BATCH_SIZE` | `12` | Jobs per model session |
| `AGENT_ENABLE_RAG` | `0` | Document search |
| `RAG_STORE` | `data/rag` | Where the index lives |
| `RAG_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `RAG_MODEL_DIR` | *(HF cache)* | Pin the model somewhere else |
| `RAG_DIMENSION` | `384` | Must match the model |
| `RAG_CHUNK_TOKENS` | `500` | Chunk size; capped at the model's 512 |
| `RAG_CHUNK_OVERLAP` | `75` | Clamped to half the chunk |
| `RAG_TOP_K` | `5` | Passages per search |
| `RAG_MIN_SCORE` | `0.3` | Cosine similarity floor |
| `RAG_CONTEXT_CHARS` | `6000` | Retrieved text handed to the model per call |
| `RAG_THREADS` | `2` | Shared with llama-server |
| `RAG_BATCH_SIZE` | `8` | Bigger costs RAM, not speed |
| `RAG_IDLE_SECONDS` | `120` | Before the embedding model is unloaded |
| `RAG_MAX_FILE_BYTES` | `20000000` | Largest file indexed |

Model paths, ports, contexts, threads, RAM thresholds and the router's
fast/strong pair live in [`models.json`](models.json).

---

## 14. Tests

```bash
cd "C:\Users\SHAMI\HAKIM\AI\Hakim Local Agent" && .venv\Scripts\python -m unittest discover -s tests -t .
```

**485 tests, no model server needed, and none of them touch the network.**
They run in about 35 seconds.

| File | Covers |
|
The document-search tests use a fake embedder, so the suite stays fast and
needs no model on disk. The three that load BGE for real are skipped unless
you ask for them:

```bash
set RAG_MODEL_TESTS=1
.venv\Scripts\python -m unittest tests.test_rag.RealModelTests
```

They take about 140 s, almost all of it importing torch.

---|---|
| `test_agent.py` | Loop: plain replies, one call, several calls, tool errors, iteration limit, malformed replies |
| `test_tools.py` | Calculator, workspace jail, OCR validation, registry |
| `test_python_tool.py` | Restricted execution; spawns real child processes |
| `test_streaming.py` | SSE parsing, tool-call fragment assembly, reasoning suppression |
| `test_manager.py` | Start, stop, switch, adopt, crash recovery, idle unload |
| `test_router.py` | Routing decisions, no-downgrade rule, scoring |
| `test_chat_store.py` | History round-trip, ordering, deletion, corrupt JSON |
| `test_ocr.py` | Validation, request shape, capability checks, error paths |
| `test_shell_tool.py` | Allowlist, chaining, dangerous options, path confinement |
| `test_http_tool.py` | Host/scheme allowlist, redirect refusal, method gating |
| `test_port_reclaim.py` | Reclaiming a port from a llama-server we did not start |
| `test_turns.py` | The queue: serialisation, positions, bounded backlog, a runner that raises |
| `test_remote.py` | Hosted models: registry shape, key handling, connectivity cache, error hints |
| `test_api.py` | Every endpoint, the SSE event sequence, routing, reasoning suppression, tool switches, remote consent, uploads |

The API tests use the same manager harness as the model tests and a scripted
chat client, so none of them depend on whether something happens to be
listening on a port — which has produced false greens here before.
| `test_file_writes.py` | Writing, overwrite gating, self-protection |
| `test_python_scripts.py` | Script files in both modes, and the workspace guard |
| `test_git_tool.py` | Real throwaway repositories; write gating |
| `test_memory.py` | Store, recall, forget |

The manager tests stub only `_spawn` and `_healthy` — the OS boundary. Everything
above it is the real implementation.

---

## 15. Security posture

Every tool call from the model is treated as untrusted input.

**The model cannot:** write, rename or delete files; reach the network; read
outside the workspace; or see your environment variables. It can run terminal
commands only if you enable the terminal tool, and then only the read-only ones
on its allowlist.

**Boundaries:**

- Calculator: AST whitelist, no `eval`
- Filesystem: resolve-then-check containment, read-only
- Python: off by default, separate process, honestly documented as *not* a sandbox
- Terminal: off by default, allowlisted programs, no shell interpretation
- HTTP: off by default, loopback-only allowlist, redirects not followed
- Writes: off by default; create only, and the agent's own source is refused
- Git: off by default; no push, and nothing that discards uncommitted work
- Registry: validates arguments, converts every failure into a message for the
  model rather than a crash
- Front end: no `dangerouslySetInnerHTML` anywhere, so model output cannot
  become markup; links are restricted to `http(s)`
- Uploads: extension allowlist, size enforced while reading, filenames rebuilt
  rather than sanitised, and written only inside the workspace
- API keys: held in a git-ignored `.env`, never stored on a model spec, never
  sent to the browser, and scrubbed from provider error bodies

### The API changes this, and it is worth being exact about how

With tools enabled, the API can write files, run allowlisted commands and
execute Python. That is a **remote code execution surface** if it is reachable,
so:

- **It binds to `127.0.0.1`, stated explicitly in the run command** rather than
  left to a default someone could helpfully "fix" later. Do not put it on
  `0.0.0.0`.
- **There is no CORS middleware at all.** Vite proxies `/api` in development
  and FastAPI serves the built app in production, so both are same-origin and
  none is needed. Permissive CORS would let any page you happened to have open
  drive your agent.
- There is no authentication, because loopback binding on a single-user
  machine is the boundary. If that ever stops being true, this needs auth
  before it needs anything else.

**Tool switches are a deliberate loosening.** The flags were environment
variables so that enabling one was a considered act taken before startup; the
UI makes it a click. Nothing underneath moved — allowlists, workspace jail, no
delete or rename, and the Python and terminal tools are still not sandboxes.
Overrides are held in memory only, so a restart returns to whatever the
environment says and a switch cannot quietly become permanent.

---

## 16. What is verified and what is not

Being straight about this, because the difference matters.

### Verified against the live model

- **Tool calling works.** The server returned exactly the expected shape:
  `{"name":"calculate","arguments":"{\"expression\": \"sqrt(144) + 25**2\"}"}`
- **A full agent turn works end-to-end** through the web UI: tool call → result
  → final answer, **637**, correct, in 285.7 s
- **OCR transcribes correctly** against the real GLM-OCR server: every field of
  a test note right, ~30 s per image
- **Ministral 3B loads and emits tool calls**, and the RAM guard warned rather
  than refused at 518 MB free, exactly as intended
- **The agent chooses OCR by itself.** Qwen3.5 2B XS called `ocr_image` with a
  prompt it wrote, read the result and answered "HK-4127-B, 36 crates" in 89 s
- **The model manager starts models for real**, not just against a fake process
  layer
- **A full turn through the React UI and the API**, against Ministral on 8084:
  `17 * 43 - 209` → `calculate` tool call → **522**, correct, 127.1 s. Every
  stage arrived as its own event, and the answer was stored with its tool list
  and elapsed time.
- **History is written by real model turns.** Conversation 10 was created,
  titled from the prompt, and holds the user message and the answer with
  `model_key=mistral`, `elapsed=127.1`, `tools=['calculate']`.
- **The manager adopts a server it did not start.** The llama-server already
  running on 8084 was reported `ready, adopted` rather than fought over.
- **Gemini 3.5 Flash answers through the whole stack in 3.7 s** — browser, API,
  queue, agent loop, calculator tool — against 127.1 s for the same turn on
  local Ministral.
- **Upload plus OCR works end to end.** `samples/note.png` uploaded through the
  API, the model called `ocr_image` with the returned workspace path, GLM-OCR
  transcribed it, and the answer carried every field: HK-4127-B, Naggalama
  Depot, 36 crates, T. Balunywa. 223.2 s on the 2B.
- **Provider errors are accurate.** A Cerebras 402 and a Gemini 429 were both
  reported against the right provider with the right hint.

### Verified without the model

- 485 tests
- The React app against the real API: conversation list, tool roster with its
  real disabled reasons, model list, theme in both schemes, no sideways scroll
- **Tool switches, in the browser.** Turning Python on took the roster from 3
  tools to 5 and offered `run_python` and `run_python_file` to the next turn;
  turning it off withdrew them
- `GET /api/models` latency, before and after the fix: **18.7 s → 1.0 s**
- Terminal tool: ran `git` for real in a scratch repository and confirmed that
  `&&` and `;` chaining does not execute a second command

### Known account limits, not bugs

- **Cerebras returns HTTP 402.** The key authenticates and lists models
  (`gpt-oss-120b`, `gemma-4-31b`), but the account has no inference quota.
- **Gemini's free tier runs out quickly.** A handful of turns produced a 429.
  Both are reported with a hint rather than as a failure of this code.

### Not verified

- **Reasoning has never been seen from a live thinking model.** The plumbing is
  covered by tests using a scripted client — the event streams, it is not
  stored, and it is not replayed into the next prompt — but no Qwen3 model has
  been run with thinking on to watch a real trace arrive. That needs the 8B
  loaded, which evicts whatever else is resident.
- **Auto-routing has never actually switched a model live**, and with it the
  hosted-model consent dialog has only been exercised by tests — verifying
  Gemini meant selecting it directly, which by design skips the prompt.
- **The queue has never been exercised by two concurrent real turns.** Its
  behaviour is covered by tests, including that only one runs at a time.
- **The offline fallback has only been tested with a stubbed network.**

---

## 17. Troubleshooting

**"Model offline"** — nothing is listening on that port. Click **Load**, or send
a message and it starts automatically.

**`QwenTimeoutError`** — generation exceeded `AGENT_REQUEST_TIMEOUT`. On this
machine that means something is badly wrong, since the default is 20 minutes.
Check free RAM first.

**Refuses to start a model** — free RAM is below the model's `min_free_mb`.
Close something, or pick a smaller model. Thresholds are in `models.json`.

**Everything is extremely slow** — check free RAM. If the model does not fit, it
pages from disk on every token. `reasoning` needs ~6.2 GB free on an 8 GB
machine, which is tight. Also confirm `-t` matches your 4 logical processors.

**Front-end edits do not appear** — Vite hot-reloads `web/` on save. Changes to
anything under `api/` need the **uvicorn process** restarted, since it runs
without `--reload` on purpose (see [section 4](#4-running-it)).

**Orphaned `llama-server.exe`** — the API stops the models it started when it
shuts down. Killing it in a way that skips that handler, or running it with
`--reload`, leaves one holding gigabytes. Check with
`netstat -ano | findstr :808` and stop it by PID.

**Tool calling stops working** — check the server was started with `--jinja`
(default on build 10373). Without it, llama.cpp does not parse tool calls and
Qwen emits raw `<tool_call>` blocks inside `content`.

---

## 18. Roadmap

**Next, in order of value:**

1. Load a model through the manager for real — the largest untested gap
2. Confirm the filesystem tool with a live turn
3. Watch auto-routing perform a real switch

**Later:**

- Web search and `fetch_url`, once you decide the network boundary is worth
    crossing
- Deciding whether memories should be injected into the prompt automatically,
  which costs tokens on every turn
- Web search tool (`tools/web.py` is a placeholder)
- Widening the terminal allowlist as specific needs appear
- Real sandboxing for the Python tool (container or VM)
- Conversation search and export
- The custom C99 inference engine at `C:\Users\SHAMI\mmengine` — parked until
  it is further along

---

*Built with llama.cpp, Python 3.11, FastAPI, React and no agent frameworks.*
