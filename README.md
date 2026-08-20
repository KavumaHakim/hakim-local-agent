# Hakim AI System

A local AI agent built on llama.cpp. Runs entirely on your machine — no API keys,
no network calls, nothing leaves the box.

Two front ends over one engine: a Streamlit chat UI and a terminal CLI. Both use
the same agent, the same tools and the same model manager.

---

## Contents

1. [What it does](#1-what-it-does)
2. [Hardware reality](#2-hardware-reality)
3. [Setup](#3-setup)
4. [Running it](#4-running-it)
5. [Project layout](#5-project-layout)
6. [Architecture](#6-architecture)
7. [Models and switching](#7-models-and-switching)
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

- Chats with a local Qwen model served by `llama-server`
- Calls tools when they help: exact arithmetic, reading files in a workspace
- Manages three models, holding only one in RAM at a time
- Optionally picks the model itself based on how hard the prompt looks
- Streams tokens as they generate
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

## 3. Setup

### Prerequisites

- Python 3.11+
- `llama-server.exe` from llama.cpp — this project was verified against
  **build 10373**
- GGUF model files

### Install

```bash
cd "C:\Users\SHAMI\HAKIM\AI\local-agent"
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Dependencies are only `requests` and `streamlit`. Everything else is standard
library.

### Point it at your files

Edit [`models.json`](models.json):

```json
{
  "server_exe": "C:\\Users\\SHAMI\\HAKIM\\AI\\LLAMA CP\\llama-server.exe",
  "models_dir": "C:/Users/SHAMI/HAKIM/AI/local-agent/weights"
}
```

No model paths are hardcoded anywhere in the Python.

---

## 4. Running it

### Web UI

```bash
cd "C:\Users\SHAMI\HAKIM\AI\local-agent" && .venv\Scripts\streamlit run app.py
```

Opens on <http://localhost:8501>. Pick a model in the sidebar, click **Load**,
then chat. The model server starts automatically on your first message if it
is not already up.

### Terminal

```bash
cd "C:\Users\SHAMI\HAKIM\AI\local-agent" && .venv\Scripts\python main.py
```

### Starting a model server by hand

You never have to — the manager does it — but if you want to:

```bash
"C:\Users\SHAMI\HAKIM\AI\LLAMA CP\llama-server.exe" -m "C:\Users\SHAMI\HAKIM\AI\local-agent\weights\Ministral-3-3B-Instruct-2512-Q4_K_M.gguf" --jinja -c 4096 -t 4 -np 1 --port 8084
```

If a healthy server is already on a model's port, the manager **adopts** it
rather than starting a rival.

> **Note on `-t 4`:** your CPU has 4 logical processors. The original
> `runner.bat` passed `-t 8`, which oversubscribes and slows things down.

---

## 5. Project layout

```
local-agent/
├── app.py               Streamlit chat UI
├── main.py              terminal CLI
├── config.py            all settings, environment-driven
├── models.json          model registry (paths, ports, RAM thresholds)
├── chat_store.py        SQLite conversation history
├── memory_store.py      durable facts, same SQLite file
│
├── ui_style.py          stylesheet + HTML fragments for the web UI
├── ui_commands.py       slash commands + the dropdown/toggle scripts
├── requirements.txt
├── .gitignore           keeps weights/ and data/ out of any repo
│
├── weights/             the GGUF model files (~9.5 GB, git-ignored)
├── data/                generated state: the SQLite database
├── samples/             example images for the OCR tool
│
├── agent/
│   ├── loop.py          the reasoning/action cycle
│   ├── parser.py        assistant message → tool calls
│   ├── prompts.py       system prompt
│   └── router.py        picks a model from the prompt
│
├── models/
│   ├── qwen.py          HTTP client (the only module that speaks HTTP)
│   └── manager.py       starts/stops/switches llama-server processes
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
│   ├── memory_tool.py   durable facts across conversations
│   ├── ocr_tool.py      GLM-OCR client (working)
│   └── web.py           placeholder
│
└── tests/               388 tests, no server required
```

---

## 6. Architecture

```
User
 → Agent            (agent/loop.py)
 → Qwen             (models/qwen.py → llama-server)
 → tool call
 → Tool Registry    (tools/base.py)
 → tool execution
 → tool result
 → Qwen
 → final response
```

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

**Reasoning is never replayed or displayed.** llama.cpp returns thinking in
`reasoning_content`, separate from `content`. The agent stores only `content`
and `tool_calls`; the streaming client counts reasoning characters but discards
the text.

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


### Why in-process, not a separate service

The plan this was built from suggested a FastAPI service on port 9000 proxying
to the model ports. That was not built, for two reasons:

1. Every caller is already Python. A second daemon and another port to keep
   alive buys nothing for a single-user local tool.
2. The venv already carries a `fastapi` ↔ `starlette` version conflict.

The manager is a class you import. Wrapping it in HTTP later is a small change,
and it is written so that stays true.

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
| `remember`, `recall`, `forget` | memory | **off by default** |
| `http_request` | http | **off by default** |
| `ocr_image` | ocr | **works** — off until you run the OCR server |

Disabled tools are not registered at all — sending the model a definition it can
only fail on wastes context and a whole round-trip.

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

### ocr — working

Reads text from images through a second llama-server running GLM-OCR.

**Verified end-to-end.** A generated delivery note was transcribed with every
field correct — reference `HK-4127-B`, recipient, crate count and signature — in
**37 s** first run, ~28 s after. A four-row table came back with all figures
intact.

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
"C:\Users\SHAMI\HAKIM\AI\LLAMA CP\llama-server.exe" -m "C:\Users\SHAMI\HAKIM\AI\local-agent\weights\GLM-OCR-Q8_0.gguf" --mmproj "C:\Users\SHAMI\HAKIM\AI\local-agent\weights\mmproj-GLM-OCR-Q8_0.gguf" -c 4096 -t 4 -np 1 --port 8081
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

---

## 10. Chat history

[`chat_store.py`](chat_store.py) — SQLite, standard library, no new dependency.

Two tables. `conversations` holds title, model, timestamps. `messages` holds
role, content, tool calls as JSON, elapsed time. Tool calls are display metadata
rather than conversation state, so they do not earn their own table.

A connection is opened per operation rather than held open — Streamlit reruns on
background threads, and a shared `sqlite3` connection is not safe across them.

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

- User turns are right-aligned violet bubbles; replies sit flat and full-width
- Tool activity appears as rounded pills with green/red status dots
- Animated typing dots until the first token arrives
- Empty state with four clickable starter prompts
- Sidebar: model picker and status stay visible; **Settings**, **Tools** and
  **History** fold into collapsible sections so the panel stays short
- **Hide panel / Show panel** button, top right

### Two Streamlit details worth remembering

**Role detection.** Custom emoji avatars render as a bare `<div>` whose only
distinguishing class is a build-specific emotion hash. The stable hook is
Streamlit's own `aria-label="Chat message from user"`.

**Layering.** Streamlit's `stHeader` sits at `z-index: 999990` with an opaque
background covering the top 60 px. Anything you pin up there must clear it —
this is why the sidebar toggle was invisible until its z-index was raised above
that.

Also: `st.components.v1.html` is deprecated (removal dated 2026-06-01) — use
`st.iframe`, which also grants the same-origin access the scripts need. Note
`height=0` renders nothing; use `height=1`.

Both browser scripts degrade safely: if the frame cannot reach the parent
document they do nothing, and every command still works typed in full.

---

## 12. Commands

Type `/` in the chat box and a dropdown appears. ↑/↓ to move, Enter or Tab to
accept, Esc to dismiss, or click.

| Command | Does |
|---|---|
| `/help` | Show the commands |
| `/models` | List models and which is loaded |
| `/model <key>` | Switch model |
| `/unload` | Unload the current model, free its RAM |
| `/tools` | List available tools |
| `/auto` | Toggle automatic routing |
| `/clear` | Start a new conversation |

The CLI has the same set plus `/quit`.

---

## 13. Configuration

Everything is environment-driven with local defaults. See [`config.py`](config.py).

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

Model paths, ports, contexts, threads, RAM thresholds and the router's
fast/strong pair live in [`models.json`](models.json).

---

## 14. Tests

```bash
cd "C:\Users\SHAMI\HAKIM\AI\local-agent" && .venv\Scripts\python -m unittest discover -s tests -t .
```

**388 tests, no model server needed.** They run in about 10 seconds.

| File | Covers |
|---|---|
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
- Web UI binds to localhost; the agent can read your workspace, so do not expose
  it to a network you do not control

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

### Verified without the model

- 388 tests
- Model registry resolves all three files; RAM probe reads correctly
- Router decisions
- Slash dropdown: filtering, positioning, hide rules
- Terminal tool: ran `git` for real in a scratch repository and confirmed that
  `&&` and `;` chaining does not execute a second command
- Sidebar toggle in expanded and collapsed states
- History: write, read, list, delete, and rendering in the sidebar

### Not verified

- **Auto-routing has never actually switched a model live.**
- **History has never been written by a real model turn** — only by direct
  store calls.

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

**Sidebar edits do not appear** — Streamlit caches imported modules. Changes to
`ui_style.py` or `ui_commands.py` need the Streamlit **process** restarted, not
just a page reload.

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

*Built with llama.cpp, Python 3.11, Streamlit and no agent frameworks.*
