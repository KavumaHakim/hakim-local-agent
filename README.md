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
- Takes a message by voice, and reads any answer back aloud — both on this
  machine, and both optional
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

### The integrated GPU helps the prompt, not the tokens

The same reasoning predicts what an integrated GPU can and cannot do, and it
was worth checking rather than assuming. `llama-bench`, Qwen3.5-2B Q3_K_S, the
**same Vulkan binary in every row** so that offloading is the only thing that
changes:

```
                prompt (pp128)      generation (tg32)
-ngl 0           19.40 / 14.67        2.51 / 2.03
-ngl 16               18.96                1.45
-ngl 99          22.57 / 21.01        2.44 / 2.45
```

Two runs where there are two numbers. Averaged: prompt processing **17.0 →
21.8 tok/s, about a quarter faster**. Generation **2.27 → 2.45**, which is
inside the noise and is not a result.

That is the memory-bandwidth ceiling again. The HD 520 reports `uma: 1` — it
has no memory of its own and reads the same DIMMs through the same controller,
so there is nothing for it to win at generation. Prompt processing is the
compute-bound half, and that is exactly the half that improves.

**Partial offload is the worst of the three.** `-ngl 16` generated at 1.45
tok/s, 40% *slower* than putting nothing on the GPU at all: the activations
cross the boundary at every handover and none of it is won back on shared
memory. It is all or nothing.

> Honest about the noise: CPU-only measured 19.40 in one run and 14.67 in the
> other, on identical settings. The machine had 504 MB free against a 694 MB
> model and was paging. The prompt-processing gain is consistent enough in
> direction and size to rely on; the generation numbers are not worth reading
> to two decimal places.

**It is not free on RAM**, which is the easy thing to get wrong. The device
has no memory of its own, so a weight handed to the GPU is still in the same
DIMMs, and the backend keeps staging buffers besides. Peak resident, same
binary, same model:

```
                    -ngl 0      -ngl 99
738 MB model         944 MB     1,169 MB
1,073 MB model     1,258 MB     1,540 MB
```

It does not double — but it is +225 MB and +282 MB, and two points fit about
**100 MB flat plus a sixth of the file**. That is what the RAM guard now
charges a model that offloads, because otherwise it asks for the same free
memory whether or not it is offloading and under-states by a couple of hundred
megabytes on a machine with eight gigabytes. Two points are two points: it is
a straight line through them, not a law, and like the rest of this sizing it
is deliberately the kind of over-estimate you can ignore.

Extrapolated, that is roughly +450 MB for a 2 GB model and +900 MB for the 8B
— which settles it for the large ones. **Offload the small models; leave the
8B alone.** It already wants 6.2 GB.

Turn it on with **GPU layers** in the model's tuner. It does nothing unless
`Server` (just below the model list in Settings) points at a build with a GPU
backend compiled in, which is why the tuner names that binary underneath the
field.

### Which llama-server runs

`Server` in Settings, under the model list. One binary runs every model, so it
is not a per-model setting.

It is kept in `data/models.local.json` rather than `models.json`, because it is
a property of one computer and `models.json` is in version control — a path
committed from somebody's laptop is wrong on every other machine. Empty goes
back to searching: the configured path, then `vendor/llama`, then PATH.
`vendor/llama-vulkan` is **never** searched, so an accelerator build is only
ever used deliberately.

Changing it takes effect on the next model start. A running server keeps the
binary it was started from, which is the honest behaviour — it is that process,
and it cannot become another one.

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

### Hitting the context window

A 4,096-token model has less room than it sounds. The system prompt and tool
schemas take about 1,080 tokens before anything happens, leaving roughly 3,000
for the question, the conversation and the answer.

Two things used to overrun that, and both are now bounded against the model's
own context rather than a fixed number:

| | limit |
|---|---|
| one tool result | `AGENT_TOOL_RESULT_SHARE`, a quarter of the context |
| all of a turn's results together | whatever is left of the history budget |
| the conversation replayed | `AGENT_HISTORY_SHARE`, half of it |

On a 4,096-token model that is 3,348 and 6,696 characters. A model with a
larger window gets proportionally more without editing anything.

This matters most for **images**. A dense scanned page OCRs to six or seven
thousand characters, which on its own is nearly twice what a 4,096-token model
can spare - so a single image used to fail the whole turn rather than part of
one result. It is now cut, and the cut is announced in the result the model
reads:

```json
{"text": "...", "characters": 6930,
 "truncated": "'text' was cut: 3,931 of 6,930 characters are not shown,
               because the whole result does not fit this model's context.
               Say so rather than treating this as complete."}
```

The wording is deliberate. A model handed the first half of a page with no
indication summarises it as though it were the whole page, which is worse than
an error - it is a confident wrong answer.

History is trimmed by size as well as by message count, and trimmed **before**
the request rather than after it. `max_history_messages` counts messages,
which says nothing about how much context they occupy: sixty short exchanges
fit comfortably and three pages of OCR do not.

If you are losing text you need, the fix is more context rather than a bigger
share of the same context. Raise the model's `context` in Settings - the tuner
shows what the KV cache will cost before you commit to it - or use a model
with a larger window for that work.

### So, to make it faster

1. **Turn off tools you are not using.** Biggest single lever, and it is a
   switch in the sidebar.
2. **Keep the conversation going** rather than starting a new one. The prefix
   is cached; a new conversation pays it again.
3. **Turn off Thinking** unless the question needs it. It does not change
   tok/s, it changes how many tokens are generated — hundreds of them, at
   2 tok/s.
4. **Mind the idle timeout.** `idle_timeout_seconds` in `models.json` is 1800.
   When the model unloads, its KV cache goes with it, so the next message pays
   both the reload and the full prefix again. Half an hour is long enough to
   sit through a working pause and short enough that a model you have finished
   with gives its RAM back. Set it to 0 to disable unloading entirely, but on
   8 GB that leaves a large model resident forever.
5. **Use a smaller model.** At 2 tok/s the model is the ceiling, and the 8B is
   far slower still.
6. **Offload to an integrated GPU, if you have one and a build that can reach
   it.** Worth about a quarter off prompt processing, which is what the tool
   prefix costs. It does nothing for generation, so it will not make answers
   appear faster once they start.

---

## 3. Setup

Windows and Linux both. macOS should work — same code paths as Linux — but has
not been tested, so it is not claimed.

### The short version

```bash
git clone https://github.com/KavumaHakim/hakim-local-agent.git
cd hakim-local-agent
```

**Windows**

```bash
setup.bat
```

**Linux / macOS**

```bash
./setup.sh
```

Run in a terminal it is a walkthrough: it says what it is about to do, asks
what you want, and shows progress while it works.

```
  ██   ██   █████   ██  ██   ██  ███   ███
  ██   ██  ██   ██  ██ ██    ██  ████ ████
  ███████  ███████  █████    ██  ██ ███ ██
  ██   ██  ██   ██  ██ ██    ██  ██  █  ██
  ██   ██  ██   ██  ██  ██   ██  ██     ██
  a local agent that runs on your own machine

  What would you like?

   [+] 1  Get llama.cpp for me
          18 MB. It is the engine that actually runs your models.
   [ ] 2  Let me talk to it, and hear it back
          220 MB. Dictate a message, and read any answer aloud.
   [ ] 3  Let it search my documents
          2 GB and slow to install. You can add this later.
   [ ] 4  Build the web interface for production
          Most people want the development mode instead. Leave this off.
   [+] 5  Check it works when you are done
          Runs the tests. About a minute, and worth it.

  number to change one, Enter when it looks right  >
```

The checklist **redraws in place** — pressing a number replaces it rather than
printing it again underneath — and it clears itself once accepted. Then:

```
Here is the plan
   1. Checking your Python        6. Looking for a model
   2. Making a private environment 7. Setting up the voice
   3. Installing the Python side  8. Writing your configuration
   4. Installing the web interface 9. Making sure it all works
   5. Fetching llama.cpp

===------  3/9 Installing the Python side
  / requirements.txt (14s)

====-----  5/9 Fetching llama.cpp
  downloading [██████████████..........]  58% 10.8/18.4 MB  2.1 MB/s
```

**The only thing left for you is a model.** Drop any `.gguf` into `weights/`
and it is picked up on the next scan, sized from its own header.

It does not just exit when it finishes. It prints where everything went, what
was and was not installed, and how to start — then waits for you to close it:

```
-- where everything is -------------------------------------------------

   project          /home/you/hakim-local-agent
   python           /home/you/hakim-local-agent/.venv/bin/python
   llama.cpp        /home/you/hakim-local-agent/vendor/llama/build/bin/llama-server
   models           /home/you/hakim-local-agent/weights
                    - Ministral-3-3B-Instruct-Q4_K_M.gguf (2.1 GB)
   settings         /home/you/hakim-local-agent/.env
   your data        /home/you/hakim-local-agent/data
   workspace        /home/you/hakim-local-agent

-- what was set up ------------------------------------------------------

   Python packages  installed into .venv, nothing system-wide
   Document search  skipped - add it with --with-rag
   Web interface    ready for development mode
   llama.cpp        ready
   A model          1 found
   Hosted models    none - everything runs locally

-- how to start it ------------------------------------------------------

   Everything it needs is in place.

   ./start.sh   starts both servers and opens the browser
   ...

  All done - press Enter to close
```

The hosted-model line reports a **count**, never a key. And the pause is
skipped entirely without a terminal, or with `--yes`: a setup script waiting
on a keypress in CI never finishes, which is worse than one that scrolls
past.

**Piped, redirected or in CI it asks nothing**, takes the defaults, and prints
plain lines — no cursor tricks, no carriage returns, no prompt waiting for
input that will never come. `--yes` forces that mode in a terminal too.

| Flag | Effect |
|---|---|
| `-y`, `--yes` | ask nothing; take the defaults and the flags below |
| `--with-rag` | also install document search (torch, about 2 GB) |
| `--build-web` | build the UI instead of running Vite in development |
| `--no-llama` | do not download llama.cpp |
| `--skip-tests` | do not run the verification tests |

The toolkit behind it is [`scripts/ui.py`](scripts/ui.py), which is standard
library only and has to be: setup runs *before* anything is installed, so
`rich`, `tqdm` and `questionary` are unavailable to it by definition.

Everything in it degrades rather than decorating:

| Where it runs | What happens |
|---|---|
| A terminal with 256 colours | Violet accent, matching the web UI |
| A terminal with eight | Cyan accent instead — no approximating |
| `NO_COLOR` or `TERM=dumb` | No colour at all |
| A console that cannot encode `█` | The banner falls back to `#` |
| Narrower than the banner | The banner falls back to the plain word |
| No cursor control | The menu prints again instead of redrawing |
| A pipe or CI | No prompts, no redraws, no carriage returns |

It reads whole lines rather than single keys. Arrow-key menus need raw
terminal mode — `msvcrt` on Windows against `termios` elsewhere — and both
have edge cases in the terminals people actually run setup in; a menu you
cannot answer is worse than one that looks plainer.

It is safe to run again at any time; nothing it does is destructive.

### Step by step, if you would rather do it by hand

**1. Prerequisites.**

| Needed | Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.11, 3.12, 3.13 and 3.14 all tested here |
| Node | 20+ | only for the web UI; the terminal client works without it |
| `llama-server` | build 10373 verified | from llama.cpp — **fetched by the setup script** |
| A GGUF model | any | **the one thing you supply** |

On Debian or Ubuntu, Python needs its venv module separately:

```bash
sudo apt install python3 python3-venv nodejs npm
```

**2. Create the virtualenv and install.**

Windows:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
```

Linux / macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
```

`requirements.txt` is four packages: `requests`, `fastapi`, `uvicorn` and
`python-multipart`. Everything else is standard library.
`requirements-dev.txt` is only needed to run the tests.

**3. Install the front end.** Skip this if you only want the terminal client.

```bash
npm --prefix web install
```

**4. Get `llama-server`.** The setup script does this for you; by hand it is:

```bash
python scripts/get_llama.py
```

It lands in `vendor/llama/`, which is git-ignored. `--build b10731` pins a
release, `--force` re-downloads, and `--list` shows what would be fetched
without fetching it. It resumes a dropped connection the same way the model
browser does, over both transports — `requests` when it is installed, urllib
when nothing is yet, which is exactly the machine running this script.

**4b. Optional: dictation and a voice.** Neither is needed to chat, and
neither is switched on anywhere — the microphone and the speaker are simply
not drawn until what they need is present.

```bash
python scripts/get_speech.py
```

About 220 MB: an 8 MB whisper.cpp build into `vendor/whisper/`, a 148 MB
speech model into `whisper/`, and a 63 MB Piper voice into `tts/`. `--what
whisper|model|voice` fetches one part, `--model tiny.en` takes the 78 MB model
instead, and `--voice` picks a different voice.

Text-to-speech needs no binary — `piper-tts` is a wheel in `requirements.txt`,
already installed by step 2 — so a voice file is the whole of it. Speech-to-text
needs a platform build, which is why this script exists at all.

> **macOS has no whisper.cpp binary release** — the project ships an
> xcframework for Xcode and expects everyone else to build it. `brew install
> whisper-cpp` and everything here works unchanged; the script says so rather
> than reporting a confusing "not found". Reading aloud is unaffected.

`--backend vulkan` fetches the Vulkan build instead, into `vendor/llama-vulkan/`
**beside** the CPU one rather than replacing it — because the only way to know
whether an integrated GPU helps is to measure both on the same machine. Only
`vendor/llama/` is searched automatically, so the accelerator build has to be
pointed at deliberately (`setup.py`'s "I already have it", which remembers the
path). An accelerator build that cannot reach its device is slower than the CPU
one, not faster, which is why it is never picked up by accident.

Having fetched it, point **Server** at it in Settings and set **GPU layers**
on a model. Both steps are needed: the binary decides whether a GPU can be
reached at all, the number decides whether anything is sent to it.

It was worth trying, and the result was narrow: on this machine full offload
took **a quarter off prompt processing and nothing off generation**, and a
partial split was slower than either end. The measurement, and why the shape of
it was predictable from memory bandwidth, is
[in §2a](#the-integrated-gpu-helps-the-prompt-not-the-tokens).

Three details, because llama.cpp's releases are not arranged the way you would
guess:

- **The binary releases are prereleases**, so `/releases/latest` returns a tag
  with no binaries at all. The release list has to be walked until one with
  assets turns up.
- **There is a build per platform per accelerator** — 27 assets in a typical
  release. Only the plain CPU build is wanted; a CUDA or ROCm build is ten
  times the size and needs a runtime that is not installed here.
- **No checksums are published** alongside these archives, so there is nothing
  to verify them against. What is done instead: HTTPS to GitHub, the download
  must match the advertised byte count, the archive must open, it must contain
  a `llama-server`, and that binary must answer `--version`. That is weaker
  than a signature, and it is said here rather than implied.

**If you already have one somewhere**, setup asks rather than downloading a
second copy:

```
  [!]  I could not find llama.cpp on this machine.

  What would you like to do?
   * 1) Download it for me
        About 18 MB, straight from the project.
     2) I already have it
        Tell me where, and I will remember.
     3) Skip for now
        Nothing local will run until it is sorted.
```

Point it at the binary *or* the folder containing it — both work — and it is
checked by running `--version` before being believed. The path is then written
to `data/models.local.json`, which is git-ignored, **not** to `models.json`,
which is in version control: a path from one laptop is wrong on every other
machine.

Where it looks, in order:

| | Where | Why it is in this position |
|---|---|---|
| 1 | `server_exe` in `data/models.local.json` | You said so. Nothing should second-guess that |
| 2 | `server_exe` in `models.json` | The committed default, when it happens to exist |
| 3 | `vendor/llama/` | What setup downloaded |
| 4 | `PATH` | Whatever else is around |

A remembered path that has since been deleted is ignored rather than honoured,
so removing the binary falls back to the search instead of failing. To point it
somewhere else by hand:

```json
{ "server_exe": "../llama.cpp/llama-server", "models_dir": "weights" }
```

**Paths in `models.json` are relative to that file**, not to the working
directory, so the project folder can be renamed or moved without editing
anything. That is not a style choice: they were absolute once, the folder was
renamed, and the hardcoded path was the one thing that broke.

**5. Get a model.** The one thing the setup script does not fetch for you.
Drop any `.gguf` into `weights/` and it is picked up on the next scan — sized
from its own GGUF header, no configuration needed.

On 8 GB of RAM, a 2–3B instruct model at `Q4_K_M` is the sensible starting
point. For the arithmetic behind that, and what fits on other machines, see
[Choosing a model for your hardware](#choosing-a-model-for-your-hardware).

**6. Optional: API keys.** Only for hosted models; the agent is fully local
without them. Setup offers to take them:

```
  These are optional. Everything works locally without them;
  a key just makes that provider's model selectable too.

  Add a hosted model API key? [y/N] y
  Gemini 3.5 Flash (GEMINI_API_KEY) (hidden, Enter to skip) >
```

Typed with **no echo** — an API key pasted in the open stays in the terminal
buffer, the scrollback and any screen recording — and never printed back
afterwards, not even the last few characters. They are written to `.env`,
replacing the commented placeholder rather than being appended below it, and
`.env` is git-ignored.

Which providers are offered comes from `models.json`, so adding a hosted entry
there is enough for setup to start asking about its key. A key already set, in
`.env` or in the environment, is not asked for again.

By hand it is `cp .env.example .env` and fill in what you need.

**7. Check it.**

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

1140 tests, no model server needed, and none of them touch the network.

### What the setup script deliberately does not do

**It never downloads a model.** Which model to run is the one genuinely
personal decision in this project — it depends on your RAM, your language, and
what you want the agent for — and several gigabytes is not something a setup
script should pull over someone's connection on their behalf. It names what is
missing and points at where to look.

It does fetch `llama.cpp`, because that choice is not personal: there is
exactly one right build for a given machine, it is 18 MB, and getting it by
hand means picking correctly out of 27 similarly-named archives.

The speech model and the voice sit between those two cases, which is why they
are **off by default and offered rather than assumed**. They are a choice —
`tiny.en` against `base.en`, one voice against another — but a small one with
an obvious default, and getting them by hand means the same 9-archive problem
whisper.cpp has. So the menu asks, the answer is remembered in what it fetched,
and nothing is downloaded for somebody who only wanted to type.

### If something goes wrong

| Symptom | Cause |
|---|---|
| `/usr/bin/env: 'bash\r'` | The scripts are committed with LF via `.gitattributes`; this means a checkout converted them. Re-clone, or `dos2unix setup.sh` |
| `Permission denied: ./setup.sh` | `chmod +x setup.sh start.sh`, or just run `bash setup.sh` |
| `ensurepip is not available` | Debian/Ubuntu split it out: `sudo apt install python3-venv` |
| `llama-server not found` | Not on `PATH` and not at the `server_exe` path. See step 4 |
| No models listed | No `.gguf` in `weights/`. See step 5 |
| No microphone in the composer | No whisper.cpp build or no speech model: `python scripts/get_speech.py` |
| No speaker on an answer | No Piper voice: `python scripts/get_speech.py --what voice` |
| Port 8000 or 5173 in use | An earlier run is still going. Both launchers name the process holding it |
| `GitHub is rate-limiting this connection` | Unauthenticated API calls are capped per IP. Wait, or fetch a build by hand and put it on `PATH` |

---

## 4. Running it

### Web UI

One command, either platform:

```bash
start.bat
```

```bash
./start.sh
```

Each starts both servers, checks the ports first, and names the process holding
one if it is busy. `start.bat` opens a window per server; `start.sh` runs them
in the foreground and stops both on Ctrl-C.

By hand it is two processes — the API, and the front end that talks to it:

```bash
.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```bash
npm --prefix web run dev
```

On Linux the interpreter is `.venv/bin/python` instead.

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
cd "C:\path\to\Hakim Local Agent" && .venv\Scripts\python main.py
```

### Starting a model server by hand

You never have to — the manager does it — but if you want to:

```bash
"C:\path\to\llama.cpp\llama-server.exe" -m "C:\path\to\Hakim Local Agent\weights\Ministral-3-3B-Instruct-2512-Q4_K_M.gguf" --jinja -c 4096 -t 4 -np 1 --port 8084
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
├── setup.bat/.sh        one-command install, both platforms
├── start.bat/.sh        run the API and the UI together
├── scripts/setup.py     what both setup wrappers actually run
├── scripts/ui.py        menus, spinners and progress bars, stdlib only
├── scripts/get_llama.py fetches the right llama.cpp build for this machine
├── scripts/get_speech.py fetches whisper.cpp, a speech model and a voice
├── vendor/llama/        that build, once fetched (git-ignored)
├── vendor/whisper/      a whisper.cpp build, if you want dictation (git-ignored)
├── whisper/             ggml-*.bin speech models (git-ignored)
├── tts/                 Piper voices, .onnx + .onnx.json (git-ignored)
│
├── api/                 the HTTP layer the front end talks to
│   ├── main.py          app, lifespan, static serving
│   ├── runtime.py       process-wide objects; runs one turn
│   ├── turns.py         the queue: one turn at a time, with positions
│   ├── schemas.py       request and response bodies
│   └── routes/          chat (SSE), conversations, models, hub, meta,
│                        uploads, speech, workspace, rag, memory
│
├── web/                 React + TypeScript + Vite + Tailwind
│   ├── src/lib/         api client, SSE reader, markdown, commands
│   ├── src/hooks/       the turn state machine and data hooks
│   ├── src/components/  sidebar, transcript, composer, palette,
│                        workspace picker
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
│   ├── hub.py           searching and downloading models from Hugging Face
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
├── speech/              dictation and reading aloud
│   ├── whisper.py       shells out to whisper-cli, one clip at a time
│   ├── piper.py         owns the voice worker: starts late, stops when idle
│   └── voice_worker.py  the Piper voice, in its own process
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
└── tests/               1140 tests, no server required
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

### Verified llama.cpp behaviour (builds 10373 and 10750)

- `--jinja` is **on by default**. The server applies Qwen3's chat template,
  parses the model's native tool syntax, and returns standard OpenAI
  `tool_calls`. This is why `parser.py` invents no protocol — it reads what the
  server already produces.
- Reasoning goes to `message.reasoning_content` under the default
  `--reasoning-format deepseek`.
- `chat_template_kwargs` is supported, which is how `enable_thinking` is sent.
- `--alias` is unset, so the server ignores the `model` field in requests.

All four were **re-checked against b10750** before that build was made the
default here, against a real server and a real model rather than by reading a
changelog: a tool call came back as a standard `tool_calls` entry naming the
function with its arguments, `chat_template_kwargs` was accepted, reasoning
arrived in `reasoning_content` separately from `content`, and a nonsense
`model` field was ignored rather than rejected. If you pin a different build,
this is the list worth re-running.

---

## 7. Models and switching

Defined in [`models.json`](models.json):

| Key | Model | File size | Port | Min free RAM |
|---|---|---|---|---|
| `gemma` | Gemma 4 E2B Q4_0 | 2709 MB | 8085 | 2627 MB |
| `mistral` | Ministral 3B Q4_K_M | 2047 MB | 8084 | 1900 MB |
| `fast` | Qwen3.5 2B (M) Q4_K_M | 1023 MB | 8080 | 1150 MB |
| `tiny` | Qwen3.5 2B (XS) Q3_K_S | 704 MB | 8083 | 900 MB |
| `reasoning` | Qwen3 8B Q4_K_M | 4794 MB | 8082 | 6200 MB |

**`gemma` is the default.** Measured on this machine: loads in **18 s** and
generates at **5.9–6.0 tok/s**, the fastest of the local models here — a
Ministral turn of the same shape runs slower. Its context is capped at 8192 of
the model's 131072, which is cheap because Gemma has a single KV head: that
cache is about 210 MB where Ministral's 4096 costs more. Its chat template
carries `tool_call` and `tool_response`, so llama.cpp parses tool calls from it.

It is the one model here big enough to be uncomfortable: 2709 MB against a
2627 MB threshold means the RAM guard warns on a typical desktop, and if
available RAM is far below that, the weights cannot stay resident and
generation slows to disk speed. `mistral` is the smaller fallback.

**`mistral` is the router's "fast" model.** Ministral 3B is Mistral AI's edge
model, not Mistral 7B. Measured: loads in **46 s**, and answered a tool-call
round in **35 s** against 252 s for the Qwen 8B. Its chat template carries
Mistral's native function-calling format (`AVAILABLE_TOOLS`, `TOOL_CALLS`) and
llama.cpp parses it into standard OpenAI `tool_calls`.

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
- **Unloads after idle** (`idle_timeout_seconds`, default 1800) to give RAM
  back. 0 disables it, and the model then stays loaded until it is unloaded or
  displaced.

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

### Finding one without leaving the app

**Settings → Find a model** searches Hugging Face for GGUF repositories, most
downloaded first — for any given model there are dozens of re-uploads, and the
popular one is overwhelmingly the complete, correctly converted one that is
still there next month.

Expanding a repository lists its quantisations smallest first, each with the
number that actually decides the matter:

```
IQ1_S     538 MB   ~0.8 GB to run
Q3_K_M    940 MB   ~1.1 GB to run
Q4_K_M    1.1 GB   ~1.3 GB to run
Q8_0      1.8 GB   ~2.0 GB to run
```

That figure is the arithmetic below, applied to the file size **before** the
file is downloaded, and compared against what is free right now. Being told a
4.8 GB file wants 6.2 GB free is worth more than any download speed.

Downloads resume, and that is not a nicety. A model is gigabytes and a
domestic connection is not reliable for that long; without `Range` a drop at
90% costs the whole thing, which on a slow link means a large model can never
finish at all. This was measured the hard way — fetching a 35 MB archive here
failed three times in a row, at 2 MB, then 20 MB, then part-way again. The
`.part` file is what gets resumed, so it survives between attempts and is
deleted only on cancellation or final failure.

Two server behaviours are handled rather than assumed: answering **200** to a
ranged request, which means the range was ignored and the whole file is coming
again — so what is on disk has to be thrown away rather than appended to — and
answering **416**, which means there was nothing left to send.

Downloads run on their own thread, so a two-hour fetch does not block a
conversation, and one at a time for the reason the model manager runs one
server at a time. Progress, speed and an estimate are polled once a second.
When one lands the catalogue is rescanned, so the model appears in the picker
without a reload.

What is enforced, because this reaches the network and writes to disk:

| | |
|---|---|
| Host | `huggingface.co` only, over HTTPS |
| Files | paths ending `.gguf` only |
| Name | rebuilt from the basename — `../../.ssh/x.gguf` saves as `x.gguf` |
| Disk | checked before any bytes move, keeping 500 MB spare |
| Atomicity | written to `.part`, renamed only when complete, so discovery never sees a half-model |
| Resume | a dropped connection is picked back up with `Range`, five tries, keeping the bytes already on disk |
| Gated repos | reported as gated; this app holds no credentials |

### Choosing a model for your hardware

The one decision setup leaves to you, so here is the arithmetic behind it.

**What a model actually costs in RAM.** This is the formula
`models/discovery.py` uses on every dropped-in `.gguf`, and it is measured
rather than guessed:

```
free RAM needed  =  file size x 0.8        resident weights
                 +  KV cache               grows with context
                 +  250 MB                 headroom for everything else
                 +  50% of the weights     only when weights exceed 3 GB
```

The **0.8** comes from watching GLM-OCR: a 906 MB file occupies 683 MB
resident, a ratio of 0.754, rounded up because under-estimating means letting
a model start that then thrashes. The **50% surcharge** on large models exists
because past about 3 GB a model stops fitting alongside the operating system,
starts paging from disk, and the formula stops describing it.

**Context is not free.** The KV cache is computed from the GGUF header —
layers x KV heads x head dimension x 2 x bytes. For GLM-OCR that is 52,224
bytes *per token*:

| Context | KV cache |
|---|---|
| 2048 | 102 MB |
| 4096 | 204 MB |
| 8192 | 408 MB |
| 16384 | 816 MB |

Which is why chat models here run at 4096 rather than the 32k or 128k they
were trained for. The ceiling is RAM, not the model.

**The models on this machine**, as a worked example. 8 GB total, of which
2–3 GB is realistically free:

| Model | File | Context | Needs free | Verdict on 8 GB |
|---|---|---|---|---|
| Qwen3.5 2B XS `Q3_K_S` | 0.7 GB | 4096 | 900 MB | comfortable |
| Qwen3.5 2B M `Q4_K_M` | 1.1 GB | 4096 | 1,150 MB | comfortable |
| Ministral 3B `Q4_K_M` | 2.1 GB | 4096 | 1,900 MB | the day-to-day choice |
| GLM-OCR `Q8_0` | 1.0 GB + 0.5 mmproj | 8192 | 1,150 MB | vision, on demand |
| Qwen3 8B `Q4_K_M` | 5.0 GB | 4096 | 6,200 MB | **does not fit** — pages from disk |

That last row is the honest one. The 8B model *runs*, and it was measured
doing so: **~130 s to load, 3.7–5.5 tok/s prompt, 0.23–0.49 tok/s generation.**
At a quarter of a token per second it is four seconds a word. Smaller models on
the same machine generate at roughly **2 tok/s** — eight times faster, because
they fit.

**So, by machine:**

| Free RAM | Sensible ceiling | Example |
|---|---|---|
| ~1 GB | a 2B at `Q3_K_S`, ~0.7 GB | scraping by; quality suffers |
| ~1.5 GB | a 2B at `Q4_K_M`, ~1.1 GB | fine for chat and simple tools |
| ~2–3 GB | a 3B at `Q4_K_M`, ~2.1 GB | the sweet spot on an 8 GB laptop |
| ~6 GB | a 7–8B at `Q4_K_M`, ~5 GB | needs 16 GB total to be pleasant |
| 12 GB+ | a 14B at `Q4_K_M`, ~9 GB | a different class of machine |

**On quantisation.** `Q4_K_M` is the default answer: about half the size of
`Q8_0` for a quality loss most people cannot pick out in chat. Drop to
`Q3_K_S` only when the alternative is not running at all — it is noticeably
worse at instruction-following, which matters here because the agent depends
on the model emitting well-formed tool calls. `Q8_0` is worth it only for
small models where accuracy is the whole point, which is why the OCR model
uses it and the chat models do not.

**Two things that are not about size.** A model must be an *instruct* or
*chat* build, not a base model — a base model will not follow the system
prompt or emit tool calls. And it needs tool-calling support in its chat
template, or the agent degrades to a plain chatbot.

**The guard warns, it does not refuse.** Starting a model with less free RAM
than the formula asks for produces a warning, not a refusal, because a
shortfall predicts *slow* rather than *broken* — measured: Ministral 3B
started with 518 MB free and completed its turn. Only a genuinely tiny margin
is fatal.

Drop any `.gguf` into `weights/` and it is picked up on the next scan, with
its context and RAM threshold worked out from its own header. Nothing needs
editing. Good sources:
[bartowski](https://huggingface.co/bartowski) and
[unsloth](https://huggingface.co/unsloth) on Hugging Face both publish
well-made GGUF quantisations of most open models.

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

### Pairing a vision model with its projector

An `mmproj-*.gguf` is recognised as a vision projector, paired with its model,
and never offered as something to talk to. Two things about that pairing were
got wrong first and are worth stating, because both produce a model that loads
and then cannot see — which reads as a broken model rather than a misfiled
file.

**The names will not match, and that is normal.** A projector ships at F16 or
Q8_0 while the model it belongs to is Q4_K_M, and two uploaders will spell the
same model `Qwen3VL` and `Qwen3-VL`. Comparing the stems whole misses almost
every real pair, so the quantisation suffix and all punctuation come off both
sides before they are compared.

**A projector may name no model at all.** The most-downloaded Qwen3-VL
repository calls its projector `mmproj-F16.gguf` — named for what it is, not
what it is for. With one model in the folder there is only one thing it can
belong to and it is paired. With several it is left alone, because handing it
to the wrong model is worse than not handing it to any: rename it after its
model and it pairs.

**A model with a projector is still a chat model.** It was briefly given
GLM-OCR's `ocr` role on the reasoning that a projector means a vision backend.
That is true only of GLM-OCR, which cannot call tools and so could never drive
the agent loop; Qwen3-VL Instruct is an ordinary chat model that can also see.
Filing it as OCR took it out of the model picker entirely — installed,
discovered, listed, and impossible to talk to. `role` is curated in
`models.json`, which is where GLM-OCR gets its own, and is never inferred.

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

Primary model, the router's two ends, which llama-server runs, and per-model
`label`, `context`, `threads`, `gpu_layers` and `min_free_mb`. Retuning applies the next time that
model starts, because llama-server is given those on the command line.

`gpu_layers` is `-ngl`, and it is passed **always**, including as `-ngl 0`.
Explicit rather than left to the build's default, so that changing which
llama-server runs cannot quietly change how an existing model runs. It is 0
everywhere until somebody sets it: the CPU build is what `get_llama.py`
installs, and a discovered model is never guessed at, because whether
offloading helps is a fact about the machine rather than about the file.
[What it actually bought here](#the-integrated-gpu-helps-the-prompt-not-the-tokens)
— a quarter off prompt processing, nothing off generation, and a partial split
slower than either end.

Deliberately **not** editable: `file`, `port` and `role`. Those decide what a
model *is* and where it runs, and getting them wrong from a settings panel
produces a model that will not start for reasons the panel cannot explain.

Models you do not want in the picker can be hidden; the file stays where it is,
and the primary cannot be hidden.

```
POST   /api/models/rescan          re-read the folder
POST   /api/models/primary         {"key": "..."}
POST   /api/models/router          {"fast": "...", "strong": "..."}
POST   /api/models/server          {"path": "..."} - which llama-server runs
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
| `search_documents`, `list_documents`, `get_document_outline` | documents | **off by default** |

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

Default workspace is the project directory. `AGENT_WORKSPACE` moves it before
startup, and the **Workspace panel or the folder pill in the composer** moves it
while the app is running - see [Choosing a workspace](#choosing-a-workspace).

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
"C:\path\to\llama.cpp\llama-server.exe" -m "C:\path\to\Hakim Local Agent\weights\GLM-OCR-Q8_0.gguf" --mmproj "C:\path\to\Hakim Local Agent\weights\mmproj-GLM-OCR-Q8_0.gguf" -c 4096 -t 4 -np 1 --port 8081
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

### Dictating a message

A microphone beside the paperclip in the composer. Press it, speak, press it
again; the words land **in the message box**, not in a turn. That is the whole
design decision, and it is not politeness — see below.

Nothing leaves the machine. `whisper-cli` is run as a subprocess, the clip goes
to a temporary file that is deleted whether or not transcription worked, and
the transcript comes back over loopback like everything else.

**No resident server, and that is measured rather than assumed.** Every other
model here runs as a long-lived server, so the obvious thing would be to run
`whisper-server` the same way. With `ggml-base.en.bin`:

```
2 s of audio      4.9 s
10 s of audio     5.2 s
30 s of audio     4.1 s
```

The wall clock barely moves with the length of the clip, because nearly all of
it is loading a 148 MB model — whisper decodes in 30-second windows, so a short
clip and a long one are one window of work either way. A resident server would
save about four seconds a clip and hold roughly 200 MB for as long as it lived.
On 8 GB, where a chat model is the thing that actually wants the RAM, four
seconds is the cheaper side of that trade. It also means nothing to supervise,
nothing to reconcile after a crash, and no second idle timeout.

**Whisper invents speech when it hears none.** Two seconds of digital silence
transcribes as " you"; a synthetic tone comes back as " (dramatic music)". This
is whisper doing what it was trained to do. `-sns` asks it to suppress
non-speech tokens and `clean_transcript` strips the bracketed annotations, but
an invented ordinary word is indistinguishable from a spoken one — which is
exactly why the transcript goes into the box to be read and corrected, and
never straight to the agent.

The browser records WebM/Opus in Chrome and Ogg/Opus in Firefox, and whisper
reads neither reliably. Rather than add ffmpeg to the install, the clip is
decoded and re-encoded in the browser as the 16 kHz mono WAV whisper resamples
to anyway — `web/src/lib/dictation.ts`, about forty lines, using the audio
decoder the browser already has.

**Setting it up.** Two files, both found automatically:

```
vendor/whisper/whisper-cli.exe    a whisper.cpp build   (or on PATH)
whisper/ggml-base.en.bin          a model               (weights/ works too)
```

Get them from [whisper.cpp releases](https://github.com/ggml-org/whisper.cpp/releases)
and [the ggml models](https://huggingface.co/ggerganov/whisper.cpp). `base.en`
is 148 MB and is the right first choice on this hardware; `tiny.en` is 75 MB
and faster if four seconds a clip is too long. With neither installed the
microphone is not drawn at all, rather than drawn and broken.

| | |
|---|---|
| Formats | `.wav`, `.mp3`, `.ogg`, `.flac` — the browser always sends `.wav` |
| Size | 25 MB, about 13 minutes, enforced while reading rather than after |
| Length | recording stops itself at 2 minutes and keeps what it has |
| Retention | none — the clip is deleted in a `finally`, worked or not |
| Microphone | released explicitly on stop, cancel, and unmount |

```
GET  /api/speech             whether it can run, and on what model
POST /api/speech/transcribe  one clip in, text out
```

### Reading a reply aloud

A speaker beside the copy button on any answer. Press it to hear that one.

Opt-in per message rather than automatic, because a long answer full of tool
output reading itself at you is worse than silence — and pressing a button is
a cheaper way to say "this one" than a setting is.

**The voice is kept warm, which is the opposite of what dictation does.**
Measured with `en_US-lessac-medium`:

```
              cold      warm
21 words      8.8 s     1.37 s
42 words     10.2 s     2.52 s
84 words     15.5 s     5.21 s
```

About **7.3 s fixed plus 0.07 s a word**, and six of those seconds are loading
the voice. Warm it runs at **0.22× realtime** — it produces speech four and a
half times faster than anybody can listen to it, so the wait is the load and
nothing else. Paying that before every spoken reply would be seven seconds of
silence for a one-line answer; paying it once a session costs **175 MB**,
measured, and the same sweeper that unloads idle llama-servers and the
embedding worker gives it back after `PIPER_IDLE_SECONDS`.

That is the whole reason this is a resident worker and whisper, twenty lines
away, is a subprocess per clip. Whisper's cost is *all* load and clips are
occasional; Piper's load is paid before every sentence you want to hear.

It runs in its own process for the reason `rag/worker.py` records: freeing the
weights does not free the onnxruntime allocations behind them, and a child
process gives every byte back when it exits.

**Markdown is stripped before it is spoken.** A fenced code block read aloud is
thirty seconds of punctuation, so it becomes the word "code"; links read as
their text, and headings, emphasis and table pipes go entirely.

**Setting it up.** `piper-tts` is in `requirements.txt`, so all you need is a
voice — an `.onnx` with its `.onnx.json` beside it — in `tts/`:

```
tts/en_US-lessac-medium.onnx        63 MB
tts/en_US-lessac-medium.onnx.json
```

Voices are at [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices).
Without one the speaker is not drawn at all, rather than drawn and broken.

> The standalone `piper` binaries from the old `rhasspy/piper` releases are a
> separate, archived line, and the Linux tarball will not run on Windows. The
> pip package is the maintained one and is cross-platform, which is why it is
> a requirement rather than something `get_llama.py` fetches per platform.

```
POST /api/speech/speak       text in, a WAV out — never written to disk
```

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

### documents — hybrid search over your own files

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
| [`rag/extract/`](rag/extract/) | file → text and headings; PDFs read their own bookmarks |
| [`rag/chunker.py`](rag/chunker.py) | text → overlapping chunks on paragraph boundaries |
| [`rag/embeddings.py`](rag/embeddings.py) | owns the worker process; starts it late, stops it early |
| [`rag/worker.py`](rag/worker.py) | the model itself, in its own process |
| [`rag/index.py`](rag/index.py) | the vector file, searched with numpy |
| [`rag/metadata.py`](rag/metadata.py) | chunk text, documents, the free list, the keyword index (SQLite) |
| [`rag/manager.py`](rag/manager.py) | decides the order everything happens in |

#### Two ways of finding a passage, fused

Embeddings are very good at "how does removing water from an alcohol make an
alkene" and very bad at **`E2`**. A term that has to *appear* — a code, a
formula, a surname, a section number — is exactly what a 384-dimension
sentence embedding has no reason to treat as special, and bge-small's noise
floor makes it worse: unrelated English sentences already score 0.4–0.55 with
this model, so there is little room left to tell a weak match from no match.

So search runs both halves and fuses them:

```
              query
                │
      ┌─────────┴─────────┐
      ▼                   ▼
 vector search       FTS5 + BM25
 (cosine, numpy)     (the words themselves)
      │                   │
      └─────────┬─────────┘
                ▼
    reciprocal rank fusion
                ▼
            passages
```

**The keyword half costs nothing.** The chunk text is already in SQLite, and
FTS5 is already in the standard library — no new model, no new process, no new
dependency, no RAM. It is an external-content FTS5 table (`content='chunks'`),
so the text is stored once and kept in step by triggers.

**Adding it to an existing index costs no re-embedding.** The chunk text is
already there, so the migration is one FTS5 `'rebuild'` and no model at all —
which is why this did *not* bump `SCHEMA_VERSION`, since that would have meant
re-embedding every document to gain something free.

**Fusion is by rank, not by score.** Cosine similarity and BM25 share no scale,
and normalising them into one number would make the weighting depend on the
spread of whatever a particular query happened to return. Reciprocal rank
fusion adds `1/(60 + position)` from each ranking, so a passage near the top of
either beats one that is middling in both.

**`score` stays cosine similarity.** A keyword hit arrives with a row and no
score, so its vector is read back and measured the same way — one column, one
meaning. Each result also carries `match`, one of `semantic`, `keyword` or
`both`, because a 0.42 means different things for each: a weak guess, or a
passage that literally contains what was asked for.

**The threshold gates guesses, not evidence.** `RAG_MIN_SCORE` still filters
purely semantic hits. A chunk that contains the search terms is admitted
whatever its similarity — which is the entire point, since the case this fixes
is a real answer scoring *below* the floor.

Set `RAG_HYBRID=0` to measure what it is buying on your own documents. If
SQLite was built without FTS5, search quietly falls back to vectors alone and
says so when it finds nothing.

#### Extraction speed, and the one call that dominated it

Measured on a generated 120-page prose textbook with a real table of contents:

| Stage | Time | Rate |
|---|---|---|
| Extract, **before** | 16.06 s | 7.5 pages/s |
| Extract, **after** | 0.61 s | 195 pages/s |
| Chunk | 0.02 s | 5,200 pages/s |

Table detection was **98% of extraction time and found nothing**. PyMuPDF's
`find_tables()` defaults to `vertical_strategy="lines"` and
`horizontal_strategy="lines"`, so it only ever detects tables delimited by
*drawn lines* — and it was being run on all 120 pages of prose to establish
that, at about 130 ms each.

So a page is now checked for vector drawings first, and detection runs only if
it has any. That is a filter rather than a heuristic: it skips only pages where
the detector's own strategy guarantees no result. A ruled table is still found;
a borderless one was never found by this configuration and still is not, and
its text is extracted and indexed as text either way.

For a 1,247-page book that is the difference between about **2.8 minutes and
3.5 seconds** of extraction.

#### Figures

Embedded raster images large enough to be figures are pulled out while
indexing, written under the store as PNG, and recorded with the page they came
from and their caption.

The caption is the searchable part, and it is found by looking at the text
blocks **directly below** the picture and horizontally overlapping it — where
captions overwhelmingly sit — and accepting only text that actually looks like
one (`Figure 3.4 …`, `Fig. 2`, `Chart 1`, `Diagram …`). Body prose under a
picture is left alone: an empty caption is honest, and calling a paragraph a
caption is not.

Furniture is filtered out. An image under 120 px on either side, or covering
less than 2% of the page, is a bullet or a logo in a running header, and
extracting them would bury the real figures in noise.

The pictures are kept because a caption tells you a figure exists and nothing
about what it shows — they are what something that can *see* would read later.
`get_document_outline` lists them, and they are removed with their document and
on a rebuild, so the store does not accumulate orphans.

**What this does not do is read a chart.** Two limits, both real:

- **Vector artwork is not a figure as far as the file is concerned.** A chart
  drawn as lines and rectangles is not an image object, so it is not extracted.
  Its axis labels and caption are still indexed as page text; the data is not.
- **Nothing looks inside the picture.** A caption is text near the image, not
  text in it. Reading the contents needs a vision model over the extracted
  PNGs — which is exactly what keeping them makes possible.

`RAG_FIGURES=0` turns extraction off.

#### What indexing a book actually costs

Measured on this machine — i5-6300U, two cores — against the real BGE model,
indexing a generated 120-page prose textbook (240 chunks, ~1,180 characters
each):

| | Time | Rate |
|---|---|---|
| Extract + chunk | 0.6 s | 195 pages/s |
| **Embed, cold** | 252.8 s | 0.95 chunks/s |
| **Embed, warm** (model already up) | 149.0 s | 1.61 chunks/s |
| Model load, one-off | ~104 s | |
| **Search a query** | **0.19 s** | |

Extrapolated from the warm rate:

| Book | Ingestion |
|---|---|
| 300 pages | ~6 minutes |
| 1,247 pages | **~26 minutes** |

Three things follow from that.

**Ingesting a book is a coffee break, not an overnight job.** This was the open
question, and the answer is comfortable.

**Embedding is essentially the whole cost.** Extraction is now 3.5 s of that 26
minutes. Anything further spent optimising the reading of PDFs is spent on 0.2%
of the bill.

**Retrieval is instant, and it is a one-off cost you pay once per book.** A
query against the finished index is 0.19 s — three orders of magnitude below a
single model turn. Against the real model, a relevant query scored 0.740, 0.736
and 0.718, all marked `both`, so the semantic and keyword halves agreed.

These were taken with about 530 MB free on a machine at 93% memory load, so the
embedding worker was competing for RAM throughout. **They are a floor**: with a
gigabyte free the warm rate would be better, not worse.

#### Structure: outlines and scoped search

Retrieval answers "which passages match this question". It cannot answer "what
is in this book", because you have to know what to ask before it helps.

Every chunk already records the section it came from — a heading starts a new
run, and the chunker never merges across one. **PDFs were the exception, and
the wrong way round:** they produced no headings at all, so every chunk of a
textbook had `section = None`, in the one format where the chapter matters
most.

They now get their structure from **the file's own bookmarks**, via
PyMuPDF's `get_toc()` — the author's table of contents rather than a guess made
from font sizes, and no model involved. A PDF without bookmarks gets no
headings, exactly as before.

The limitation is worth stating plainly: a bookmark points at a *page*, not a
position on it. A section therefore begins at the top of its page, and a page
holding the end of one section and the start of the next is credited entirely
to the new one. Fixing that means reasoning about text coordinates, which is a
great deal of work to move a boundary by a paragraph.

Two things use it:

`get_document_outline(document)` lists the sections in order with their pages,
derived from the chunks rather than stored twice:

```
Chapter 2 Alkanes      p2–14   28 chunks
Chapter 3 Alkenes      p15–31  41 chunks
  3.7 Dehydration      p27–31  9 chunks
```

`search_documents(query, document=…, section=…)` narrows *before* searching,
which is the difference between "the best five passages in this chapter" and
"whichever of the best five overall happened to land in it". A section is
matched loosely — `section="Chapter 3"` finds `Chapter 3 Alkenes` — because
people ask by chapter, not by pasting a bookmark verbatim.

A scoped search scores every candidate in the scope outright rather than
searching the whole index and discarding: a section is tens of chunks, so that
is both exact and cheaper. Keyword matching applies inside a scope too.

Same on the API: `GET /api/rag/documents/{name}/outline`, and `document` /
`section` on `POST /api/rag/search`.

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

### Editing a question

Hover any question and there is a pencil: change it, and it is asked again.

The important part is what happens to what came after. `DELETE
/api/conversations/{id}/messages/{message_id}` removes that message **and every
one after it**, and only then is the edited text sent as a new turn. The old
question, the answer it got and everything that followed are all a reply to
something that is no longer what was asked — keeping them would leave a
transcript of a conversation nobody had, and would show the model the same
question twice.

So it is destructive, and the editor says so before you commit to it.

Two consequences worth stating:

- **It is refused with a 409 while any turn is running or queued.** A queued
  turn is identified by the id of its own user message and reads its history
  when it runs, so deleting rows underneath it would either change what it is
  answering or delete the question itself. The pencil is disabled and says why.
- **It does not reach memory.** Anything already extracted into the memory store
  from the old messages stays there. That store has its own lifecycle and its
  own way of being corrected, and quietly deleting from it here would be a
  second, invisible deletion nobody asked for.

Editing the *first* question retitles the conversation, since the title was
taken from a question that no longer exists.

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
- **Ask the next question while one is still running** — it queues, with its
  position, and runs in order. See [Queueing questions](#queueing-questions)
- The model's **reasoning** streams into a collapsible panel — see below
- **A turn can be ended**, at any stage, from the Stop control beside it - see
  [Ending a turn](#ending-a-turn)
- **Any question can be edited and asked again**, which rewinds the
  conversation to that point — see above
- **Tool switches** in the sidebar, with each tool's own risk text
- **The workspace is chosen here**, from the folder pill in the composer or
  the Workspace panel - see below
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

### Queueing questions

The composer used to lock while a turn ran. On a machine where a turn is
minutes, that means holding the next question in your head until the last one
lands - so it does not lock any more. Send it and it queues.

The queue was already there and already tested; what was missing was any way
to reach it from the UI. Each question is a POST of its own, stored server-side
the moment it is accepted, and its stream reports its position while it waits.
So a queued question survives a reload, and the position it shows is the
server's, not a guess.

**At most four in flight from one tab**, and that number is the browser's, not
the server's. Every turn in flight holds an open SSE connection, and browsers
allow six per origin over HTTP/1.1; reach that and every other request queues
behind them, so the page cannot even fetch `/health` and looks hung. Four
leaves room for the rest of the app. It is also about right on its own terms:
at under a token a second, three questions waiting is the best part of an
hour. Past that the send button greys out and says so - the draft is kept, and
the textarea stays editable. The server keeps its own, larger bound (eight
waiting, refused with a 429), which is what covers several tabs at once.

**Any of them can be stopped**, running or waiting, from its own control - see
[Ending a turn](#ending-a-turn).

**What a queued turn sees when it runs** was a real bug this exposed. Rows are
not written in conversation order: a queued question is stored when it is
accepted, which is *before* the answer to the turn ahead of it exists. History
had been selected as "everything with a lower id", which therefore dropped
exactly the answer the user was most likely replying to. It now takes the
conversation as it stands, minus this turn's own question and minus any
question queued behind it.

### Ending a turn

A turn used to be unstoppable. The Stop control ended the *watching* - the
server carried on, finished, and stored the answer - which is the right
default for a closed tab and the wrong one for a five-minute turn that is
visibly going nowhere.

Stop now ends the turn. Two different things share the button, and which one
applies is an accident of timing:

| Where it was | What happens |
|---|---|
| Waiting in the queue | Dropped. It never runs, and its stream is closed by the queue, because no worker will ever pick it up to do that |
| Running | Asked to stop, and it does at its next checkpoint |

**Checkpoints, because there is no other honest mechanism.** A Python thread
cannot be interrupted from outside, so stopping means the loop noticing
between one piece of work and the next. It checks before each model round,
after every tool call, and - the one that matters - between streamed chunks.
In practice that is under a second while text is arriving. The exception is
the silent stretch where the model is reading the prompt and nothing is coming
back: there is no chunk to check between, so a stop asked for then waits for
the first token.

Leaving the chunk loop closes the response, which drops the connection, and
**that** is what makes llama-server stop generating. The flag alone would free
nothing - it would only stop this end listening while the CPU carried on.

**Whatever was written is kept.** At under a token per second, throwing away
two minutes of prose because the last word never arrived is the wrong trade.
The partial answer is stored with a note - *(stopped before finishing)* - for
the same reason a clipped tool result says it was clipped: a truncated answer
with nothing to say so reads, on reload, as one that simply ended. Nothing is
stored when nothing was generated.

Ending a turn is a deliberate request (`POST /api/chat/{turn_id}/stop`), not
something a closed tab does by accident. Disconnecting and ending are
different intentions, and this is the one place the difference is visible. A
turn that finished before the click is reported as `unknown` rather than a
404: that is the outcome that was asked for, not an error.

### Choosing a workspace

The workspace is the jail. `WorkspaceFiles` resolves every path the model gives
it and refuses anything landing outside; the terminal tool runs with it as its
working directory; git works on it; uploads land inside it. Until now it could
only be chosen before startup, with `AGENT_WORKSPACE` - which made the agent
useful mainly for reading its own source.

It is now a control: the **folder pill in the composer**, beside the model and
the tool count, and a **Change folder** button in the Workspace panel.
`/workspace` opens the same picker, and `/workspace <path>` skips it.

**The picker walks the filesystem server-side, and it has to.** A directory
picker in a browser hands the page a folder's *name* and never its absolute
path, which is the one thing the tools need. So `GET /api/workspace/browse`
lists one level at a time - sub-directory names only, never a file's contents -
and the path shown is the real one rather than a reconstruction. Pasting a path
from Explorer works too, because eleven clicks is not an improvement on Ctrl+V.

Two folders are refused, in `runtime.resolve_workspace`:

| Refused | Why |
|---|---|
| A drive root (`C:\`) | A jail around the whole disk is not a jail |
| Windows, Program Files, `/usr`, `/etc` … | The operating system is not a project |

Neither is a security boundary - with the terminal switch on this API can hand a
model a shell - they stop the two choices that would make the jail meaningless.

**It applies from the next turn**, and moving it is refused with a 409 while one
is running: the tool registry is built when a turn starts, so moving the jail
underneath a running turn would either do nothing or move it halfway through.

Like the tool switches, the choice **lives in memory only**. A restart returns to
whatever `AGENT_WORKSPACE` says, which stays the durable answer to "what can
this thing reach"; the panel marks a session-only choice `this session`. Two
things follow the workspace when it moves: uploads, which land in the new
folder, and every file tool. Attachments uploaded before a move stay in the old
folder and the agent can no longer read them - re-attach them.

The panel names which switched-on tools act on the folder, and the picker
repeats it before you choose. "File writes is switched on, so the agent will act
on whatever you pick, and can change files there" is worth reading before
pointing it at Documents.

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
"C:\path\to\llama.cpp\llama-server.exe" -m weights\GLM-OCR-Q8_0.gguf --mmproj weights\mmproj-GLM-OCR-Q8_0.gguf -c 4096 -t 4 -np 1 --port 8081
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
| `/workspace [path]` | Move the workspace. No path opens the picker |
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
| `AGENT_WORKSPACE` | project dir | The only directory tools may read. The starting point; the UI can move it for the session |
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
| `RAG_MIN_SCORE` | `0.3` | Cosine similarity floor, for semantic hits only |
| `RAG_HYBRID` | `1` | Keyword matching beside the embeddings, fused by rank |
| `RAG_FIGURES` | `1` | Pull raster figures out of PDFs while indexing |
| `RAG_CONTEXT_CHARS` | `6000` | Retrieved text handed to the model per call |
| `RAG_THREADS` | `2` | Shared with llama-server |
| `RAG_BATCH_SIZE` | `8` | Bigger costs RAM, not speed |
| `RAG_IDLE_SECONDS` | `120` | Before the embedding model is unloaded |
| `RAG_MAX_FILE_BYTES` | `20000000` | Largest file indexed |

Model paths, ports, contexts, threads, GPU layers, RAM thresholds and the
router's fast/strong pair live in [`models.json`](models.json).

---

## 14. Tests

```bash
cd "C:\path\to\Hakim Local Agent" && .venv\Scripts\python -m unittest discover -s tests -t .
```

**1140 tests, no model server needed, and none of them touch the network.**
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
| `test_agent.py` | Loop: plain replies, one call, several calls, tool errors, iteration limit, malformed replies, stopping at each checkpoint |
| `test_tools.py` | Calculator, workspace jail, OCR validation, registry |
| `test_python_tool.py` | Restricted execution; spawns real child processes |
| `test_streaming.py` | SSE parsing, tool-call fragment assembly, reasoning suppression, abandoning a stream part-way |
| `test_manager.py` | Start, stop, switch, adopt, crash recovery, idle unload |
| `test_router.py` | Routing decisions, no-downgrade rule, scoring |
| `test_chat_store.py` | History round-trip, ordering, deletion, corrupt JSON |
| `test_ocr.py` | Validation, request shape, capability checks, error paths |
| `test_shell_tool.py` | Allowlist, chaining, dangerous options, path confinement |
| `test_http_tool.py` | Host/scheme allowlist, redirect refusal, method gating |
| `test_port_reclaim.py` | Reclaiming a port from a llama-server we did not start |
| `test_turns.py` | The queue: serialisation, positions, bounded backlog, a runner that raises, stopping a queued or running turn |
| `test_remote.py` | Hosted models: registry shape, key handling, connectivity cache, error hints |
| `test_api.py` | Every endpoint, the SSE event sequence, routing, reasoning suppression, tool switches, the workspace and its picker, ending a turn, what a queued turn sees, a refused backlog, remote consent, uploads, editing a question |

The API tests use the same manager harness as the model tests and a scripted
chat client, so none of them depend on whether something happens to be
listening on a port — which has produced false greens here before.
| `test_file_writes.py` | Writing, overwrite gating, self-protection |
| `test_python_scripts.py` | Script files in both modes, and the workspace guard |
| `test_git_tool.py` | Real throwaway repositories; write gating |
| `test_memory.py` | Store, recall, forget |
| `test_rag.py` | Extraction, chunking, the vector file, hybrid search and its migration |

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

**So is choosing the workspace from the UI**, and in the same shape. What the
jail does has not moved: one directory, paths resolved before they are checked,
nothing outside it reachable, no delete or rename anywhere in the filesystem
tools. What moved is that *which* directory is a click rather than a restart -
the change that makes the agent useful on your own files. Drive roots and the
operating system's own directories are refused, because a jail around either is
not a jail; that is a guard against a mistake, not a boundary against an
attacker, and with the terminal switch on there is no such boundary here anyway.
The choice is held in memory like every other override.

---

## 16. What is verified and what is not

Being straight about this, because the difference matters.

### Verified against the live model

- **Tool calling works.** The server returned exactly the expected shape:
  `{"name":"calculate","arguments":"{\"expression\": \"sqrt(144) + 25**2\"}"}`
- **A full agent turn works end-to-end** through the web UI: tool call → result
  → final answer, **637**, correct, in 285.7 s
- **A 120-page book indexes in 149 s warm** against the real BGE model — 240
  chunks at 1.61 chunks/s — and a query against the finished index takes
  **0.19 s**. Extrapolates to about 26 minutes for a 1,247-page book
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

- 1140 tests
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
- The custom C99 inference engine at `C:\path\to\mmengine` — parked until
  it is further along

**Measured and decided against:**

- **Borderless table detection** (`find_tables(strategy="text")`). It does
  recover the data rows of an unruled table, but measured on the 120-page prose
  book it took **56.9 s against 0.61 s** — about 93× — and reported a table on
  **120 pages out of 120**, every one surviving the existing two-row filter.
  Enabling it would give every page of a book a duplicate "table" of prose
  chopped mid-word at the column boundaries.

  Two refinements do work, if this is ever wanted: keep only rows that fill most
  of their columns, which drops the mangled title and trailing sentence cleanly;
  and run the expensive strategy only on pages whose text carries a `Table 3.2`
  style caption, which matched 0 of the 120 prose pages, so the cost on prose is
  nil.

  Left alone because the degradation is mild. An unruled table already survives
  as one line per row, which retrieves about as well; what proper detection buys
  is explicit columns, and that helps precise lookup more than it helps search.

---

*Built with llama.cpp, Python 3.11, FastAPI, React and no agent frameworks.*
