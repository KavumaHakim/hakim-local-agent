"""A small terminal toolkit for the setup script. No dependencies, by force.

Setup runs before anything is installed, so `rich`, `tqdm` and `questionary`
are all unavailable to it by definition. Everything here is standard library.

Three rules shape it:

**It must degrade, not decorate.** Every one of these works when stdout is a
pipe, a CI log or a cmd.exe window with no ANSI support: spinners fall back to
a single printed line, progress bars to periodic percentages, menus to their
default answer. A setup script that hangs waiting for input nobody can give,
or that fills a CI log with carriage returns, is worse than a plain one.

**Colour is optional and asked for politely.** Windows 10 needs virtual
terminal processing switched on explicitly; if that fails, colour is dropped
rather than printed as escape codes. `NO_COLOR` and `TERM=dumb` are honoured.

**Nothing here reads a key at a time.** Arrow-key menus need raw terminal mode,
which is `msvcrt` on Windows and `termios` everywhere else, and both have edge
cases in the terminals people actually run setup in. Numbered menus read a
whole line, work identically everywhere, and can be answered by someone piping
input in.
"""

from __future__ import annotations

import itertools
import os
import shutil
import sys
import threading
import time

# --- capability detection -------------------------------------------------


def _enable_windows_ansi() -> bool:
    """Ask the Windows console for ANSI support. False if it will not."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:  # noqa: BLE001 - any failure means "no colour"
        return False


def interactive() -> bool:
    """Whether there is a person on the other end to answer a question."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def fancy() -> bool:
    """Whether to redraw lines in place, rather than printing plain ones."""
    if os.environ.get("TERM", "") == "dumb":
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _colour_ok() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    if not fancy():
        return False
    return _enable_windows_ansi()


COLOUR = _colour_ok()


def _rich() -> bool:
    """Whether 256 colours are available, rather than the basic eight.

    The web UI uses a violet accent and this borrows it, but violet is not one
    of the eight colours every terminal has. Where it is unavailable the
    accent falls back to cyan rather than to something approximate.
    """
    if not COLOUR:
        return False
    if os.environ.get("COLORTERM", "") in ("truecolor", "24bit"):
        return True
    return "256color" in os.environ.get("TERM", "")


RICH = _rich()


def _code(value: str) -> str:
    return value if COLOUR else ""


DIM = _code("\033[2m")
BOLD = _code("\033[1m")
RESET = _code("\033[0m")
GREEN = _code("\033[32m")
YELLOW = _code("\033[33m")
RED = _code("\033[31m")
CYAN = _code("\033[36m")

# The accent, matching the front end's violet where the terminal has it.
ACCENT = _code("\033[38;5;141m") if RICH else CYAN
ACCENT_DIM = _code("\033[38;5;103m") if RICH else CYAN


def _can_encode(text: str) -> bool:
    """Whether stdout can actually print this.

    cmd.exe with a legacy code page cannot render block characters, and a
    UnicodeEncodeError would be an absurd way for a setup script to fail. So
    the banner asks first and falls back to ASCII.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


# H A K I M, five rows. Hand-set rather than pulled from a figlet font,
# because one word is not worth a font file and a parser for it.
_LETTERS = {
    "H": ("##   ##", "##   ##", "#######", "##   ##", "##   ##"),
    "A": (" ##### ", "##   ##", "#######", "##   ##", "##   ##"),
    "K": ("##  ## ", "## ##  ", "#####  ", "## ##  ", "##  ## "),
    "I": ("##", "##", "##", "##", "##"),
    "M": ("###   ###", "#### ####", "## ### ##", "##  #  ##", "##     ##"),
}


def banner(word: str = "HAKIM", subtitle: str = "") -> None:
    """Big letters, when there is room and the terminal can draw them.

    Falls back to the plain name on a narrow terminal or a console that cannot
    encode the blocks - a banner is decoration, and decoration that breaks the
    thing it decorates is worse than none.
    """
    rows = ["  ".join(_LETTERS[letter][row] for letter in word) for row in range(5)]
    solid = [row.replace("#", "\u2588") for row in rows]

    if _can_encode("\u2588"):
        rows = solid

    if max(len(row) for row in rows) + 4 > width():
        say(f"\n  {BOLD}{word}{RESET}")
        if subtitle:
            say(f"  {DIM}{subtitle}{RESET}")
        return

    say()
    for row in rows:
        say(f"  {ACCENT}{row}{RESET}")
    if subtitle:
        say(f"  {DIM}{subtitle}{RESET}")


def width(default: int = 80) -> int:
    try:
        return max(40, min(shutil.get_terminal_size((default, 24)).columns, 100))
    except Exception:  # noqa: BLE001
        return default


def erase(lines: int) -> None:
    """Move up `lines` and clear everything below, ready to redraw.

    What makes a menu feel like one control rather than a transcript of every
    time you pressed a key. Only in a terminal - anywhere else the previous
    draw is already scrolled away in a log file and cannot be taken back.
    """
    if lines <= 0 or not fancy():
        return
    sys.stdout.write(f"\033[{lines}A\033[J")
    sys.stdout.flush()


# --- plain output ---------------------------------------------------------


def say(message: str = "") -> None:
    print(message, flush=True)


def rule(title: str = "") -> None:
    line = "-" * (width() - 2)
    if title:
        say(f"\n{DIM}-- {title} {line[len(title) + 4:]}{RESET}")
    else:
        say(f"{DIM}{line}{RESET}")


def heading(text: str) -> None:
    say(f"\n{BOLD}{text}{RESET}")


def ok(message: str) -> None:
    say(f"  {GREEN}[ok]{RESET} {message}")


def warn(message: str) -> None:
    say(f"  {YELLOW}[!] {RESET} {message}")


def fail(message: str) -> None:
    say(f"  {RED}[X] {RESET} {message}")


def note(message: str) -> None:
    say(f"      {DIM}{message}{RESET}")


# --- the plan, and where we are in it -------------------------------------


class Steps:
    """A checklist printed once and then ticked off.

    Shown up front because the first question anyone has about a setup script
    is how long it is going to take and what it is about to do to their
    machine.
    """

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.done: set[int] = set()
        self.current = -1

    def show(self) -> None:
        heading("Here is the plan")
        for index, name in enumerate(self.names, start=1):
            say(f"   {DIM}{index}.{RESET} {name}")
        say()

    def start(self, index: int) -> None:
        self.current = index
        label = self.names[index]
        bar = "".join(
            f"{ACCENT}={RESET}" if position <= index else f"{DIM}-{RESET}"
            for position in range(len(self.names))
        )
        say(
            f"\n{bar}  {ACCENT}{BOLD}{index + 1}/{len(self.names)}{RESET} "
            f"{BOLD}{label}{RESET}"
        )

    def finish(self, index: int) -> None:
        self.done.add(index)


# --- waiting ---------------------------------------------------------------

FRAMES = ("|", "/", "-", "\\")


class Spinner:
    """Something to watch while a subprocess says nothing for two minutes.

    pip and npm are given `--quiet`, so there is no output to stream and no
    total to count against. What can honestly be shown is that the thing is
    still alive and how long it has been going, which is the question being
    asked when someone stares at a still terminal.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def __enter__(self) -> "Spinner":
        self._started = time.monotonic()
        if not fancy():
            say(f"  ... {self.label}")
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            sys.stdout.write("\r" + " " * (width() - 1) + "\r")
            sys.stdout.flush()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def _spin(self) -> None:
        for frame in itertools.cycle(FRAMES):
            if self._stop.is_set():
                return
            seconds = int(self.elapsed)
            clock = f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"
            sys.stdout.write(f"\r  {ACCENT}{frame}{RESET} {self.label} {DIM}({clock}){RESET}   ")
            sys.stdout.flush()
            time.sleep(0.12)


class Progress:
    """A bar for the one thing with a known total: the llama.cpp download."""

    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(0, int(total))
        self.seen = 0
        self._last = 0.0
        self._started = time.monotonic()

    def advance(self, amount: int) -> None:
        self.seen += amount
        # Redrawing faster than this is invisible and costs syscalls. Only in
        # a terminal, though: the plain path prints one line per quarter and
        # throttling it by time as well swallows most of those, leaving a CI
        # log with two lines instead of the five it was meant to have.
        if fancy():
            now = time.monotonic()
            if now - self._last < 0.1 and self.seen < self.total:
                return
            self._last = now
        self.draw()

    def draw(self) -> None:
        fraction = (self.seen / self.total) if self.total else 0.0
        fraction = min(1.0, max(0.0, fraction))
        # Clamped as well: a server that sends a little more than it
        # advertised should not produce "19.2/18.4 MB", which reads as a bug
        # in the thing doing the counting.
        shown = min(self.seen, self.total) if self.total else self.seen
        megabytes = f"{shown / 1e6:.1f}/{self.total / 1e6:.1f} MB"

        if not fancy():
            # A pipe or a CI log: one line per 25%, never a carriage return.
            step = int(fraction * 4)
            if step > getattr(self, "_step", -1):
                self._step = step
                say(f"  {self.label} {int(fraction * 100)}%  {megabytes}")
            return

        elapsed = time.monotonic() - self._started
        rate = self.seen / elapsed if elapsed > 0.5 else 0
        speed = f"{rate / 1e6:.1f} MB/s" if rate else "..."

        room = width() - len(self.label) - len(megabytes) - len(speed) - 14
        cells = max(10, min(room, 40))
        filled = int(cells * fraction)
        bar = "#" * filled + "." * (cells - filled)
        sys.stdout.write(
            f"\r  {self.label} {ACCENT}[{bar}]{RESET} {int(fraction * 100):3d}% "
            f"{DIM}{megabytes} {speed}{RESET}  "
        )
        sys.stdout.flush()

    def done(self) -> None:
        if fancy():
            sys.stdout.write("\r" + " " * (width() - 1) + "\r")
            sys.stdout.flush()


# --- asking ----------------------------------------------------------------


def _read(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        say()
        raise


def confirm(question: str, *, default: bool = True, assume: bool | None = None) -> bool:
    """A yes/no question.

    `assume` short-circuits it, which is how the command-line flags and
    non-interactive runs get their answer without a prompt appearing in a log
    nobody is reading.
    """
    if assume is not None:
        return assume
    if not interactive():
        return default

    hint = "Y/n" if default else "y/N"
    while True:
        answer = _read(f"  {question} {DIM}[{hint}]{RESET} ").lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        warn("Answer y or n.")


def choose(title: str, options: list[tuple[str, str]], *, default: int = 0) -> int:
    """A numbered menu. Returns the index chosen.

    `options` is (label, explanation) pairs. Returns `default` immediately when
    there is nobody to ask.
    """
    if not interactive():
        return default

    say(f"\n  {BOLD}{title}{RESET}")
    for index, (label, explanation) in enumerate(options, start=1):
        marker = "*" if index - 1 == default else " "
        say(f"   {marker} {ACCENT}{index}{RESET}) {label}")
        if explanation:
            note(explanation)

    while True:
        answer = _read(f"  Choose 1-{len(options)} {DIM}[{default + 1}]{RESET} ")
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        warn(f"Enter a number between 1 and {len(options)}.")


def toggle(title: str, items: list[dict], *, assume: bool = False) -> list[dict]:
    """A checklist you can flip entries on and off in, redrawn in place.

    Each item is {"label", "note", "on"}. Numbers toggle, Enter accepts.
    Returns the same list, with "on" updated.

    The redraw is what makes this feel like one control rather than a
    transcript of every key you pressed. Where the terminal cannot move the
    cursor it simply prints again, which is worse-looking and just as usable.
    """
    if assume or not interactive():
        return items

    drawn = 0
    complaint = ""

    while True:
        erase(drawn)
        lines = [f"  {BOLD}{title}{RESET}", ""]
        for index, item in enumerate(items, start=1):
            if item["on"]:
                box = f"{GREEN}[+]{RESET}"
                label = item["label"]
            else:
                box = f"{DIM}[ ]{RESET}"
                label = f"{DIM}{item['label']}{RESET}"
            lines.append(f"   {box} {ACCENT}{index}{RESET}  {label}")
            if item.get("note"):
                lines.append(f"          {DIM}{item['note']}{RESET}")
        lines.append("")
        if complaint:
            lines.append(f"  {YELLOW}{complaint}{RESET}")

        for line in lines:
            say(line)

        prompt = (
            f"  {DIM}number to change one, Enter when it looks right{RESET}  "
            f"{ACCENT}>{RESET} "
        )
        answer = _read(prompt)
        # The lines just printed, plus the prompt line the Enter ended.
        drawn = len(lines) + 1

        if not answer:
            erase(drawn)
            return items
        if answer.isdigit() and 1 <= int(answer) <= len(items):
            items[int(answer) - 1]["on"] = not items[int(answer) - 1]["on"]
            complaint = ""
            continue
        complaint = f"That is not one of them - pick 1 to {len(items)}, or press Enter."


def ask_text(question: str, *, default: str = "") -> str:
    if not interactive():
        return default
    answer = _read(f"  {question} {DIM}[{default or 'skip'}]{RESET} ")
    return answer or default
