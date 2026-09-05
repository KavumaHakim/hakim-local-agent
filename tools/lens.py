"""Which tool schemas a turn actually sends.

Every registered tool's JSON schema travels with every request. With the whole
roster switched on that is 24 tools and about 3,060 tokens, and prompt
processing on this hardware runs at roughly 14.5 tokens per second - so a cold
prefix cache spends around three and a half minutes reading tool definitions
before the model writes anything. Most of them are irrelevant to the question
being asked, and a small model choosing between 24 schemas chooses worse than
one choosing between five.

The lens sends a short index instead, and expands a category's real schemas
only once the conversation looks like it needs them. Two things open a
category:

* the heuristic below, scored against the user's message - free, and right
  often enough that the common case never pays for a round trip;
* the model asking, through `load_tools`, when the heuristic missed.

**Expansion is monotonic, and that is the whole design.** `tools` is rendered
into the prompt *prefix* by the chat template, ahead of the messages, so
changing the set invalidates llama-server's prefix cache and re-reads the
entire conversation. Opening a category is therefore paid for once; closing one
again would cost a second cache miss to save context that has already been
read. This is the same trade the model router makes when it refuses to route
back down, for the same reason.

So the cost model is: a conversation that only ever needs files pays for
`filesystem`; one that wanders into git pays one extra miss and keeps both.
Neither pays for OCR, HTTP, memory and documents it never touches.
"""

from __future__ import annotations

import re
from typing import Any

from tools.base import ToolRegistry, failure

# The name the model calls to ask for more. Registered like any other tool, so
# the loop dispatches it without knowing the lens exists.
LOAD_TOOLS = "load_tools"

# One line per category, shown in the index. This is the only description of a
# closed category the model gets, so it has to say what the category is *for*
# rather than what it contains - the names are listed beside it anyway.
CATEGORY_HELP = {
    "calculator": "arithmetic and unit conversion",
    "filesystem": "read, list, write and create files in the workspace",
    "python": "run Python, as a snippet or a file",
    "terminal": "run a shell command",
    "git": "inspect a repository, and commit to it",
    "http": "fetch a URL",
    "memory": "remember, recall and forget things across conversations",
    "documents": "search the indexed documents",
    "ocr": "read text out of an image",
    "skills": "written instructions for particular kinds of task",
    "results": "read back a tool result that was too large to show",
}

# Words that mean a category is probably wanted. Matched whole, lowercased.
#
# Deliberately narrow. A miss costs one round trip through `load_tools`; a
# false positive costs a prefix-cache miss and permanently occupies context for
# the rest of the conversation, because opening is monotonic. So generic verbs
# that appear in every other sentence - "run", "search", "show", "get" - are
# left out even though they would each catch a real case.
CATEGORY_SIGNALS: dict[str, tuple[str, ...]] = {
    "calculator": (
        "calculate", "arithmetic", "multiply", "divide", "subtract",
        "percent", "percentage", "average", "sum of", "square root",
    ),
    # No bare "read", "write" or "save": they are ordinary English verbs -
    # "write a haiku" opened the filesystem before they were taken out, and a
    # false positive here is permanent. Nouns, plus the filename patterns
    # below, carry this category instead.
    "filesystem": (
        "file", "files", "directory", "directories", "folder", "folders",
        "path", "paths", "ls", "dir", "listing", "subdirectory",
    ),
    "python": ("python", "script", "pandas", "numpy", "plot", "snippet"),
    # The command names themselves, which is how someone asks for a command:
    # "run mkdir reports", not "use the terminal to create a directory". Only
    # the ones that are unambiguously programs - `ls`, `find`, `sort`, `make`,
    # `date` and `file` are all ordinary English too, and a false positive
    # here is permanent.
    "terminal": (
        "command", "shell", "terminal", "bash", "powershell", "cmd",
        "npm", "pip", "install", "cli", "mkdir", "rmdir", "touch",
        "curl", "wget", "ping", "docker", "cargo", "rustc", "dotnet",
        "chmod", "whoami", "hostname", "uname",
        "run this", "in the shell", "command line",
    ),
    "git": (
        "git", "commit", "commits", "branch", "branches", "diff", "repo",
        "repository", "staged", "unstaged", "checkout", "merge",
    ),
    "http": ("http", "https", "url", "fetch", "download", "api", "endpoint"),
    "memory": (
        "remember", "recall", "forget", "memory", "memories", "remind",
        "note that", "keep in mind",
    ),
    "documents": (
        "document", "documents", "pdf", "docx", "my notes", "indexed",
        "corpus", "knowledge base",
    ),
    "ocr": (
        "ocr", "screenshot", "scanned", "scan", "image", "photo", "picture",
        "receipt",
    ),
}

# Shapes that mean a category as reliably as a word does. Kept separate
# because a regex cannot go in the whole-word matcher above.
CATEGORY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "filesystem": (
        # Something that looks like a filename: `config.py`, `notes.md`.
        re.compile(
            r"(?<![\w.])[\w.-]+\."
            r"(py|txt|md|json|csv|ya?ml|toml|ini|log|html?|css|jsx?|tsx?|pdf)"
            r"(?![\w])"
        ),
        # Or a path with a separator in it, rather than a bare word.
        re.compile(r"[\w.-]+[\\/][\w.-]+"),
    ),
}


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _mentions(text: str, phrase: str) -> bool:
    """Whether `phrase` appears in `text` as a whole word or phrase."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


class ToolLens:
    """Decides which of a registry's tools a request can see.

    Holds the set of open categories for one conversation. Nothing here is
    per-turn state: the point is that it survives across turns, because what
    was opened stays open.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        # `skills` is open from the start, and that is not an oversight: its
        # tool *is* the index, and a model cannot ask for instructions it has
        # never been told exist. Gating that behind a keyword would hide the
        # feature rather than defer it.
        #
        # `results` is deliberately *not* here. The loop opens it the moment
        # something is actually set aside, so `read_result` costs nothing in a
        # conversation that never overflows a result - which is most of them.
        always: tuple[str, ...] = ("calculator", "skills"),
    ) -> None:
        self._registry = registry
        # Categories the roster actually has, so a signal for a switched-off
        # category cannot open something that does not exist.
        self._known = {tool.category for tool in registry.list_tools()}
        self._open: set[str] = {c for c in always if c in self._known}
        # Every tool's own name opens its category. A model that has been told
        # a tool exists asks for it by name - "use run_command" - and that was
        # missing a group whose hand-written signals did not happen to include
        # the word. Derived rather than listed, so a new tool brings its own
        # signal and cannot be forgotten. Underscores are word characters, so
        # `run_command` matches as one word and never inside another.
        self._tool_signals: dict[str, str] = {
            tool.name.lower(): tool.category for tool in registry.list_tools()
        }

    @property
    def open_categories(self) -> set[str]:
        return set(self._open)

    @property
    def closed_categories(self) -> set[str]:
        return self._known - self._open

    def consider(self, text: str) -> set[str]:
        """Open whatever the message looks like it needs. Returns what opened.

        Called once per user turn, before the first request, so that a turn
        which obviously needs files opens `filesystem` without spending a round
        trip discovering it.
        """
        lowered = (text or "").lower()
        named = {
            category
            for name, category in self._tool_signals.items()
            if _mentions(lowered, name)
        }
        opened = {
            category
            for category in self.closed_categories
            if category in named
            or any(
                _mentions(lowered, phrase)
                for phrase in CATEGORY_SIGNALS.get(category, ())
            )
            or any(
                pattern.search(lowered)
                for pattern in CATEGORY_PATTERNS.get(category, ())
            )
        }
        self._open |= opened
        return opened

    def open_categories_by_name(self, names: list[str]) -> set[str]:
        """Open categories the model asked for. Returns what actually opened."""
        opened = {n for n in names if n in self.closed_categories}
        self._open |= opened
        return opened

    def definitions(self) -> list[dict[str, Any]]:
        """The tool definitions this turn sends: open schemas, plus the index."""
        defs = [
            tool.definition()
            for tool in self._registry.list_tools()
            if tool.category in self._open and tool.name != LOAD_TOOLS
        ]
        # Nothing left to offer, so the index would be a tool the model can
        # only waste a call on.
        if self.closed_categories:
            defs.append(self._index_definition())
        return defs

    # --- the index ---

    def _index_definition(self) -> dict[str, Any]:
        """`load_tools`, with the menu of what is still closed in its text.

        Built fresh each turn rather than registered once, because the menu
        shrinks as categories open. The registered copy exists only so the
        loop can dispatch a call to it; this is the version the model reads.
        """
        closed = sorted(self.closed_categories)
        listing = "\n".join(
            f"- {name} ({_plural(self._count(name), 'tool')}): "
            f"{CATEGORY_HELP.get(name, '')}"
            for name in closed
        )
        return {
            "type": "function",
            "function": {
                "name": LOAD_TOOLS,
                "description": (
                    "Load more tools. Their full descriptions are not shown "
                    "yet - only the summaries below. Call this when the task "
                    "needs one of them, then use the tools it returns.\n"
                    f"{listing}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "categories": {
                            "type": "array",
                            "items": {"type": "string", "enum": closed},
                            "description": "Which groups to load.",
                        }
                    },
                    "required": ["categories"],
                },
            },
        }

    def _count(self, category: str) -> int:
        return sum(
            1
            for tool in self._registry.list_tools()
            if tool.category == category and tool.name != LOAD_TOOLS
        )

    def load(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a `load_tools` call.

        Dispatched by the agent rather than registered in the registry. The
        registry is shared between conversations while what a lens has opened
        is not, so a registered closure would let one conversation open
        categories in another.

        Arguments come from the model and are checked here for the same
        reason every tool checks its own: nothing upstream validates them,
        because nothing upstream knows this call exists.
        """
        asked = arguments.get("categories")
        if isinstance(asked, str):
            # A small model given an array parameter will sometimes send one
            # string. Cheaper to accept than to spend a round trip refusing.
            asked = [asked]
        if not isinstance(asked, list) or not asked:
            return failure("Name at least one group to load.")

        asked = [str(item) for item in asked]
        unknown = [c for c in asked if c not in self._known]
        if unknown:
            return failure(
                f"No such tool group: {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(self.closed_categories)) or 'none'}."
            )

        opened = self.open_categories_by_name(asked)
        names = sorted(
            tool.name
            for tool in self._registry.list_tools()
            if tool.category in set(asked)
        )
        return {
            "success": True,
            "loaded": sorted(opened),
            "already_loaded": sorted(set(asked) - opened),
            "tools": names,
            # Said explicitly because the schemas arrive with the *next*
            # request, not this one. A model told only "success" tends to
            # guess at the arguments straight away and get them wrong.
            "note": (
                "These tools are available from your next message. "
                "Call one of them now."
            ),
        }
