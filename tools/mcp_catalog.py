"""The MCP servers this project offers as a starting point.

A catalogue, not an installation. Nothing here is bundled and nothing runs
until someone switches it on: each entry is a command line that `npx` or `uvx`
resolves the first time it is called, which means **the first run downloads a
package from npm or PyPI and executes it**. That is worth saying plainly in the
interface rather than burying, because it is the one thing about adding a
server that is not local, not reversible by deleting a line, and not this
project's code.

Why a catalogue at all, when `mcp.json` is four lines of JSON: the hard part of
adding a server is not the JSON, it is knowing the package name and what its
arguments mean. Getting those right once, with an honest note about what each
one costs, is most of the value.

**Servers reaching an external service need a credential**, declared in
`needs`. Two ways to supply one, and the second is better: paste the value and
it is written into the `env` block of mcp.json, or enter `${SOME_VAR}` and it
is read from this process's environment when the server starts, so the secret
never touches the file at all. The interface offers both and says which is
which. Values are never sent back to the browser - only the names of the ones
that are set.

**The external-service entries are marked unmaintained, and that is not a
hedge.** Checked against the registry: every `@modelcontextprotocol` server
that talks to a third-party service - github, slack, gitlab, brave-search,
google-maps - is published with "Package no longer supported", last released
in 2025. The reference servers that run locally - filesystem, memory,
sequential-thinking, everything - are still shipping. The vendors took their
own integrations over and mostly publish them as **remote HTTP servers**,
which this client now reaches. The archived packages still install and still
work; they are offered on that basis and labelled, so nobody wonders later why
a tool stopped getting fixes.

**An entry is a command or a url, never both.** A url entry is somebody else's
server, reached over Streamable HTTP: nothing is downloaded, nothing executes
here, and its `runtime` is "none". The trade is the other way round from a
local one - there is no package to audit, but the arguments of every call
leave this machine. That belongs in `caution`, and for the one such entry it
is the first thing it says.

Two things are deliberately *not* here:

* **Postgres**, whose connection string - password included - is a command
  line argument rather than an environment variable. This interface displays
  the command line, so adding it would put a password on screen.
* **Servers that duplicate a built-in tool without beating it.** There is
  already a workspace-jailed filesystem tool and a memory subsystem; a second
  one that does the same thing worse is a schema the model pays for on every
  turn.

`{workspace}` in an argument is replaced with the workspace path when the
server is switched on, so the filesystem server is scoped the way the built-in
file tools are rather than to the whole disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EnvNeed:
    """One credential a server cannot start without."""

    # The environment variable the server reads.
    variable: str
    label: str
    # Where to get one. The single most useful sentence on the whole form.
    hint: str = ""


@dataclass(frozen=True)
class CatalogEntry:
    """One offered server: how to start it, and what it costs to say yes."""

    name: str
    title: str
    # What it gives the model, in the interface's voice.
    summary: str
    command: str = ""
    args: tuple[str, ...] = ()
    # Set instead of `command` for a server that is somebody else's, reached
    # over Streamable HTTP rather than started here. One or the other.
    url: str = ""
    # Sent with every request to a remote one. `${VAR}` is read from the
    # environment when the server is reached.
    headers: dict[str, str] = field(default_factory=dict)
    # "node" or "python" - what has to be installed for it to start at all.
    # "none" for a remote server: nothing is installed and nothing runs here.
    runtime: str = "node"
    # The package resolved on first use, shown so it can be checked before
    # agreeing to run it.
    package: str = ""
    # Anything someone should know before switching it on: what it can reach,
    # what it changes. Empty when there is genuinely nothing to flag.
    caution: str = ""
    # Credentials, if it reaches a service. Collected when it is switched on.
    needs: tuple[EnvNeed, ...] = ()
    # Published as "no longer supported". Shown as such rather than hidden:
    # these still install and run, and there is no stdio replacement for
    # several of them.
    unmaintained: bool = False
    env: dict[str, str] = field(default_factory=dict)

    @property
    def remote(self) -> bool:
        return bool(self.url)

    def resolve(self, workspace: Path) -> dict[str, object]:
        """The mcp.json entry for this server, with placeholders filled in.

        A url entry and a command entry are different shapes, and writing both
        keys would produce exactly the ambiguous record `load_servers` refuses
        to guess at - so this writes one or the other, never both.
        """
        if self.remote:
            return {
                "url": self.url,
                **({"headers": dict(self.headers)} if self.headers else {}),
                **({"env": dict(self.env)} if self.env else {}),
            }
        return {
            "command": self.command,
            "args": [
                arg.replace("{workspace}", str(workspace)) for arg in self.args
            ],
            **({"env": dict(self.env)} if self.env else {}),
        }


CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        name="exa",
        title="Web search",
        summary=(
            "Searches the web and returns clean text, and reads a page in "
            "full. The real answer to this project not having web access."
        ),
        # Exa's own hosted server, over Streamable HTTP. Nothing is installed
        # and nothing runs on this machine, which is why the runtime is
        # "none" and there is no package to check.
        url="https://mcp.exa.ai/mcp",
        runtime="none",
        package="mcp.exa.ai (Exa's hosted server)",
        caution=(
            "Every query and every page it reads leaves this machine and "
            "goes to Exa. Both of its tools are annotated read-only, so "
            "they run without asking - switch this on only if that is what "
            "you want."
        ),
    ),
    CatalogEntry(
        name="fetch",
        title="Fetch a web page",
        summary=(
            "Retrieves a URL and converts it to markdown for the model to "
            "read. The closest thing here to web access."
        ),
        command="uvx",
        args=("mcp-server-fetch",),
        runtime="python",
        package="mcp-server-fetch (PyPI)",
        caution=(
            "Reaches the public internet. Whatever it fetches becomes text "
            "the model reads and may act on."
        ),
    ),
    CatalogEntry(
        name="git",
        title="Git history and diffs",
        summary=(
            "Reads a repository's log, status and diffs, and can stage and "
            "commit in it."
        ),
        command="uvx",
        args=("mcp-server-git", "--repository", "{workspace}"),
        runtime="python",
        package="mcp-server-git (PyPI)",
        caution=(
            "Scoped to the workspace, and it can commit. The built-in git "
            "tools cover reading already - this adds a second way to do it."
        ),
    ),
    CatalogEntry(
        name="filesystem",
        title="Files, outside the workspace jail",
        summary=(
            "Reads, writes, moves and searches files under the folders it is "
            "given."
        ),
        command="npx",
        args=("-y", "@modelcontextprotocol/server-filesystem", "{workspace}"),
        package="@modelcontextprotocol/server-filesystem (npm)",
        caution=(
            "Pointed at the workspace, but it is a separate jail from the "
            "built-in file tools and it can delete and move, which they "
            "cannot. Changing its argument changes what it reaches."
        ),
    ),
    CatalogEntry(
        name="sequential-thinking",
        title="Step-by-step reasoning",
        summary=(
            "A scratchpad the model works a problem through in, one step at "
            "a time, revising as it goes."
        ),
        command="npx",
        args=("-y", "@modelcontextprotocol/server-sequential-thinking"),
        package="@modelcontextprotocol/server-sequential-thinking (npm)",
        caution=(
            "Costs turns rather than permissions: it reaches nothing, but "
            "each step is a round trip, and a round trip here is seconds."
        ),
    ),
    CatalogEntry(
        name="everything",
        title="The MCP test server",
        summary=(
            "The reference server, exercising every part of the protocol. "
            "Useful for checking this end works."
        ),
        command="npx",
        args=("-y", "@modelcontextprotocol/server-everything"),
        package="@modelcontextprotocol/server-everything (npm)",
        caution="A demonstration, not something to leave switched on.",
    ),
    # --- external services ---
    #
    # All of these are archived upstream. See the module docstring: the
    # vendors moved their integrations to remote HTTP servers, which this
    # client does not speak. They install and work; they are not maintained.
    CatalogEntry(
        name="github",
        title="GitHub",
        summary=(
            "Reads and searches repositories, issues and pull requests, and "
            "can open and comment on them."
        ),
        command="npx",
        args=("-y", "@modelcontextprotocol/server-github"),
        package="@modelcontextprotocol/server-github (npm)",
        unmaintained=True,
        caution=(
            "It can write: opening issues and pull requests, pushing files. "
            "Scope the token to the repositories you mean."
        ),
        needs=(
            EnvNeed(
                variable="GITHUB_PERSONAL_ACCESS_TOKEN",
                label="Personal access token",
                hint=(
                    "github.com → Settings → Developer settings → Personal "
                    "access tokens. Give it the narrowest scope that works."
                ),
            ),
        ),
    ),
    CatalogEntry(
        name="slack",
        title="Slack",
        summary=(
            "Lists channels, reads their history and posts messages and "
            "replies as the bot user."
        ),
        command="npx",
        args=("-y", "@modelcontextprotocol/server-slack"),
        package="@modelcontextprotocol/server-slack (npm)",
        unmaintained=True,
        caution=(
            "It can post. Anything it writes appears to your colleagues as a "
            "message from your workspace's bot."
        ),
        needs=(
            EnvNeed(
                variable="SLACK_BOT_TOKEN",
                label="Bot token",
                hint="Starts xoxb-. From your Slack app's OAuth settings.",
            ),
            EnvNeed(
                variable="SLACK_TEAM_ID",
                label="Team ID",
                hint="Starts T. Slack → About this workspace.",
            ),
        ),
    ),
    CatalogEntry(
        name="gitlab",
        title="GitLab",
        summary="Reads projects, issues and merge requests, and can edit them.",
        command="npx",
        args=("-y", "@modelcontextprotocol/server-gitlab"),
        package="@modelcontextprotocol/server-gitlab (npm)",
        unmaintained=True,
        caution="It can write to the projects the token reaches.",
        needs=(
            EnvNeed(
                variable="GITLAB_PERSONAL_ACCESS_TOKEN",
                label="Personal access token",
                hint="GitLab → Preferences → Access tokens.",
            ),
            EnvNeed(
                variable="GITLAB_API_URL",
                label="API URL",
                hint="Only for self-hosted. Otherwise leave this blank.",
            ),
        ),
    ),
    CatalogEntry(
        name="brave-search",
        title="Web search",
        summary=(
            "Searches the web through Brave and returns results the model "
            "can read. The closest thing here to a search tool."
        ),
        command="npx",
        args=("-y", "@modelcontextprotocol/server-brave-search"),
        package="@modelcontextprotocol/server-brave-search (npm)",
        unmaintained=True,
        caution=(
            "Every query leaves this machine, and the results are text the "
            "model reads and may act on."
        ),
        needs=(
            EnvNeed(
                variable="BRAVE_API_KEY",
                label="API key",
                hint="From api-dashboard.search.brave.com. A free tier exists.",
            ),
        ),
    ),
    CatalogEntry(
        name="google-maps",
        title="Maps and places",
        summary="Geocoding, directions, and details of places.",
        command="npx",
        args=("-y", "@modelcontextprotocol/server-google-maps"),
        package="@modelcontextprotocol/server-google-maps (npm)",
        unmaintained=True,
        caution="Read-only, but every lookup is a billable API call.",
        needs=(
            EnvNeed(
                variable="GOOGLE_MAPS_API_KEY",
                label="API key",
                hint="Google Cloud console, with the Maps APIs enabled.",
            ),
        ),
    ),
)

BY_NAME = {entry.name: entry for entry in CATALOG}


def entry(name: str) -> CatalogEntry | None:
    return BY_NAME.get((name or "").strip().lower())
