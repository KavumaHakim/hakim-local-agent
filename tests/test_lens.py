"""The tool lens: what a turn sends, and what opens a group.

The point of the lens is a smaller prompt, so one test measures that rather
than trusting it. The rest are about the property that makes it safe on this
hardware: opening is monotonic, because the tool list sits in the prompt
prefix and changing it costs a full re-read.
"""

from __future__ import annotations

import dataclasses
import json
import unittest

from agent.loop import Agent
from config import Config
from tests.fake_client import FakeQwenClient, text_message, tool_call_message
from tools.base import Tool, ToolRegistry
from tools.lens import LOAD_TOOLS, ToolLens
from tools.registry import build_default_registry


def full_registry() -> ToolRegistry:
    """Every optional tool switched on - the case the lens exists for."""
    flags = {
        field.name: True
        for field in dataclasses.fields(Config)
        if field.type == "bool"
        and ("enabled" in field.name or "allow" in field.name)
    }
    registry, _ = build_default_registry(dataclasses.replace(Config(), **flags))
    return registry


def tokens(definitions) -> int:
    """Rough token count. Only ever compared against another of these."""
    return len(json.dumps(definitions)) // 4


def names(definitions) -> set[str]:
    return {d["function"]["name"] for d in definitions}


def tiny_registry() -> ToolRegistry:
    def run(**_):
        return {"success": True}

    schema = {"type": "object", "properties": {}}
    return ToolRegistry(
        [
            Tool("calculate", "calculator", "adds", schema, run),
            Tool("git_status", "git", "status", schema, run),
            Tool("git_log", "git", "log", schema, run),
            Tool("ocr_image", "ocr", "reads", schema, run),
        ]
    )


def fs_registry() -> ToolRegistry:
    """A roster with filesystem in it, for the signals that were too eager."""

    def run(**_):
        return {"success": True}

    schema = {"type": "object", "properties": {}}
    return ToolRegistry(
        [
            Tool("read_text_file", "filesystem", "reads", schema, run),
            Tool("ocr_image", "ocr", "reads", schema, run),
        ]
    )


class WhatATurnSendsTests(unittest.TestCase):
    def test_it_is_much_smaller_than_the_whole_roster(self):
        registry = full_registry()
        whole = registry.get_tool_definitions()
        lens = ToolLens(registry)

        opening = lens.definitions()

        # The saving is the entire justification, so it is asserted rather
        # than described. The real numbers at the time of writing are about
        # 3,060 tokens against about 410.
        self.assertLess(tokens(opening), tokens(whole) / 4)

    def test_the_index_names_every_closed_group(self):
        lens = ToolLens(tiny_registry())

        index = [d for d in lens.definitions() if d["function"]["name"] == LOAD_TOOLS]
        self.assertEqual(len(index), 1)

        text = index[0]["function"]["description"]
        for category in ("git", "ocr"):
            self.assertIn(category, text)
        # The enum is what stops the model inventing a group name.
        enum = index[0]["function"]["parameters"]["properties"]["categories"]["items"]
        self.assertEqual(set(enum["enum"]), {"git", "ocr"})

    def test_a_closed_group_sends_no_schemas(self):
        lens = ToolLens(tiny_registry())
        self.assertEqual(names(lens.definitions()), {"calculate", LOAD_TOOLS})

    def test_the_index_disappears_once_nothing_is_closed(self):
        lens = ToolLens(tiny_registry())
        lens.open_categories_by_name(["git", "ocr"])

        sent = names(lens.definitions())
        self.assertNotIn(LOAD_TOOLS, sent)
        self.assertEqual(sent, {"calculate", "git_status", "git_log", "ocr_image"})


class OpeningTests(unittest.TestCase):
    def test_a_message_opens_what_it_names(self):
        lens = ToolLens(tiny_registry())
        self.assertEqual(lens.consider("what does git diff say?"), {"git"})
        self.assertIn("git_status", names(lens.definitions()))

    def test_an_unrelated_message_opens_nothing(self):
        lens = ToolLens(tiny_registry())
        self.assertEqual(lens.consider("who was Ada Lovelace?"), set())
        self.assertEqual(names(lens.definitions()), {"calculate", LOAD_TOOLS})

    def test_matching_is_whole_word(self):
        """'digit' contains 'git', and must not open the repository tools."""
        lens = ToolLens(tiny_registry())
        self.assertEqual(lens.consider("how many digits are in a postcode?"), set())

    def test_opening_is_monotonic(self):
        """The prefix cache is why: a group that closed again costs a re-read."""
        lens = ToolLens(tiny_registry())
        lens.consider("check the git log")
        self.assertIn("git", lens.open_categories)

        lens.consider("now tell me a joke")

        self.assertIn("git", lens.open_categories)
        self.assertIn("git_log", names(lens.definitions()))

    def test_a_group_the_roster_does_not_have_never_opens(self):
        """Signals exist for tools that are switched off in this config."""
        lens = ToolLens(tiny_registry())
        lens.consider("read the pdf documents in that folder")

        self.assertNotIn("documents", lens.open_categories)
        self.assertNotIn("filesystem", lens.open_categories)


class LoadToolsCallTests(unittest.TestCase):
    def test_it_opens_and_says_what_arrived(self):
        lens = ToolLens(tiny_registry())

        result = lens.load({"categories": ["git"]})

        self.assertTrue(result["success"])
        self.assertEqual(result["loaded"], ["git"])
        self.assertEqual(result["tools"], ["git_log", "git_status"])
        self.assertIn("git_status", names(lens.definitions()))

    def test_a_bare_string_is_accepted(self):
        """Small models send one string where an array was asked for."""
        lens = ToolLens(tiny_registry())
        self.assertTrue(lens.load({"categories": "ocr"})["success"])
        self.assertIn("ocr", lens.open_categories)

    def test_an_unknown_group_fails_without_opening_anything(self):
        lens = ToolLens(tiny_registry())

        result = lens.load({"categories": ["telepathy"]})

        self.assertFalse(result["success"])
        self.assertIn("telepathy", result["error"])
        self.assertEqual(lens.open_categories, {"calculator"})

    def test_nothing_asked_for_is_a_failure(self):
        lens = ToolLens(tiny_registry())
        for arguments in ({}, {"categories": []}, {"categories": None}):
            self.assertFalse(lens.load(arguments)["success"], arguments)

    def test_asking_twice_reports_the_second_as_already_loaded(self):
        lens = ToolLens(tiny_registry())
        lens.load({"categories": ["git"]})

        again = lens.load({"categories": ["git"]})

        self.assertEqual(again["loaded"], [])
        self.assertEqual(again["already_loaded"], ["git"])


class TwoConversationsTests(unittest.TestCase):
    def test_lenses_sharing_a_registry_stay_independent(self):
        """The registry is shared between conversations; what is open is not."""
        registry = tiny_registry()
        first, second = ToolLens(registry), ToolLens(registry)

        first.load({"categories": ["git"]})

        self.assertIn("git", first.open_categories)
        self.assertNotIn("git", second.open_categories)

    def test_building_a_lens_does_not_add_a_tool_to_the_registry(self):
        registry = tiny_registry()
        before = registry.names()

        ToolLens(registry)

        self.assertEqual(registry.names(), before)
        self.assertNotIn(LOAD_TOOLS, registry)


class FilesystemSignalTests(unittest.TestCase):
    """`read`, `write` and `save` are ordinary verbs, and used to misfire."""

    def opened_by(self, prompt: str) -> set[str]:
        lens = ToolLens(fs_registry())
        lens.consider(prompt)
        return lens.open_categories

    def test_an_ordinary_verb_does_not_open_the_filesystem(self):
        for prompt in (
            "write a haiku about rain",
            "read me a poem",
            "save the day",
            "write a function that reverses a list",
        ):
            self.assertNotIn("filesystem", self.opened_by(prompt), prompt)

    def test_a_filename_opens_the_filesystem(self):
        for prompt in (
            "read config.py and tell me the defaults",
            "what is in notes.md?",
            "open package.json",
        ):
            self.assertIn("filesystem", self.opened_by(prompt), prompt)

    def test_a_path_opens_the_filesystem(self):
        """Both separators: this is a Windows machine talking to a POSIX repo."""
        for prompt in ("look in web/src for it", "look in web\src for it"):
            self.assertIn("filesystem", self.opened_by(prompt), prompt)

    def test_the_nouns_still_work(self):
        for prompt in ("list the files", "what is in that folder?"):
            self.assertIn("filesystem", self.opened_by(prompt), prompt)


class ThroughTheAgentTests(unittest.TestCase):
    """The loop end to end, with the schemas the model actually received."""

    def agent(self, client, *, lazy: bool) -> Agent:
        config = dataclasses.replace(Config(), lazy_tools=lazy)
        return Agent(client, config, tiny_registry())

    def test_off_by_default_sends_the_whole_roster(self):
        client = FakeQwenClient([text_message("hello")])

        self.agent(client, lazy=False).send("hello")

        self.assertEqual(
            names(client.tools_seen[0]),
            {"calculate", "git_status", "git_log", "ocr_image"},
        )

    def test_the_model_asks_for_a_group_and_uses_it_next_round(self):
        client = FakeQwenClient(
            [
                tool_call_message((LOAD_TOOLS, {"categories": ["git"]})),
                tool_call_message(("git_status", {})),
                text_message("the tree is clean"),
            ]
        )
        agent = self.agent(client, lazy=True)

        # Deliberately says nothing the heuristic matches, so the only route
        # to the git tools is the model asking for them.
        turn = agent.send("is it clean?")

        first, second = names(client.tools_seen[0]), names(client.tools_seen[1])
        self.assertIn(LOAD_TOOLS, first)
        self.assertNotIn("git_status", first)
        self.assertIn("git_status", second)
        self.assertEqual(turn.content, "the tree is clean")

    def test_the_heuristic_spends_no_round_trip(self):
        client = FakeQwenClient(
            [tool_call_message(("git_status", {})), text_message("clean")]
        )
        agent = self.agent(client, lazy=True)

        agent.send("what does git status say?")

        # The git schemas were there on the very first request.
        self.assertIn("git_status", names(client.tools_seen[0]))

    def test_what_opened_survives_the_next_message(self):
        client = FakeQwenClient([text_message("ok")], repeat_last=True)
        agent = self.agent(client, lazy=True)

        agent.send("check the git log")
        agent.send("thanks")

        self.assertIn("git_status", names(client.tools_seen[1]))


if __name__ == "__main__":
    unittest.main()


class ToolNameSignalTests(unittest.TestCase):
    """A tool's own name opens its group.

    A model that has been told a tool exists asks for it by name, and the
    hand-written signals did not always contain that name: a live turn asking
    for `run_command` left the terminal group closed, and the model answered
    that it could only evaluate mathematical expressions.
    """

    def test_naming_a_tool_opens_its_category(self):
        lens = ToolLens(tiny_registry())
        self.assertEqual(lens.consider("please use git_status"), {"git"})

    def test_it_is_derived_from_the_registry_not_a_list(self):
        """A tool added later brings its own signal without anyone listing it."""

        def run(**_):
            return {"success": True}

        registry = ToolRegistry(
            [
                Tool("calculate", "calculator", "adds", {"type": "object"}, run),
                Tool("brand_new_tool", "invented", "does", {"type": "object"}, run),
            ]
        )
        lens = ToolLens(registry)
        self.assertEqual(lens.consider("call brand_new_tool for me"), {"invented"})

    def test_a_name_inside_a_longer_word_does_not_count(self):
        lens = ToolLens(tiny_registry())
        self.assertEqual(lens.consider("the git_statuses are confusing"), set())


class TerminalSignalTests(unittest.TestCase):
    """Command names are how people ask for commands."""

    def registry(self) -> ToolRegistry:
        def run(**_):
            return {"success": True}

        schema = {"type": "object", "properties": {}}
        return ToolRegistry(
            [
                Tool("calculate", "calculator", "adds", schema, run),
                Tool("run_command", "terminal", "runs", schema, run),
            ]
        )

    def opened_by(self, prompt: str) -> set[str]:
        lens = ToolLens(self.registry())
        lens.consider(prompt)
        return lens.open_categories

    def test_a_named_command_opens_the_terminal(self):
        for prompt in (
            "run this terminal command: mkdir reports",
            "npm install left-pad",
            "curl the api",
            "what does docker ps say",
            "check whoami",
        ):
            self.assertIn("terminal", self.opened_by(prompt), prompt)

    def test_ordinary_english_does_not(self):
        """`make`, `find`, `sort`, `date` and `file` are words as well as
        programs, so they are deliberately not signals."""
        for prompt in (
            "make it shorter",
            "find the error in my reasoning",
            "sort these ideas by importance",
            "what is the date of the moon landing?",
            "write a haiku about rain",
        ):
            self.assertNotIn("terminal", self.opened_by(prompt), prompt)


def mcp_registry() -> ToolRegistry:
    """A roster with an MCP server in it, named the way McpManager names them.

    The prefix matters to every test here: a server's tools are registered as
    `<server>__<tool>`, and its category as `mcp:<server>`.
    """

    def run(**_):
        return {"success": True}

    schema = {"type": "object", "properties": {}}
    return ToolRegistry(
        [
            Tool("calculate", "calculator", "adds", schema, run),
            Tool(
                "exa__web_search_exa",
                "mcp:exa",
                "Search the web for any topic and get clean, ready-to-use "
                "content.\n\n   Best for: finding current information.",
                schema,
                run,
            ),
            Tool(
                "exa__web_fetch_exa",
                "mcp:exa",
                "Read a webpage's full content as clean markdown.",
                schema,
                run,
            ),
        ]
    )


def index_text(lens: ToolLens) -> str:
    for definition in lens.definitions():
        if definition["function"]["name"] == LOAD_TOOLS:
            return definition["function"]["description"]
        continue
    raise AssertionError("the index is not being sent")


class McpInTheIndexTests(unittest.TestCase):
    """What the model is told about a server it has not loaded yet.

    CATEGORY_HELP is hand-written and keyed on the built-in category names.
    An MCP server's category is `mcp:<whatever is in mcp.json>`, so it never
    had an entry, and the line came out as `- mcp:exa (2 tools):` with
    nothing after the colon. The model was shown a group it could load and
    told nothing whatever about it, which is how a run ends with the model
    saying it has no way to search the web while exa sits in the index.
    """

    def test_a_server_line_says_what_the_server_does(self):
        line = next(
            row
            for row in index_text(ToolLens(mcp_registry())).splitlines()
            if row.startswith("- mcp:exa")
        )

        self.assertNotEqual(line.strip(), "- mcp:exa (2 tools):")
        self.assertIn("Search the web", line)

    def test_it_names_every_tool_in_the_server(self):
        line = index_text(ToolLens(mcp_registry()))
        self.assertIn("web_search_exa", line)
        self.assertIn("web_fetch_exa", line)

    def test_the_server_prefix_is_not_repeated_on_each_tool(self):
        """`mcp:exa` already names the server; `exa__web_search_exa` in the
        same line pays for it twice."""
        self.assertNotIn("exa__", index_text(ToolLens(mcp_registry())))

    def test_only_the_first_sentence_of_a_description_is_used(self):
        """MCP descriptions are written for a documentation pane - several
        paragraphs of usage notes under a one-line summary."""
        self.assertNotIn("Best for", index_text(ToolLens(mcp_registry())))

    def test_a_built_in_still_uses_its_written_line(self):
        line = index_text(ToolLens(fs_registry()))
        self.assertIn("read, list, write and create files", line)

    def test_a_long_server_is_bounded(self):
        """The lens exists to keep the prompt small, so a server with thirty
        tools must not spend the saving on its own index line."""

        def run(**_):
            return {"success": True}

        schema = {"type": "object", "properties": {}}
        registry = ToolRegistry(
            [Tool("calculate", "calculator", "adds", schema, run)]
            + [
                Tool(
                    f"gh__issue_{n}",
                    "mcp:gh",
                    f"Does the {n}th thing with issues, described at length.",
                    schema,
                    run,
                )
                for n in range(30)
            ]
        )
        line = next(
            row
            for row in index_text(ToolLens(registry)).splitlines()
            if row.startswith("- mcp:gh")
        )
        self.assertLess(len(line), 240)

    def test_a_bounded_line_ends_on_a_whole_tool_name(self):
        """Clipping mid-name would leave something that reads like a tool
        name and is not one, which the model then tries to call."""

        def run(**_):
            return {"success": True}

        schema = {"type": "object", "properties": {}}
        registry = ToolRegistry(
            [Tool("calculate", "calculator", "adds", schema, run)]
            + [
                Tool(f"gh__a_rather_long_tool_name_{n}", "mcp:gh", "Does it.", schema, run)
                for n in range(30)
            ]
        )
        line = index_text(ToolLens(registry))
        self.assertNotIn("\u2026", line)
        for shown in line.split("- mcp:gh (30 tools): ")[1].split(", "):
            self.assertIn(shown.split(" (")[0], {t.name[4:] for t in registry.list_tools()})


class McpSignalTests(unittest.TestCase):
    """What opens a server's group without the model spending a round trip.

    Nothing did. The only derived signal was the registered name,
    `exa__web_search_exa`, which nobody writes - so a server could only ever
    be opened by the model reading the index and calling `load_tools`.
    """

    def opened_by(self, prompt: str) -> set[str]:
        lens = ToolLens(mcp_registry())
        return lens.consider(prompt)

    def test_the_server_name_opens_it(self):
        self.assertEqual(self.opened_by("ask exa about GGUF quantisation"), {"mcp:exa"})

    def test_the_bare_tool_name_opens_it(self):
        """What the server calls the tool, and what a skill file would write."""
        self.assertEqual(self.opened_by("use web_search_exa for this"), {"mcp:exa"})

    def test_the_registered_name_still_opens_it(self):
        self.assertEqual(self.opened_by("call exa__web_fetch_exa"), {"mcp:exa"})

    def test_an_unrelated_message_opens_nothing(self):
        self.assertEqual(self.opened_by("write me a haiku about rain"), set())

    def test_a_server_cannot_take_a_name_a_real_tool_owns(self):
        """A server offering `read_text_file` must not stop that phrase
        opening the filesystem group."""

        def run(**_):
            return {"success": True}

        schema = {"type": "object", "properties": {}}
        registry = ToolRegistry(
            [
                Tool("read_text_file", "filesystem", "reads", schema, run),
                Tool("files__read_text_file", "mcp:files", "also reads", schema, run),
            ]
        )
        lens = ToolLens(registry)
        self.assertIn("filesystem", lens.consider("read_text_file the notes"))
