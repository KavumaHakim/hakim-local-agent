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
