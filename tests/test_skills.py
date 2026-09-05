"""Instruction packs: finding them, reading them, offering them.

A skill is how-we-do-this-here, written down once. It reaches the model as a
tool result rather than as part of the prompt, because the prompt prefix is
what llama.cpp's cache is keyed on and rewriting it costs a full re-read -
around 200 seconds on this machine. Appending to the end costs nothing.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from agent.loop import Agent
from config import Config
from tests.fake_client import FakeQwenClient, text_message, tool_call_message
from tools.base import Tool, ToolRegistry
from tools.registry import build_default_registry
from tools.skills import (
    LOAD_SKILL,
    MAX_SKILL_CHARS,
    NEEDS_TOOLS,
    SkillError,
    SkillLibrary,
    discover,
    parse_skill,
)


class ParseTests(unittest.TestCase):
    def test_frontmatter_wins(self):
        name, description, body, _ = parse_skill(
            "---\nname: writing-tests\ndescription: How we test here\n---\n\nBody.",
            "folder-name",
        )
        self.assertEqual(name, "writing-tests")
        self.assertEqual(description, "How we test here")
        self.assertEqual(body, "Body.")

    def test_quotes_around_values_come_off(self):
        name, description, _, _ = parse_skill(
            '---\nname: "quoted"\ndescription: \'also quoted\'\n---\nx', "f"
        )
        self.assertEqual(name, "quoted")
        self.assertEqual(description, "also quoted")

    def test_a_plain_markdown_file_still_works(self):
        """Someone's first attempt will be a file with no frontmatter."""
        name, description, body, _ = parse_skill(
            "# Balancing equations\n\nAlways check the charges.", "chemistry"
        )
        self.assertEqual(name, "chemistry")
        self.assertEqual(description, "Balancing equations")
        self.assertTrue(body.startswith("# Balancing equations"))

    def test_the_name_is_lowercased(self):
        name, _, _, _ = parse_skill("---\nname: Loud-Name\n---\nx", "f")
        self.assertEqual(name, "loud-name")


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_a_folder_with_a_skill_file(self):
        self.write("balancing/SKILL.md", "---\nname: balancing\n---\nHow to balance.")
        self.assertEqual([s.name for s in discover(self.root)], ["balancing"])

    def test_a_bare_markdown_file(self):
        self.write("quick.md", "Just a note about something.")
        self.assertEqual([s.name for s in discover(self.root)], ["quick"])

    def test_a_readme_is_not_a_skill(self):
        """The folder wants a README explaining what skills are."""
        self.write("README.md", "These are skills.")
        self.assertEqual(discover(self.root), [])

    def test_an_empty_file_is_not_a_skill(self):
        self.write("empty.md", "   \n\n")
        self.assertEqual(discover(self.root), [])

    def test_a_name_that_is_not_an_identifier_is_skipped(self):
        self.write("bad.md", "---\nname: Not A Valid Name!\n---\nbody")
        self.assertEqual(discover(self.root), [])

    def test_a_missing_directory_is_not_an_error(self):
        self.assertEqual(discover(self.root / "nowhere"), [])

    def test_a_long_body_is_capped(self):
        self.write("long/SKILL.md", "---\nname: long\n---\n" + "x" * 50_000)
        skill = discover(self.root)[0]
        self.assertEqual(len(skill.body), MAX_SKILL_CHARS)
        self.assertTrue(skill.truncated)

    def test_results_are_sorted_by_name(self):
        for name in ("zebra", "alpha", "middle"):
            self.write(f"{name}.md", f"About {name}.")
        self.assertEqual(
            [s.name for s in discover(self.root)], ["alpha", "middle", "zebra"]
        )


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "greeting.md").write_text(
            "---\nname: greeting\ndescription: How to greet\n---\nSay hello warmly.",
            encoding="utf-8",
        )
        self.library = SkillLibrary(self.root)

    def test_loading_returns_the_body_and_an_instruction(self):
        loaded = self.library.load("greeting")
        self.assertTrue(loaded["success"])
        self.assertEqual(loaded["instructions"], "Say hello warmly.")
        # A model handed instructions should act on them, not recite them.
        self.assertIn("Follow these", loaded["note"])

    def test_the_name_is_matched_case_insensitively(self):
        self.assertTrue(self.library.load("GREETING")["success"])

    def test_an_unknown_skill_lists_what_there_is(self):
        with self.assertRaises(SkillError) as caught:
            self.library.load("nonexistent")
        self.assertIn("greeting", str(caught.exception))

    def test_the_tool_carries_the_index_and_an_enum(self):
        """The index is the discovery mechanism; the enum stops invention."""
        tool = self.library.tool()
        self.assertIn("greeting: How to greet", tool.description)
        self.assertEqual(
            tool.parameters["properties"]["name"]["enum"], ["greeting"]
        )

    def test_an_empty_library_reports_itself_empty(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(len(SkillLibrary(Path(empty))), 0)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def registry(self, skills_dir: Path):
        import dataclasses

        config = dataclasses.replace(
            Config(), workspace=self.root, skills_dir=skills_dir
        )
        return build_default_registry(config)[0]

    def test_no_skills_means_no_tool(self):
        """An index of nothing is a schema paid for on every request."""
        registry = self.registry(self.root / "absent")
        self.assertNotIn("load_skill", registry.names())

    def test_one_skill_registers_the_tool(self):
        skills = self.root / "skills"
        skills.mkdir()
        (skills / "a.md").write_text("Something useful.", encoding="utf-8")

        self.assertIn("load_skill", self.registry(skills).names())

    def test_the_lens_offers_it_from_the_start(self):
        """A model cannot ask for instructions it was never told exist."""
        from tools.lens import ToolLens

        skills = self.root / "skills"
        skills.mkdir()
        (skills / "a.md").write_text("Something useful.", encoding="utf-8")

        lens = ToolLens(self.registry(skills))
        offered = {d["function"]["name"] for d in lens.definitions()}
        self.assertIn("load_skill", offered)


def skill(*frontmatter: str, body: str = "Use it.") -> str:
    """A SKILL.md, written the way one is written."""
    return "\n".join(["---", *frontmatter, "---", body])


class ToolDeclarationTests(unittest.TestCase):
    """`tools:` in the frontmatter - what counts as a group name."""

    def parse(self, line: str) -> tuple[str, ...]:
        _, _, _, tools = parse_skill(skill("name: x", line), "f")
        return tools

    def test_a_comma_separated_list(self):
        self.assertEqual(
            self.parse("tools: python, filesystem"), ("python", "filesystem")
        )

    def test_a_yaml_style_list_means_the_same_thing(self):
        """Both are things people write, so both work."""
        self.assertEqual(
            self.parse("tools: [python, filesystem]"), ("python", "filesystem")
        )

    def test_a_missing_comma_is_not_a_silent_failure(self):
        self.assertEqual(
            self.parse("tools: python filesystem"), ("python", "filesystem")
        )

    def test_names_are_lowercased_and_junk_is_dropped(self):
        self.assertEqual(self.parse("tools: Python, foo!bar, git"), ("python", "git"))

    def test_duplicates_collapse_and_order_is_kept(self):
        self.assertEqual(self.parse("tools: git, python, git"), ("git", "python"))

    def test_no_line_means_no_groups(self):
        self.assertEqual(self.parse("description: nothing here"), ())

    def test_an_mcp_server_group_is_a_valid_name(self):
        """MCP categories carry a colon; they are groups like any other."""
        self.assertEqual(self.parse("tools: mcp:files"), ("mcp:files",))


class DeclaredToolsReachTheLoopTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def library(self, *frontmatter: str) -> SkillLibrary:
        (self.root / "plotting.md").write_text(
            skill(*frontmatter), encoding="utf-8"
        )
        return SkillLibrary(self.root)

    def test_the_groups_travel_on_the_result(self):
        loaded = self.library("name: plotting", "tools: python").load("plotting")
        self.assertEqual(loaded[NEEDS_TOOLS], ["python"])

    def test_a_skill_declaring_none_carries_no_key_at_all(self):
        loaded = self.library("name: plotting").load("plotting")
        self.assertNotIn(NEEDS_TOOLS, loaded)


class ThroughTheAgentTests(unittest.TestCase):
    """Loading a skill opens the tool groups it named.

    Instructions that say "use matplotlib" are no use to a model that cannot
    see the python tool, and the round trip it would otherwise spend on
    `load_tools` finding that out is one the skill already knew about.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def registry(self, *frontmatter: str) -> ToolRegistry:
        (self.root / "plotting.md").write_text(
            skill(*frontmatter), encoding="utf-8"
        )

        def run(**_):
            return {"success": True}

        schema = {"type": "object", "properties": {}}
        return ToolRegistry(
            [
                Tool("calculate", "calculator", "adds", schema, run),
                Tool("run_python", "python", "runs", schema, run),
                Tool("git_status", "git", "status", schema, run),
                SkillLibrary(self.root).tool(),
            ]
        )

    def run_turn(self, *frontmatter: str, lazy: bool = True):
        """One turn: load the skill, then reach for the tool it named."""
        client = FakeQwenClient(
            [
                tool_call_message((LOAD_SKILL, {"name": "plotting"})),
                tool_call_message(("run_python", {})),
                text_message("done"),
            ]
        )
        config = dataclasses.replace(Config(), lazy_tools=lazy)
        agent = Agent(client, config, self.registry(*frontmatter))

        # "chart it" on purpose: nothing in it matches a python signal or a
        # tool name, so the only route to that group is the skill.
        agent.send("chart it")

        payload = json.loads(
            next(m for m in agent.history if m.get("role") == "tool")["content"]
        )
        offered = [{d["function"]["name"] for d in seen} for seen in client.tools_seen]
        return payload, offered

    def test_the_named_group_arrives_on_the_next_request(self):
        _, offered = self.run_turn("name: plotting", "tools: python")

        self.assertNotIn("run_python", offered[0])
        self.assertIn("run_python", offered[1])

    def test_only_the_named_group_opens(self):
        _, offered = self.run_turn("name: plotting", "tools: python")

        self.assertNotIn("git_status", offered[1])

    def test_the_model_is_told_what_opened_and_when(self):
        """The schemas arrive with the next request, not this one."""
        payload, _ = self.run_turn("name: plotting", "tools: python")

        self.assertEqual(payload["loaded_tools"], ["python"])
        self.assertIn("Follow these", payload["note"])
        self.assertIn("available from your next message", payload["note"])

    def test_the_private_key_never_reaches_the_model(self):
        payload, _ = self.run_turn("name: plotting", "tools: python")

        self.assertNotIn(NEEDS_TOOLS, payload)

    def test_a_group_that_does_not_exist_is_not_claimed_to_have_opened(self):
        """A model told about a tool whose schema is not coming will call it."""
        payload, offered = self.run_turn("name: plotting", "tools: browser")

        self.assertNotIn("loaded_tools", payload)
        self.assertNotIn("available from your next message", payload["note"])
        self.assertNotIn("run_python", offered[1])

    def test_a_group_that_was_already_open_is_not_claimed_either(self):
        payload, _ = self.run_turn("name: plotting", "tools: calculator")

        self.assertNotIn("loaded_tools", payload)

    def test_with_the_lens_off_there_is_nothing_to_open(self):
        """Every tool is in every request already; the key still has to go."""
        payload, offered = self.run_turn(
            "name: plotting", "tools: python", lazy=False
        )

        self.assertNotIn(NEEDS_TOOLS, payload)
        self.assertNotIn("loaded_tools", payload)
        self.assertIn("run_python", offered[0])

if __name__ == "__main__":
    unittest.main()
