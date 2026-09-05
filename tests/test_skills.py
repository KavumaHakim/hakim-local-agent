"""Instruction packs: finding them, reading them, offering them.

A skill is how-we-do-this-here, written down once. It reaches the model as a
tool result rather than as part of the prompt, because the prompt prefix is
what llama.cpp's cache is keyed on and rewriting it costs a full re-read -
around 200 seconds on this machine. Appending to the end costs nothing.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import Config
from tools.registry import build_default_registry
from tools.skills import (
    MAX_SKILL_CHARS,
    SkillError,
    SkillLibrary,
    discover,
    parse_skill,
)


class ParseTests(unittest.TestCase):
    def test_frontmatter_wins(self):
        name, description, body = parse_skill(
            "---\nname: writing-tests\ndescription: How we test here\n---\n\nBody.",
            "folder-name",
        )
        self.assertEqual(name, "writing-tests")
        self.assertEqual(description, "How we test here")
        self.assertEqual(body, "Body.")

    def test_quotes_around_values_come_off(self):
        name, description, _ = parse_skill(
            '---\nname: "quoted"\ndescription: \'also quoted\'\n---\nx', "f"
        )
        self.assertEqual(name, "quoted")
        self.assertEqual(description, "also quoted")

    def test_a_plain_markdown_file_still_works(self):
        """Someone's first attempt will be a file with no frontmatter."""
        name, description, body = parse_skill(
            "# Balancing equations\n\nAlways check the charges.", "chemistry"
        )
        self.assertEqual(name, "chemistry")
        self.assertEqual(description, "Balancing equations")
        self.assertTrue(body.startswith("# Balancing equations"))

    def test_the_name_is_lowercased(self):
        name, _, _ = parse_skill("---\nname: Loud-Name\n---\nx", "f")
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


if __name__ == "__main__":
    unittest.main()
