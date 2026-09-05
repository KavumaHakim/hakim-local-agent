"""Results too large for the window, and getting them back.

A tool result is capped at a quarter of the model's context - about 3,300
characters on a 4,096-token model. The cut was announced but the remainder was
gone, so "12,000 of 12,400 lines are not shown" was a dead end: the model
could not ask for the part it needed. Now the whole thing is kept and the
model is given its address.
"""

from __future__ import annotations

import dataclasses
import re
import tempfile
import unittest
from pathlib import Path

from agent.loop import Agent
from config import Config
from tests.fake_client import FakeQwenClient, text_message, tool_call_message
from tools.base import Tool, ToolRegistry
from tools.results import (
    MAX_STORED_FILES,
    MAX_WINDOW,
    ResultStore,
    ResultStoreError,
    build_result_tool,
)

SCHEMA = {"type": "object", "properties": {}}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ResultStore(Path(self._tmp.name) / "results")

    def test_saving_reports_the_size(self):
        stored = self.store.save("one\ntwo\nthree")
        self.assertEqual(stored.characters, 13)
        self.assertEqual(stored.lines, 3)
        self.assertRegex(stored.id, r"^r[0-9a-f]{6}$")

    def test_reading_returns_a_window_and_says_there_is_more(self):
        stored = self.store.save("abcdefghij" * 100)

        page = self.store.read(stored.id, offset=0, limit=10)

        self.assertEqual(page["text"], "abcdefghij")
        self.assertEqual(page["returned"], 10)
        self.assertEqual(page["total_characters"], 1000)
        self.assertTrue(page["more"])
        self.assertEqual(page["next_offset"], 10)

    def test_next_offset_walks_to_the_end(self):
        stored = self.store.save("x" * 25)
        seen, offset = "", 0
        while offset is not None:
            page = self.store.read(stored.id, offset=offset, limit=10)
            seen += page["text"]
            offset = page["next_offset"]
        self.assertEqual(seen, "x" * 25)

    def test_the_last_page_says_there_is_no_more(self):
        stored = self.store.save("short")
        page = self.store.read(stored.id)
        self.assertFalse(page["more"])
        self.assertIsNone(page["next_offset"])

    def test_a_window_is_capped(self):
        stored = self.store.save("y" * (MAX_WINDOW + 5000))
        page = self.store.read(stored.id, limit=10**9)
        self.assertEqual(page["returned"], MAX_WINDOW)

    def test_an_offset_past_the_end_returns_nothing_rather_than_failing(self):
        stored = self.store.save("short")
        page = self.store.read(stored.id, offset=10_000)
        self.assertEqual(page["text"], "")
        self.assertFalse(page["more"])

    def test_an_unknown_id_says_so(self):
        with self.assertRaises(ResultStoreError) as caught:
            self.store.read("rabcdef")
        self.assertIn("no stored result", str(caught.exception))

    def test_an_id_that_is_not_an_id_is_refused_without_touching_the_disk(self):
        """The id is generated here, so an invented one is not a lookup."""
        for bad in ("../../.ssh/config", "r3f9a1/../..", "", "hello", "R3F9A1"):
            with self.assertRaises(ResultStoreError) as caught:
                self.store.read(bad)
            self.assertIn("not a result id", str(caught.exception), bad)

    def test_saving_into_an_impossible_place_returns_none(self):
        """Offloading is an improvement on truncation; failing it is not fatal."""
        blocked = Path(self._tmp.name) / "afile"
        blocked.write_text("not a directory", encoding="utf-8")
        self.assertIsNone(ResultStore(blocked / "under").save("x"))

    def test_the_store_is_pruned(self):
        """This machine has run out of disk once already."""
        for index in range(MAX_STORED_FILES + 12):
            self.store.save(f"result {index}")
        kept = list((Path(self._tmp.name) / "results").glob("r*.json"))
        self.assertLessEqual(len(kept), MAX_STORED_FILES)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ResultStore(Path(self._tmp.name) / "results")
        self.tool = build_result_tool(self.store)

    def test_it_reads_through_the_registry(self):
        stored = self.store.save("hello there")
        registry = ToolRegistry([self.tool])

        result = registry.execute("read_result", {"id": stored.id})

        self.assertTrue(result.ok)
        self.assertEqual(result.payload["text"], "hello there")

    def test_a_bad_id_comes_back_as_a_refusal_not_a_crash(self):
        registry = ToolRegistry([self.tool])
        result = registry.execute("read_result", {"id": "nope"})
        self.assertFalse(result.ok)
        self.assertIn("not a result id", result.payload["error"])


class OffloadingTests(unittest.TestCase):
    """The loop's half: cut the result, keep the whole, hand over the address."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = ResultStore(self.root / "results")
        self.body = "".join(f"line {i} of a long file\n" for i in range(4000))

    def agent(self, *, results=True) -> tuple[Agent, ToolRegistry]:
        def big(**_):
            return {"success": True, "text": self.body}

        registry = ToolRegistry(
            [
                Tool("read_big", "filesystem", "reads", SCHEMA, big),
                build_result_tool(self.store),
            ]
        )
        config = dataclasses.replace(
            Config(), model_context=4096, results_dir=self.root / "results"
        )
        client = FakeQwenClient(
            [tool_call_message(("read_big", {})), text_message("done")]
        )
        return (
            Agent(
                client,
                config,
                registry,
                results=self.store if results else None,
            ),
            registry,
        )

    def tool_message(self, agent: Agent) -> str:
        return [m for m in agent.history if m.get("role") == "tool"][0]["content"]

    def test_the_model_is_given_an_id_it_can_use(self):
        agent, _ = self.agent()
        agent.send("read it")

        content = self.tool_message(agent)
        self.assertLess(len(content), len(self.body))
        found = re.search(r"'(r[0-9a-f]{6})'", content)
        self.assertIsNotNone(found, content[-300:])

        page = self.store.read(found.group(1))
        self.assertIn("line 0 of a long file", page["text"])

    def test_the_whole_result_is_kept_not_just_the_part_that_fitted(self):
        agent, _ = self.agent()
        agent.send("read it")

        found = re.search(r"'(r[0-9a-f]{6})'", self.tool_message(agent))
        page = self.store.read(found.group(1), offset=0, limit=MAX_WINDOW)
        self.assertGreater(page["total_characters"], len(self.body))

    def test_the_note_tells_the_model_not_to_answer_from_the_excerpt(self):
        agent, _ = self.agent()
        agent.send("read it")
        self.assertIn("Do not answer from the excerpt", self.tool_message(agent))

    def test_it_opens_the_results_group_so_the_tool_is_reachable(self):
        """`read_result` is not free, so it is offered once it is useful."""
        agent, _ = self.agent()
        self.assertNotIn("results", agent._lens.open_categories)

        agent.send("read it")

        self.assertIn("results", agent._lens.open_categories)

    def test_a_result_that_fits_is_left_alone(self):
        def small(**_):
            return {"success": True, "text": "tiny"}

        registry = ToolRegistry(
            [
                Tool("read_small", "filesystem", "reads", SCHEMA, small),
                build_result_tool(self.store),
            ]
        )
        config = dataclasses.replace(
            Config(), model_context=4096, results_dir=self.root / "results"
        )
        client = FakeQwenClient(
            [tool_call_message(("read_small", {})), text_message("done")]
        )
        agent = Agent(client, config, registry, results=self.store)
        agent.send("read it")

        self.assertNotIn("read_result", self.tool_message(agent))
        self.assertNotIn("results", agent._lens.open_categories)

    def test_without_a_store_it_behaves_as_before(self):
        """The CLI passes none; a cut result is still a cut result."""
        agent, _ = self.agent(results=False)
        agent.send("read it")

        content = self.tool_message(agent)
        self.assertNotIn("read_result", content)
        self.assertLess(len(content), len(self.body))


if __name__ == "__main__":
    unittest.main()
