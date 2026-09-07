"""Chat history store tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chat_store import ChatStore, make_title


class TitleTests(unittest.TestCase):
    def test_short_text_is_kept(self):
        self.assertEqual(make_title("Hello there"), "Hello there")

    def test_whitespace_is_collapsed(self):
        self.assertEqual(make_title("  a\n\n  b  "), "a b")

    def test_long_text_is_truncated(self):
        title = make_title("word " * 40)
        self.assertLessEqual(len(title), 60)
        self.assertTrue(title.endswith("…"))

    def test_empty_text_gets_a_placeholder(self):
        self.assertEqual(make_title("   "), "New conversation")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(Path(self._tmp.name) / "history.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_database_file_is_created(self):
        self.assertTrue(self.store.path.exists())

    def test_reopening_an_existing_database_is_safe(self):
        conversation = self.store.create_conversation("keep me")
        again = ChatStore(self.store.path)
        self.assertEqual(again.get_conversation(conversation).title, "keep me")

    def test_round_trip(self):
        conversation = self.store.create_conversation("Maths", model_key="fast")
        self.store.add_message(conversation, "user", "sqrt(144)?")
        self.store.add_message(
            conversation,
            "assistant",
            "12",
            tools=[{"name": "calculate", "ok": True, "summary": "result=12"}],
            elapsed=4.2,
            model_key="fast",
        )

        messages = self.store.get_messages(conversation)
        self.assertEqual([m.role for m in messages], ["user", "assistant"])
        self.assertEqual(messages[1].content, "12")
        self.assertEqual(messages[1].tools[0]["name"], "calculate")
        self.assertEqual(messages[1].elapsed, 4.2)

    def test_messages_keep_insertion_order(self):
        conversation = self.store.create_conversation()
        for index in range(6):
            self.store.add_message(conversation, "user", f"m{index}")
        contents = [m.content for m in self.store.get_messages(conversation)]
        self.assertEqual(contents, [f"m{i}" for i in range(6)])

    def test_ui_dict_shape(self):
        conversation = self.store.create_conversation()
        self.store.add_message(conversation, "assistant", "hi", elapsed=1.5)
        entry = self.store.get_messages(conversation)[0].as_ui_dict()
        self.assertEqual(entry["role"], "assistant")
        self.assertEqual(entry["elapsed"], 1.5)
        self.assertNotIn("tools", entry)  # omitted when empty

    def test_message_count(self):
        conversation = self.store.create_conversation()
        self.store.add_message(conversation, "user", "one")
        self.store.add_message(conversation, "assistant", "two")
        self.assertEqual(self.store.message_count(conversation), 2)

    def test_listing_is_newest_first(self):
        first = self.store.create_conversation("first")
        second = self.store.create_conversation("second")
        # Touching the older one should float it back to the top.
        self.store.add_message(first, "user", "later activity")

        titles = [c.title for c in self.store.list_conversations()]
        self.assertEqual(titles[0], "first")
        self.assertIn("second", titles)
        self.assertEqual(len(titles), 2)
        self.assertNotEqual(first, second)

    def test_listing_reports_message_counts(self):
        conversation = self.store.create_conversation("counted")
        self.store.add_message(conversation, "user", "a")
        self.store.add_message(conversation, "assistant", "b")
        entry = self.store.list_conversations()[0]
        self.assertEqual(entry.message_count, 2)

    def test_listing_respects_the_limit(self):
        for index in range(5):
            self.store.create_conversation(f"c{index}")
        self.assertEqual(len(self.store.list_conversations(limit=3)), 3)

    def test_rename(self):
        conversation = self.store.create_conversation("old")
        self.store.rename_conversation(conversation, "new")
        self.assertEqual(self.store.get_conversation(conversation).title, "new")

    def test_delete_removes_messages_too(self):
        conversation = self.store.create_conversation()
        self.store.add_message(conversation, "user", "gone soon")
        self.store.delete_conversation(conversation)

        self.assertIsNone(self.store.get_conversation(conversation))
        self.assertEqual(self.store.get_messages(conversation), [])

    def test_delete_leaves_other_conversations(self):
        keep = self.store.create_conversation("keep")
        drop = self.store.create_conversation("drop")
        self.store.add_message(keep, "user", "still here")
        self.store.delete_conversation(drop)

        self.assertEqual(len(self.store.list_conversations()), 1)
        self.assertEqual(len(self.store.get_messages(keep)), 1)

    def test_missing_conversation_returns_none(self):
        self.assertIsNone(self.store.get_conversation(999))

    def test_purge(self):
        conversation = self.store.create_conversation()
        self.store.add_message(conversation, "user", "x")
        self.store.purge()
        self.assertEqual(self.store.list_conversations(), [])

    def test_corrupt_tool_json_does_not_break_reads(self):
        conversation = self.store.create_conversation()
        self.store.add_message(conversation, "assistant", "hi")
        with self.store._connect() as connection:
            connection.execute("UPDATE messages SET tools = ?", ("{not json",))

        self.assertEqual(self.store.get_messages(conversation)[0].tools, [])


class SearchTests(unittest.TestCase):
    """Searching conversations by title and by what was said in them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(Path(self._tmp.name) / "history.db")

    def tearDown(self):
        self._tmp.cleanup()

    def conversation(self, title: str, *messages: str) -> int:
        conversation = self.store.create_conversation(title)
        for index, text in enumerate(messages):
            self.store.add_message(
                conversation, "user" if index % 2 == 0 else "assistant", text
            )
        return conversation

    def titles(self, needle: str) -> list[str]:
        return [hit.title for hit in self.store.search(needle)]

    def test_a_message_body_is_searched_not_only_the_title(self):
        """The whole point: the title said nothing about what was discussed."""
        self.conversation("Untitled", "the mitochondrion is the powerhouse")

        self.assertEqual(self.titles("mitochondrion"), ["Untitled"])

    def test_a_title_still_matches(self):
        """What the pane did before search reached message bodies."""
        self.conversation("About photosynthesis", "nothing relevant here")

        hits = self.store.search("photosynthesis")

        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0].title_matched)

    def test_a_partial_word_matches(self):
        """The box is typed into a few characters at a time."""
        self.conversation("x", "what is the integral of ln x")

        self.assertEqual(len(self.store.search("integ")), 1)

    def test_matching_is_case_insensitive(self):
        self.conversation("x", "The Integral Of Ln X")

        self.assertEqual(len(self.store.search("integral")), 1)

    def test_the_number_of_matching_messages_is_counted(self):
        self.conversation("x", "alpha here", "no", "alpha again", "alpha third")

        self.assertEqual(self.store.search("alpha")[0].matches, 3)

    def test_a_conversation_matching_nothing_is_absent(self):
        self.conversation("Weather", "it is raining")

        self.assertEqual(self.store.search("mitochondrion"), [])

    def test_an_empty_query_finds_nothing_rather_than_everything(self):
        """A cleared search box must not return the whole history."""
        self.conversation("Weather", "it is raining")

        self.assertEqual(self.store.search(""), [])
        self.assertEqual(self.store.search("   "), [])

    # --- the wildcards, which is where a LIKE search goes wrong ---

    def test_an_underscore_is_a_character_not_a_wildcard(self):
        """Unescaped, `_` matches any single character - so, everything."""
        self.conversation("has one", "the load_skill tool")
        self.conversation("has none", "nothing relevant")

        self.assertEqual(self.titles("_"), ["has one"])

    def test_a_percent_is_a_character_not_a_wildcard(self):
        # The decoy has to contain "100" followed by something else, or an
        # unescaped `%` would fail to match it anyway and the test would pass
        # with the escaping removed - which it did, the first time round.
        self.conversation("has one", "it was 100% correct")
        self.conversation("has none", "there were 100 apples")

        self.assertEqual(self.titles("100%"), ["has one"])

    def test_a_backslash_is_a_character(self):
        """It is the escape character, so it has to escape itself."""
        self.conversation("windows", r"the path is C:\Users")

        self.assertEqual(self.titles("C:\\Users"), ["windows"])

    # --- the snippet ---

    def test_the_snippet_splits_around_the_match(self):
        self.conversation("x", "before the needle after")

        hit = self.store.search("needle")[0]

        self.assertEqual(hit.match, "needle")
        self.assertTrue(hit.before.endswith("before the "))
        self.assertTrue(hit.after.startswith(" after"))

    def test_the_snippet_keeps_the_stored_casing(self):
        """It shows what is in the conversation, not what was typed."""
        self.conversation("x", "The Needle here")

        self.assertEqual(self.store.search("needle")[0].match, "Needle")

    def test_a_long_message_is_trimmed_on_both_sides(self):
        self.conversation("x", "a" * 500 + " needle " + "b" * 500)

        hit = self.store.search("needle")[0]

        self.assertTrue(hit.before.startswith("…"))
        self.assertTrue(hit.after.endswith("…"))
        self.assertLess(len(hit.before), 200)
        self.assertLess(len(hit.after), 200)

    def test_newlines_do_not_become_a_paragraph_of_whitespace(self):
        """A snippet is one line in a list, and code blocks are full of them."""
        self.conversation("x", "start\n\n\n    indented needle\n\n\nend")

        hit = self.store.search("needle")[0]

        self.assertNotIn("\n", hit.before + hit.match + hit.after)

    def test_the_first_matching_message_is_the_one_shown(self):
        """Usually the one that started the thread being looked for."""
        conversation = self.conversation("x", "no", "first needle", "second needle")
        first = self.store.get_messages(conversation)[1]

        self.assertEqual(self.store.search("needle")[0].message_id, first.id)

    def test_a_title_only_match_has_no_message(self):
        self.conversation("about needles", "nothing relevant")

        hit = self.store.search("needle")[0]

        self.assertIsNone(hit.message_id)
        self.assertTrue(hit.title_matched)

    # --- ordering and limits ---

    def test_the_most_recently_updated_comes_first(self):
        self.conversation("older", "needle")
        self.conversation("newer", "needle")

        self.assertEqual(self.titles("needle"), ["newer", "older"])

    def test_the_limit_is_honoured(self):
        for index in range(5):
            self.conversation(f"c{index}", "needle")

        self.assertEqual(len(self.store.search("needle", limit=2)), 2)


if __name__ == "__main__":
    unittest.main()
