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


if __name__ == "__main__":
    unittest.main()
