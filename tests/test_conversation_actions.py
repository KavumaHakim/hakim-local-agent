"""Naming a conversation, forking one, and deleting one message of it.

The three things a transcript needed that it did not have. Deleting from a
point down already existed - that is the rewind behind editing a question -
but removing one wrong answer, keeping the first branch while trying a
second, and having a name that is a name rather than a truncated question
all did not.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chat_store import ChatStore, clean_title


class CleanTitleTests(unittest.TestCase):
    """What small models actually answer when asked for a title."""

    def test_a_plain_title_is_kept(self):
        self.assertEqual(
            clean_title("Potassium Chromate Solubility"),
            "Potassium Chromate Solubility",
        )

    def test_quotes_come_off(self):
        self.assertEqual(clean_title('"Lead Hydroxide Reaction"'), "Lead Hydroxide Reaction")
        self.assertEqual(clean_title("“Smart quotes”"), "Smart quotes")

    def test_a_label_comes_off(self):
        for raw in (
            "Title: Solubility Rules",
            "**Title:** Solubility Rules",
            "Here is a title: Solubility Rules",
            "Sure: Solubility Rules",
        ):
            self.assertEqual(clean_title(raw), "Solubility Rules", raw)

    def test_emphasis_is_stripped_before_bullets(self):
        """Doing it the other way round leaves `*Title:` behind."""
        self.assertEqual(clean_title('**Title:** "Group 2 Reactivity"'), "Group 2 Reactivity")

    def test_only_the_first_of_several_candidates(self):
        self.assertEqual(
            clean_title("1. Chromate Equilibrium\n2. Something Else"),
            "Chromate Equilibrium",
        )
        self.assertEqual(clean_title("- Integral of x squared"), "Integral of x squared")

    def test_a_trailing_full_stop_goes(self):
        self.assertEqual(clean_title("Chemistry."), "Chemistry")

    def test_a_paragraph_is_not_a_title(self):
        """Better to keep the plain name than to show half a sentence."""
        rambling = (
            "This is a long explanation of what the conversation was about, "
            "which is not a title at all"
        )
        self.assertEqual(clean_title(rambling), "")

    def test_nothing_useful_gives_nothing(self):
        for raw in ("", "   ", "\n\n"):
            self.assertEqual(clean_title(raw), "")


class TitledFlagTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ChatStore(Path(self._tmp.name) / "h.db")

    def test_a_new_conversation_is_not_titled(self):
        cid = self.store.create_conversation("New conversation")
        self.assertFalse(self.store.get_conversation(cid).titled)

    def test_the_placeholder_name_does_not_count_as_titled(self):
        """It is the question with its end cut off, which the namer replaces."""
        cid = self.store.create_conversation("x")
        self.store.rename_conversation(cid, "What is the integral of", titled=False)
        self.assertFalse(self.store.get_conversation(cid).titled)

    def test_renaming_by_hand_counts(self):
        """The default is True, so a person's name is never overwritten."""
        cid = self.store.create_conversation("x")
        self.store.rename_conversation(cid, "My Own Name")
        self.assertTrue(self.store.get_conversation(cid).titled)

    def test_the_flag_survives_a_reopen(self):
        path = Path(self._tmp.name) / "h.db"
        cid = self.store.create_conversation("x")
        self.store.rename_conversation(cid, "Kept")
        self.assertTrue(ChatStore(path).get_conversation(cid).titled)


class MigrationTests(unittest.TestCase):
    def test_a_database_without_the_column_gains_it(self):
        """The real database has conversations in it; recreating is not an option."""
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.db"
            # A conversations table from before `titled` existed.
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL, model_key TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO conversations (title, created_at, updated_at)
                VALUES ('Existing talk', 'then', 'then');
                """
            )
            connection.commit()
            connection.close()

            store = ChatStore(path)

            conversation = store.get_conversation(1)
            self.assertEqual(conversation.title, "Existing talk")
            self.assertFalse(conversation.titled)

    def test_opening_twice_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.db"
            ChatStore(path)
            ChatStore(path)
            ChatStore(path)  # ALTER TABLE would raise on a second run


class ForkTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ChatStore(Path(self._tmp.name) / "h.db")
        self.cid = self.store.create_conversation("Original")
        self.ids = [
            self.store.add_message(self.cid, "user", "one"),
            self.store.add_message(self.cid, "assistant", "two"),
            self.store.add_message(self.cid, "user", "three"),
            self.store.add_message(self.cid, "assistant", "four"),
        ]

    def texts(self, conversation_id: int) -> list[str]:
        return [m.content for m in self.store.get_messages(conversation_id)]

    def test_it_copies_up_to_and_including_the_message(self):
        fork = self.store.fork_conversation(self.cid, self.ids[1])
        self.assertEqual(self.texts(fork), ["one", "two"])

    def test_the_original_is_untouched(self):
        self.store.fork_conversation(self.cid, self.ids[1])
        self.assertEqual(self.texts(self.cid), ["one", "two", "three", "four"])

    def test_the_two_are_independent_afterwards(self):
        fork = self.store.fork_conversation(self.cid, self.ids[1])
        self.store.add_message(fork, "user", "a different direction")
        self.assertEqual(self.texts(self.cid), ["one", "two", "three", "four"])

    def test_the_copy_is_not_titled(self):
        """It is about to diverge, so it deserves its own name."""
        fork = self.store.fork_conversation(self.cid, self.ids[1])
        self.assertFalse(self.store.get_conversation(fork).titled)

    def test_forking_at_the_last_message_copies_everything(self):
        fork = self.store.fork_conversation(self.cid, self.ids[-1])
        self.assertEqual(self.texts(fork), ["one", "two", "three", "four"])

    def test_an_unknown_conversation_or_message_gives_nothing(self):
        self.assertIsNone(self.store.fork_conversation(9999, self.ids[0]))
        self.assertIsNone(self.store.fork_conversation(self.cid, 9999))


class DeleteMessageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ChatStore(Path(self._tmp.name) / "h.db")
        self.cid = self.store.create_conversation("c")
        self.ids = [
            self.store.add_message(self.cid, "user", "one"),
            self.store.add_message(self.cid, "assistant", "two"),
            self.store.add_message(self.cid, "user", "three"),
        ]

    def test_it_removes_only_that_message(self):
        self.assertTrue(self.store.delete_message(self.cid, self.ids[1]))
        self.assertEqual(
            [m.content for m in self.store.get_messages(self.cid)], ["one", "three"]
        )

    def test_it_differs_from_rewinding(self):
        """Rewinding takes everything after it as well; this does not."""
        self.store.delete_message(self.cid, self.ids[1])
        self.assertEqual(self.store.message_count(self.cid), 2)

    def test_an_unknown_message_is_reported_not_raised(self):
        self.assertFalse(self.store.delete_message(self.cid, 9999))

    def test_a_message_from_another_conversation_is_not_touched(self):
        other = self.store.create_conversation("other")
        theirs = self.store.add_message(other, "user", "theirs")

        self.assertFalse(self.store.delete_message(self.cid, theirs))
        self.assertEqual(self.store.message_count(other), 1)


if __name__ == "__main__":
    unittest.main()
