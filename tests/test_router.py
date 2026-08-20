"""Task router tests."""

from __future__ import annotations

import unittest

from agent.router import TaskRouter


def router(enabled=True):
    return TaskRouter("fast", "strong", enabled=enabled)


class DisabledTests(unittest.TestCase):
    def test_disabled_keeps_current(self):
        decision = router(enabled=False).choose("debug this", current_key="fast")
        self.assertEqual(decision.key, "fast")
        self.assertIn("off", decision.reason)

    def test_disabled_falls_back_to_fast(self):
        self.assertEqual(router(enabled=False).choose("anything").key, "fast")


class SimplePromptTests(unittest.TestCase):
    def test_greeting(self):
        self.assertEqual(router().choose("hello there").key, "fast")

    def test_short_question(self):
        self.assertEqual(router().choose("what is the capital of Uganda?").key, "fast")

    def test_arithmetic(self):
        self.assertEqual(router().choose("what is 17 * 43 - 209?").key, "fast")

    def test_simple_file_request(self):
        self.assertEqual(
            router().choose("list the files in the workspace root").key, "fast"
        )

    def test_empty_prompt(self):
        self.assertEqual(router().choose("").key, "fast")


class HardPromptTests(unittest.TestCase):
    def test_debugging(self):
        decision = router().choose(
            "Debug why the agent loop stalls when a tool returns an error, "
            "and trace the root cause through the parser."
        )
        self.assertEqual(decision.key, "strong")
        self.assertIn("debug", decision.reason)

    def test_code_block(self):
        prompt = "Fix this:\n```python\ndef f():\n    return 1/0\n```"
        self.assertEqual(router().choose(prompt).key, "strong")

    def test_long_prompt(self):
        self.assertEqual(router().choose("word " * 200).key, "strong")

    def test_refactor_request(self):
        self.assertEqual(
            router().choose("Refactor the registry and review the design").key,
            "strong",
        )

    def test_several_files(self):
        decision = router().choose(
            "Compare config.py and manager.py and explain why they disagree "
            "about the default port and which one should change."
        )
        self.assertEqual(decision.key, "strong")


class NoDowngradeTests(unittest.TestCase):
    def test_stays_on_strong_once_escalated(self):
        decision = router().choose("hi", current_key="fast", escalated=True)
        self.assertEqual(decision.key, "strong")
        self.assertIn("staying", decision.reason)

    def test_stays_on_strong_when_already_there(self):
        self.assertEqual(router().choose("hi", current_key="strong").key, "strong")


class ScoringTests(unittest.TestCase):
    def test_word_boundary_avoids_false_positives(self):
        # 'plan' must not fire inside 'explanation'.
        score, _ = router().score("give a short explanation")
        self.assertLess(score, 3)

    def test_keyword_contribution_is_capped(self):
        many = "debug refactor optimise migrate benchmark profile review audit"
        score, _ = router().score(many)
        self.assertLessEqual(score, 6)

    def test_score_never_negative(self):
        score, _ = router().score("hi")
        self.assertGreaterEqual(score, 0)

    def test_reason_is_always_populated(self):
        _, reasons = router().score("hmm")
        self.assertTrue(reasons)

    def test_threshold_is_configurable(self):
        strict = TaskRouter("fast", "strong", enabled=True, threshold=99)
        self.assertEqual(strict.choose("debug and refactor everything").key, "fast")


if __name__ == "__main__":
    unittest.main()
