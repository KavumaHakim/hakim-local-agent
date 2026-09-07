"""Task router tests."""

from __future__ import annotations

import unittest

from agent.router import TaskRouter


def router(enabled=True):
    return TaskRouter(["fast", "strong"], enabled=enabled)


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
        decision = router().choose("hi", current_key="fast", reached="strong")
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
        strict = TaskRouter(["fast", "strong"], enabled=True, threshold=99)
        self.assertEqual(strict.choose("debug and refactor everything").key, "fast")


class ChainTests(unittest.TestCase):
    """More than two models, ordered cheapest first."""

    def chain(self, *keys, **kwargs):
        return TaskRouter(list(keys), enabled=True, **kwargs)

    def test_a_simple_prompt_takes_the_first(self):
        router = self.chain("small", "middle", "big")
        self.assertEqual(router.choose("hello there").key, "small")

    def test_a_hard_prompt_climbs_past_the_first(self):
        router = self.chain("small", "middle", "big")
        # One threshold of score: a single hard signal is worth 2, plus the
        # length of this prompt.
        decision = router.choose("please review this and explain why it stalls")
        self.assertNotEqual(decision.key, "small")

    def test_the_hardest_prompts_reach_the_end(self):
        router = self.chain("small", "middle", "big")
        hard = (
            "Debug and refactor this, trace the root cause and analyse the "
            "complexity:\n```\n" + "x = 1\n" * 40 + "```\n" + "why does it "
            "stall? why is it slow? " + "detail " * 200
        )
        self.assertEqual(router.choose(hard).key, "big")

    def test_it_never_goes_past_the_end(self):
        """A very high score must cap, not index off the end."""
        router = self.chain("small", "big")
        hard = "debug refactor analyse ```code``` " + "word " * 400
        self.assertEqual(router.choose(hard).key, "big")

    def test_a_chain_of_one_never_switches(self):
        """Which is the useful way to say "leave the model alone"."""
        router = self.chain("only")
        self.assertEqual(router.choose("hello").key, "only")
        self.assertEqual(
            router.choose("debug and refactor everything, and explain why").key,
            "only",
        )

    # --- the floor: it never routes down ---

    def test_it_does_not_drop_back_down_the_chain(self):
        router = self.chain("small", "middle", "big")
        decision = router.choose("hi", current_key="big")
        self.assertEqual(decision.key, "big")

    def test_a_model_already_reached_sets_the_floor(self):
        """The conversation used it earlier, so the RAM is already spent."""
        router = self.chain("small", "middle", "big")
        decision = router.choose("hi", current_key="small", reached="middle")
        self.assertEqual(decision.key, "middle")

    def test_it_may_still_climb_above_the_floor(self):
        router = self.chain("small", "middle", "big")
        hard = (
            "Debug and refactor this, trace the root cause and analyse the "
            "complexity:\n```\n" + "x = 1\n" * 40 + "```\n" + "why does it "
            "stall? why is it slow? " + "detail " * 200
        )
        self.assertEqual(router.choose(hard, current_key="middle").key, "big")

    def test_a_model_outside_the_chain_is_not_a_floor(self):
        """Picking one by hand must not let the router demote the next turn.

        Position -1 rather than 0: treating an unknown key as the bottom would
        make a hand-picked model behave differently from no choice at all.
        """
        router = self.chain("small", "big")
        self.assertEqual(router.position("something-else"), -1)
        self.assertEqual(router.choose("hello", current_key="something-else").key, "small")

    # --- the chain itself ---

    def test_blanks_are_dropped(self):
        self.assertEqual(self.chain("small", "", "big").chain, ["small", "big"])

    def test_repeats_are_dropped_keeping_the_first_position(self):
        """A repeat would make one step of escalation change nothing."""
        router = self.chain("small", "big", "small", "big")
        self.assertEqual(router.chain, ["small", "big"])

    def test_first_and_last_name_the_ends(self):
        router = self.chain("small", "middle", "big")
        self.assertEqual(router.first, "small")
        self.assertEqual(router.last, "big")

    def test_an_empty_chain_keeps_whatever_is_current(self):
        router = TaskRouter([], enabled=True)
        self.assertEqual(router.choose("hi", current_key="whatever").key, "whatever")
        self.assertEqual(router.first, "")
        self.assertEqual(router.last, "")

    def test_two_entries_behave_exactly_as_the_old_pair_did(self):
        """The old rule was: under the threshold the first, at or over it the
        second. That is this generalisation at length two, and it is the
        reason none of the tests above this class had to change."""
        router = self.chain("fast", "strong")
        for prompt in ("hello there", "what is 2 + 2?", ""):
            self.assertEqual(router.choose(prompt).key, "fast", prompt)
        involved = "Debug why the loop stalls and trace the root cause."
        self.assertEqual(router.choose(involved).key, "strong")


if __name__ == "__main__":
    unittest.main()
