"""The setup walkthrough's terminal toolkit.

Two things are worth testing here and they pull in opposite directions: that
the prompts do what a person expects when there is a person, and that none of
them block, redraw or ask anything when there is not. A setup script that
hangs on a question nobody can see is the worst failure mode available to it,
and it is the one that only shows up in CI.
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ui  # noqa: E402


class NonInteractiveTests(unittest.TestCase):
    """With no terminal, every prompt answers itself and moves on."""

    def setUp(self):
        patch = mock.patch.object(ui, "interactive", return_value=False)
        patch.start()
        self.addCleanup(patch.stop)

    def test_confirm_takes_its_default_without_asking(self):
        with mock.patch.object(ui, "_read", side_effect=AssertionError("asked!")):
            self.assertTrue(ui.confirm("go?", default=True))
            self.assertFalse(ui.confirm("go?", default=False))

    def test_choose_takes_its_default_without_asking(self):
        options = [("first", ""), ("second", "")]
        with mock.patch.object(ui, "_read", side_effect=AssertionError("asked!")):
            self.assertEqual(ui.choose("pick", options, default=1), 1)

    def test_toggle_returns_the_items_untouched(self):
        items = [{"label": "a", "on": True}, {"label": "b", "on": False}]
        with mock.patch.object(ui, "_read", side_effect=AssertionError("asked!")):
            result = ui.toggle("choose", items)
        self.assertEqual([item["on"] for item in result], [True, False])

    def test_a_flag_overrides_the_prompt_entirely(self):
        """--yes and friends answer before the question is reached."""
        self.assertTrue(ui.confirm("go?", default=False, assume=True))
        self.assertFalse(ui.confirm("go?", default=True, assume=False))


class InteractiveTests(unittest.TestCase):
    """With a terminal, the answers come from what was typed."""

    def setUp(self):
        patch = mock.patch.object(ui, "interactive", return_value=True)
        patch.start()
        self.addCleanup(patch.stop)

    def answer(self, *replies: str):
        return mock.patch.object(ui, "_read", side_effect=list(replies))

    def test_confirm_reads_yes_and_no(self):
        for reply, expected in (("y", True), ("yes", True), ("n", False), ("no", False)):
            with self.answer(reply):
                self.assertIs(ui.confirm("go?"), expected)

    def test_empty_input_means_the_default(self):
        with self.answer(""):
            self.assertTrue(ui.confirm("go?", default=True))
        with self.answer(""):
            self.assertFalse(ui.confirm("go?", default=False))

    def test_confirm_asks_again_after_nonsense(self):
        with self.answer("maybe", "y"):
            with redirect_stdout(io.StringIO()):
                self.assertTrue(ui.confirm("go?"))

    def test_choose_returns_the_numbered_option(self):
        options = [("first", ""), ("second", ""), ("third", "")]
        with self.answer("3"):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ui.choose("pick", options), 2)

    def test_choose_rejects_out_of_range(self):
        options = [("first", ""), ("second", "")]
        with self.answer("9", "0", "2"):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ui.choose("pick", options), 1)

    def test_toggle_flips_then_accepts(self):
        items = [
            {"label": "llama.cpp", "note": "", "on": True},
            {"label": "torch", "note": "", "on": False},
        ]
        # Turn the second on, turn the first off, then Enter.
        with self.answer("2", "1", ""):
            with redirect_stdout(io.StringIO()):
                result = ui.toggle("Optional pieces", items)
        self.assertEqual([item["on"] for item in result], [False, True])

    def test_toggle_accepts_immediately_on_enter(self):
        items = [{"label": "a", "note": "", "on": True}]
        with self.answer(""):
            with redirect_stdout(io.StringIO()):
                result = ui.toggle("choose", items)
        self.assertEqual([item["on"] for item in result], [True])


class ProgressTests(unittest.TestCase):
    """The bar, and what it does when nothing can be redrawn."""

    def test_a_pipe_gets_lines_not_carriage_returns(self):
        """Redrawing into a CI log produces thousands of unreadable lines."""
        with mock.patch.object(ui, "fancy", return_value=False):
            bar = ui.Progress("downloading", 1_000_000)
            output = io.StringIO()
            with redirect_stdout(output):
                for _ in range(10):
                    bar.advance(100_000)
            text = output.getvalue()

        self.assertNotIn("\r", text)
        # One line per quarter, not one per chunk - and not two, which is
        # what happens if the terminal's time throttle is applied here too.
        lines = text.strip().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertIn("100%", text)

    def test_the_byte_count_stops_at_the_total(self):
        """"19.2/18.4 MB" reads as a bug in the thing doing the counting."""
        with mock.patch.object(ui, "fancy", return_value=False):
            bar = ui.Progress("downloading", 1_000_000)
            output = io.StringIO()
            with redirect_stdout(output):
                bar.advance(1_500_000)
        self.assertIn("1.0/1.0 MB", output.getvalue())

    def test_the_bar_never_exceeds_full(self):
        """A server that sends more than it advertised must not draw past the
        end of the bar or report 130%."""
        with mock.patch.object(ui, "fancy", return_value=True):
            bar = ui.Progress("downloading", 100)
            output = io.StringIO()
            with redirect_stdout(output):
                bar.advance(500)
            text = output.getvalue()
        self.assertIn("100%", text)
        self.assertNotIn("500%", text)

    def test_a_zero_length_download_does_not_divide_by_zero(self):
        with mock.patch.object(ui, "fancy", return_value=True):
            bar = ui.Progress("downloading", 0)
            with redirect_stdout(io.StringIO()):
                bar.advance(0)
                bar.done()

    def test_done_is_safe_without_a_terminal(self):
        with mock.patch.object(ui, "fancy", return_value=False):
            bar = ui.Progress("downloading", 10)
            with redirect_stdout(io.StringIO()):
                bar.done()


class SpinnerTests(unittest.TestCase):
    def test_it_prints_one_plain_line_without_a_terminal(self):
        with mock.patch.object(ui, "fancy", return_value=False):
            output = io.StringIO()
            with redirect_stdout(output):
                with ui.Spinner("installing"):
                    pass
            text = output.getvalue()
        self.assertIn("installing", text)
        self.assertNotIn("\r", text)

    def test_it_reports_how_long_it_took(self):
        with mock.patch.object(ui, "fancy", return_value=False):
            with redirect_stdout(io.StringIO()):
                with ui.Spinner("working") as spinner:
                    pass
                self.assertGreaterEqual(spinner.elapsed, 0.0)


class StepsTests(unittest.TestCase):
    def test_the_plan_is_printed_before_anything_happens(self):
        steps = ui.Steps(["One", "Two", "Three"])
        output = io.StringIO()
        with redirect_stdout(output):
            steps.show()
        text = output.getvalue()
        for name in ("One", "Two", "Three"):
            self.assertIn(name, text)

    def test_each_step_says_where_it_is_in_the_whole(self):
        steps = ui.Steps(["One", "Two"])
        output = io.StringIO()
        with redirect_stdout(output):
            steps.start(1)
        self.assertIn("[2/2]", output.getvalue())


class ColourTests(unittest.TestCase):
    def test_colour_codes_are_empty_when_colour_is_off(self):
        """Everything concatenates these unconditionally, so when colour is
        off they must be empty strings rather than absent."""
        if not ui.COLOUR:
            for code in (ui.DIM, ui.BOLD, ui.RESET, ui.GREEN, ui.RED):
                self.assertEqual(code, "")

    def test_no_color_is_honoured(self):
        with mock.patch.dict("os.environ", {"NO_COLOR": "1"}):
            self.assertFalse(ui._colour_ok())

    def test_a_dumb_terminal_gets_no_redrawing(self):
        with mock.patch.dict("os.environ", {"TERM": "dumb"}):
            self.assertFalse(ui.fancy())


if __name__ == "__main__":
    unittest.main()
