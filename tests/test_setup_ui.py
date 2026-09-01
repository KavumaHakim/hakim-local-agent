"""The setup walkthrough's terminal toolkit.

Two things are worth testing here and they pull in opposite directions: that
the prompts do what a person expects when there is a person, and that none of
them block, redraw or ask anything when there is not. A setup script that
hangs on a question nobody can see is the worst failure mode available to it,
and it is the one that only shows up in CI.
"""

from __future__ import annotations

import io
import re
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


class RedrawTests(unittest.TestCase):
    """The menu replaces itself rather than printing again underneath."""

    def setUp(self):
        patch = mock.patch.object(ui, "interactive", return_value=True)
        patch.start()
        self.addCleanup(patch.stop)

    def menu(self, replies, *, fancy):
        items = [
            {"label": "one", "note": "first", "on": True},
            {"label": "two", "note": "second", "on": False},
        ]
        with mock.patch.object(ui, "fancy", return_value=fancy), mock.patch.object(
            ui, "_read", side_effect=list(replies)
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                ui.toggle("pick", items)
        return output.getvalue(), items

    def test_it_moves_the_cursor_up_instead_of_scrolling(self):
        text, _ = self.menu(["1", ""], fancy=True)
        moves = re.findall(r"\033\[(\d+)A\033\[J", text)
        # One redraw for the toggle, one to clear the menu on accept.
        self.assertEqual(len(moves), 2)

    def test_it_moves_up_exactly_what_it_drew(self):
        """Off by one and the menu eats the line above it, or leaves a
        duplicate behind - both of which look like a broken terminal."""
        text, _ = self.menu(["1", ""], fancy=True)
        moves = [int(value) for value in re.findall(r"\033\[(\d+)A", text)]

        first_frame = re.split(r"\033\[\d+A", text)[0]
        drawn = first_frame.count("\n")
        # The prompt line has no newline of its own; the user's Enter ends it.
        self.assertEqual(moves[0], drawn + 1)

    def test_a_plain_terminal_never_moves_the_cursor(self):
        """Where the cursor cannot be moved, printing again is correct - and
        emitting the escape codes anyway would litter the log with them."""
        text, _ = self.menu(["1", ""], fancy=False)
        self.assertNotIn("\033[", text)

    def test_the_menu_is_cleared_once_accepted(self):
        """It has served its purpose; leaving it on screen above the install
        log is clutter."""
        text, _ = self.menu([""], fancy=True)
        self.assertRegex(text, r"\033\[\d+A\033\[J$")

    def test_a_bad_answer_is_explained_in_the_redraw(self):
        text, _ = self.menu(["nope", ""], fancy=True)
        self.assertIn("pick 1 to 2", text)


class BannerTests(unittest.TestCase):
    def test_it_falls_back_to_plain_text_on_a_narrow_terminal(self):
        with mock.patch.object(ui, "width", return_value=30):
            output = io.StringIO()
            with redirect_stdout(output):
                ui.banner("HAKIM", "subtitle")
        text = output.getvalue()
        self.assertIn("HAKIM", text)
        self.assertNotIn("###", text)

    def test_it_draws_letters_when_there_is_room(self):
        with mock.patch.object(ui, "width", return_value=100):
            output = io.StringIO()
            with redirect_stdout(output):
                ui.banner("HAKIM")
        text = output.getvalue()
        self.assertEqual(len(text.strip().splitlines()), 5)

    def test_it_uses_ascii_when_the_console_cannot_encode_blocks(self):
        """cmd.exe with a legacy code page cannot print a block character, and
        a UnicodeEncodeError would be an absurd way for setup to die."""
        with mock.patch.object(ui, "width", return_value=100), mock.patch.object(
            ui, "_can_encode", return_value=False
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                ui.banner("HAKIM")
        self.assertIn("#", output.getvalue())
        self.assertNotIn("\u2588", output.getvalue())


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
        text = output.getvalue()
        self.assertIn("2/2", text)
        self.assertIn("Two", text)

    def test_the_progress_bar_fills_as_the_steps_pass(self):
        """The header carries a bar so the shape of the run is visible
        without counting the numbers."""
        steps = ui.Steps(["a", "b", "c", "d"])
        output = io.StringIO()
        with redirect_stdout(output):
            steps.start(1)
        plain = re.sub(r"\[[0-9;]*m", "", output.getvalue())
        self.assertIn("==--", plain)


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
