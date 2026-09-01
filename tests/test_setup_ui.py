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


class ServerPathTests(unittest.TestCase):
    """Remembering where somebody's llama-server actually is."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    def a_server_at(self, *parts: str) -> Path:
        path = self.tmp.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("binary", encoding="utf-8")
        return path

    def registry_pointing_nowhere(self) -> Path:
        import json

        source = json.loads(
            (Path(__file__).resolve().parent.parent / "models.json").read_text(
                encoding="utf-8"
            )
        )
        source["server_exe"] = "../nowhere/llama-server.exe"
        path = self.tmp / "models.json"
        path.write_text(json.dumps(source), encoding="utf-8")
        return path

    def test_a_remembered_path_beats_the_search(self):
        """Someone who has said where theirs is should not be second-guessed
        by a stale binary on PATH."""
        from models.manager import load_registry
        from models.preferences import ModelPreferences

        server = self.a_server_at("odd", "place", "llama-server.exe")
        ModelPreferences.load(self.tmp).set_server_exe(str(server))

        registry = load_registry(
            self.registry_pointing_nowhere(), preferences_dir=self.tmp
        )
        self.assertEqual(Path(registry["server_exe"]), server)

    def test_a_remembered_path_that_has_gone_is_ignored(self):
        """Deleting the binary should fall back to the search, not fail."""
        from models.manager import load_registry
        from models.preferences import ModelPreferences

        server = self.a_server_at("odd", "llama-server.exe")
        ModelPreferences.load(self.tmp).set_server_exe(str(server))
        server.unlink()

        registry = load_registry(
            self.registry_pointing_nowhere(), preferences_dir=self.tmp
        )
        self.assertNotEqual(Path(registry["server_exe"]), server)

    def test_it_survives_being_written_and_read_back(self):
        from models.preferences import ModelPreferences

        ModelPreferences.load(self.tmp).set_server_exe(r"C:\tools\llama-server.exe")
        again = ModelPreferences.load(self.tmp)
        self.assertEqual(again.server_exe, r"C:\tools\llama-server.exe")

    def test_it_is_kept_out_of_version_control(self):
        """models.json is committed, so a path from one laptop would be wrong
        on every other machine."""
        from models.preferences import ModelPreferences

        ModelPreferences.load(self.tmp).set_server_exe("/opt/llama-server")
        self.assertTrue((self.tmp / "models.local.json").is_file())
        registry = Path(__file__).resolve().parent.parent / "models.json"
        self.assertNotIn("/opt/llama-server", registry.read_text(encoding="utf-8"))


class ApiKeyTests(unittest.TestCase):
    """Saving hosted-model keys into .env, without ever printing one."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.tmp / ".env").write_text(
            "# comment\n# GEMINI_API_KEY=\n# CEREBRAS_API_KEY=\n", encoding="utf-8"
        )

        import setup as setup_script

        self.setup = setup_script
        patch = mock.patch.object(setup_script, "ROOT", self.tmp)
        patch.start()
        self.addCleanup(patch.stop)

    def env(self) -> str:
        return (self.tmp / ".env").read_text(encoding="utf-8")

    def test_a_key_replaces_its_commented_placeholder(self):
        """Appending instead would leave the example line above the real one,
        which reads as though the key is still unset."""
        self.setup.write_key("GEMINI_API_KEY", "abc123")
        self.assertIn("GEMINI_API_KEY=abc123", self.env())
        self.assertNotIn("# GEMINI_API_KEY=", self.env())

    def test_writing_twice_does_not_duplicate_the_line(self):
        self.setup.write_key("GEMINI_API_KEY", "first")
        self.setup.write_key("GEMINI_API_KEY", "second")
        lines = [
            line
            for line in self.env().splitlines()
            if line.startswith("GEMINI_API_KEY=")
        ]
        self.assertEqual(lines, ["GEMINI_API_KEY=second"])

    def test_a_key_with_no_placeholder_is_appended(self):
        self.setup.write_key("SOMETHING_NEW", "value")
        self.assertIn("SOMETHING_NEW=value", self.env())

    def test_an_answered_key_is_not_asked_for_again(self):
        self.assertNotIn("GEMINI_API_KEY", self.setup.existing_keys())
        self.setup.write_key("GEMINI_API_KEY", "abc")
        self.assertIn("GEMINI_API_KEY", self.setup.existing_keys())

    def test_a_commented_placeholder_does_not_count_as_answered(self):
        """The example file ships with all of them commented out."""
        self.assertNotIn("CEREBRAS_API_KEY", self.setup.existing_keys())

    def test_nothing_is_asked_without_a_terminal(self):
        with mock.patch.object(ui, "interactive", return_value=False):
            with mock.patch.object(
                ui, "ask_secret", side_effect=AssertionError("asked!")
            ):
                self.setup.ask_for_keys(ask=True)

    def test_the_providers_come_from_the_registry(self):
        """Adding one to models.json should be enough for setup to ask."""
        with mock.patch.object(
            self.setup, "ROOT", Path(__file__).resolve().parent.parent
        ):
            variables = [name for _, name in self.setup.hosted_providers()]
        self.assertIn("GEMINI_API_KEY", variables)
        self.assertIn("CEREBRAS_API_KEY", variables)


if __name__ == "__main__":
    unittest.main()
