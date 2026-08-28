"""Calculator, filesystem, OCR and registry tests."""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from config import Config
from tools.base import ToolRegistry, ToolResult
from tools.calculator import CALCULATOR_TOOL, CalculationError, calculate, evaluate
from tools.filesystem import FilesystemToolError, WorkspaceFiles
from tools.ocr_tool import OcrClient, OcrError
from tools.registry import build_default_registry


class CalculatorTests(unittest.TestCase):
    def test_the_required_examples(self):
        self.assertEqual(evaluate("2 + 2"), 4)
        self.assertEqual(evaluate("sqrt(144)"), 12)
        self.assertEqual(evaluate("2**10"), 1024)
        self.assertAlmostEqual(evaluate("log(100)"), 4.605170185988092)
        self.assertAlmostEqual(evaluate("sin(pi / 2)"), 1.0)
        self.assertEqual(evaluate("17 * 43"), 731)

    def test_caret_is_a_power_operator(self):
        # What Ministral actually emits. Before this, it was rejected.
        self.assertEqual(evaluate("2^10"), 1024)
        self.assertEqual(evaluate("sqrt(144) + 25^2"), 637)

    def test_caret_keeps_power_precedence_not_xor_precedence(self):
        # The trap: Python's ^ binds looser than +, so handling this after
        # parsing would give (12 + 25)**2 = 1369. The rewrite happens before
        # the parse precisely so this reads as 12 + 625.
        self.assertEqual(evaluate("sqrt(144) + 25^2"), 637)
        self.assertNotEqual(evaluate("sqrt(144) + 25^2"), 1369)
        self.assertEqual(evaluate("2 + 3^2"), 11)
        self.assertEqual(evaluate("2 * 3^2"), 18)

    def test_caret_and_double_star_agree(self):
        for a, b in [("2^8", "2**8"), ("3^3 + 1", "3**3 + 1")]:
            self.assertEqual(evaluate(a), evaluate(b), msg=a)

    def test_caret_exponent_limit_still_applies(self):
        with self.assertRaises(CalculationError):
            evaluate("9^999999")

    def test_arithmetic(self):
        self.assertEqual(evaluate("2 + 3 * 4"), 14)
        self.assertEqual(evaluate("(2 + 3) * 4"), 20)
        self.assertEqual(evaluate("7 // 2"), 3)
        self.assertEqual(evaluate("7 % 2"), 1)
        self.assertEqual(evaluate("10 - 4"), 6)
        self.assertEqual(evaluate("9 / 2"), 4.5)

    def test_structured_success_payload(self):
        self.assertEqual(
            calculate("sqrt(144)"),
            {"success": True, "result": 12.0, "formatted": "12"},
        )

    def test_structured_failure_via_registry(self):
        registry = ToolRegistry([CALCULATOR_TOOL])
        result = registry.execute("calculate", {"expression": "import os"})
        self.assertFalse(result.ok)
        self.assertIn("Invalid expression", result.payload["error"])

    # --- the restrictions ---

    def test_imports_are_rejected(self):
        with self.assertRaises(CalculationError):
            evaluate("__import__('os').listdir('.')")

    def test_attribute_access_is_rejected(self):
        with self.assertRaises(CalculationError):
            evaluate("().__class__.__bases__[0]")

    def test_dangerous_names_are_rejected(self):
        for expression in ("os.getcwd()", "open('x')", "eval('1')", "exec('x=1')"):
            with self.assertRaises(CalculationError, msg=expression):
                evaluate(expression)

    def test_function_definitions_are_rejected(self):
        with self.assertRaises(CalculationError):
            evaluate("lambda: 1")

    def test_statements_are_rejected(self):
        with self.assertRaises(CalculationError):
            evaluate("x = 1")

    def test_strings_are_rejected(self):
        with self.assertRaises(CalculationError):
            evaluate("'abc' * 3")

    def test_huge_exponent_is_refused(self):
        with self.assertRaises(CalculationError):
            evaluate("9**999999")

    def test_division_by_zero(self):
        with self.assertRaises(CalculationError):
            evaluate("1 / 0")


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "notes.txt").write_text("hello workspace", encoding="utf-8")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "inner.txt").write_text("inner", encoding="utf-8")
        self.files = WorkspaceFiles(self.root, max_read_bytes=1000)

    def tearDown(self):
        self._tmp.cleanup()

    def test_list_directory(self):
        result = self.files.list_directory(".")
        self.assertTrue(result["success"])
        names = [entry["name"] for entry in result["files"]]
        self.assertEqual(names, ["sub", "notes.txt"])  # directories first
        self.assertEqual(result["files"][0]["type"], "dir")

    def test_read_text_file(self):
        result = self.files.read_text_file("notes.txt")
        self.assertTrue(result["success"])
        self.assertEqual(result["content"], "hello workspace")

    def test_read_nested_file(self):
        self.assertEqual(
            self.files.read_text_file("sub/inner.txt")["content"], "inner"
        )

    def test_relative_escape_is_rejected(self):
        with self.assertRaises(FilesystemToolError) as ctx:
            self.files.read_text_file("../outside.txt")
        self.assertIn("outside the workspace", str(ctx.exception))

    def test_deep_relative_escape_is_rejected(self):
        with self.assertRaises(FilesystemToolError):
            self.files.read_text_file("../../etc/hosts")

    def test_absolute_path_outside_is_rejected(self):
        with self.assertRaises(FilesystemToolError):
            self.files.list_directory(str(self.root.parent))

    def test_missing_file(self):
        with self.assertRaises(FilesystemToolError) as ctx:
            self.files.read_text_file("nope.txt")
        self.assertIn("does not exist", str(ctx.exception))

    def test_reading_a_directory_is_rejected(self):
        with self.assertRaises(FilesystemToolError):
            self.files.read_text_file("sub")

    def test_oversized_file_is_rejected(self):
        (self.root / "big.txt").write_text("x" * 2000, encoding="utf-8")
        with self.assertRaises(FilesystemToolError) as ctx:
            self.files.read_text_file("big.txt")
        self.assertIn("over the", str(ctx.exception))

    def test_only_read_only_tools_are_exposed(self):
        names = {tool.name for tool in self.files.tools()}
        self.assertEqual(names, {"list_directory", "read_text_file"})


class OcrValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "scan.png").write_bytes(b"\x89PNG fake")
        (self.root / "notes.txt").write_text("not an image", encoding="utf-8")
        (self.root / "empty.png").write_bytes(b"")
        config = Config(workspace=self.root, ocr_max_image_bytes=100)
        self.client = OcrClient(config, WorkspaceFiles(self.root))

    def tearDown(self):
        self._tmp.cleanup()

    def test_valid_image_passes_validation(self):
        self.assertEqual(self.client.validate("scan.png").name, "scan.png")

    def test_wrong_extension_is_rejected(self):
        with self.assertRaises(OcrError) as ctx:
            self.client.validate("notes.txt")
        self.assertIn("Unsupported file type", str(ctx.exception))

    def test_missing_file_is_rejected(self):
        with self.assertRaises(OcrError):
            self.client.validate("nope.png")

    def test_empty_file_is_rejected(self):
        with self.assertRaises(OcrError):
            self.client.validate("empty.png")

    def test_escape_is_rejected(self):
        with self.assertRaises(OcrError) as ctx:
            self.client.validate("../outside.png")
        self.assertIn("outside the workspace", str(ctx.exception))

    def test_oversized_image_is_rejected(self):
        (self.root / "big.png").write_bytes(b"x" * 500)
        with self.assertRaises(OcrError) as ctx:
            self.client.validate("big.png")
        self.assertIn("over the", str(ctx.exception))


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "a.txt").write_text("A", encoding="utf-8")
        self.config = Config(workspace=self.root)
        self.registry, self.disabled = build_default_registry(self.config)

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_tools(self):
        self.assertEqual(
            self.registry.names(), ["calculate", "list_directory", "read_text_file"]
        )

    def test_risky_tools_are_disabled_by_default(self):
        # Asserted by membership, not as an exact set: adding a new optional
        # tool should not break an unrelated test, which it did three times.
        categories = {item.category for item in self.disabled}
        for risky in ("python", "terminal", "http", "file writes"):
            self.assertIn(risky, categories)

    def test_nothing_risky_is_registered_by_default(self):
        # The real invariant behind the default set: only reading and
        # arithmetic are offered without an explicit opt-in.
        self.assertEqual(
            self.registry.names(), ["calculate", "list_directory", "read_text_file"]
        )

    def test_python_tool_registers_when_enabled(self):
        registry, disabled = build_default_registry(
            Config(workspace=self.root, python_tool_enabled=True)
        )
        self.assertIn("run_python", registry.names())
        self.assertNotIn("python", {item.category for item in disabled})

    def test_definitions_are_openai_shaped(self):
        definition = self.registry.get_tool_definitions()[0]
        self.assertEqual(definition["type"], "function")
        self.assertIn("name", definition["function"])
        self.assertIn("parameters", definition["function"])

    def test_get_tool_resolves(self):
        self.assertEqual(self.registry.get_tool("calculate").category, "calculator")

    def test_get_tool_rejects_unknown(self):
        from tools.base import UnknownToolError

        with self.assertRaises(UnknownToolError):
            self.registry.get_tool("nope")

    def test_execute_valid_tool(self):
        result = self.registry.execute("read_text_file", {"path": "a.txt"})
        self.assertTrue(result.ok)
        self.assertEqual(result.payload["content"], "A")

    def test_execute_invalid_tool(self):
        result = self.registry.execute("nope", {})
        self.assertFalse(result.ok)
        self.assertIn("No tool named", result.payload["error"])

    def test_execute_malformed_arguments(self):
        result = self.registry.execute("read_text_file", {"wrong": 1})
        self.assertFalse(result.ok)
        self.assertIn("Unknown argument", result.payload["error"])

    def test_execute_contains_path_escape(self):
        result = self.registry.execute("read_text_file", {"path": "../secret"})
        self.assertFalse(result.ok)
        self.assertIn("outside the workspace", result.payload["error"])

    def test_optional_argument_may_be_omitted(self):
        self.assertTrue(self.registry.execute("list_directory", {}).ok)

    def test_categories_group_tools(self):
        categories = self.registry.categories()
        self.assertEqual(sorted(categories), ["calculator", "filesystem"])
        self.assertEqual(len(categories["filesystem"]), 2)


if __name__ == "__main__":
    unittest.main()


class ToolResultTruncationTests(unittest.TestCase):
    """A tool result has to fit the model's context, or the turn is lost.

    Several tools are unbounded - a page of OCR, a file read - and on a
    4,096-token model the prompt and tool schemas already cost about 1,080
    tokens. One dense page overflows the window and fails the whole turn
    rather than one result.
    """

    def result(self, text: str) -> ToolResult:
        return ToolResult(
            "ocr_image",
            {
                "success": True,
                "path": "scan.png",
                "text": text,
                "characters": len(text),
                "backend": "model",
            },
        )

    def test_a_small_result_is_untouched(self):
        result = self.result("a short line of text")
        self.assertEqual(result.content_within(4000), result.content)
        self.assertNotIn("truncated", json.loads(result.content_within(4000)))

    def test_a_limit_of_zero_means_no_limit(self):
        result = self.result("x" * 50_000)
        self.assertEqual(len(result.content_within(0)), len(result.content))

    def test_an_oversized_result_is_cut_to_the_limit(self):
        result = self.result("x" * 20_000)
        capped = result.content_within(3_000)
        self.assertLessEqual(len(capped), 3_000)

    def test_the_cut_result_is_still_valid_json(self):
        # Cutting the serialised text instead of the payload would produce a
        # broken JSON string, which the model cannot read at all.
        payload = json.loads(self.result("x" * 20_000).content_within(3_000))
        self.assertTrue(payload["success"])
        self.assertEqual(payload["path"], "scan.png")

    def test_the_model_is_told_what_is_missing(self):
        # A model handed half a page with no indication will summarise it as
        # though it were the whole thing.
        payload = json.loads(self.result("x" * 20_000).content_within(3_000))
        self.assertIn("truncated", payload)
        self.assertIn("20,000", payload["truncated"])
        self.assertIn("not shown", payload["truncated"])

    def test_the_true_length_survives_the_cut(self):
        payload = json.loads(self.result("x" * 20_000).content_within(3_000))
        self.assertEqual(payload["characters"], 20_000)
        self.assertLess(len(payload["text"]), 20_000)

    def test_as_much_text_as_fits_is_kept(self):
        # The point is to keep the useful part, not to throw the field away -
        # which is what the first attempt at this did.
        payload = json.loads(self.result("x" * 20_000).content_within(3_000))
        self.assertGreater(len(payload["text"]), 2_000)

    def test_the_longest_field_is_the_one_cut(self):
        result = ToolResult(
            "read_text_file",
            {"success": True, "path": "a" * 200, "content": "x" * 20_000},
        )
        payload = json.loads(result.content_within(3_000))
        self.assertEqual(len(payload["path"]), 200)
        self.assertLess(len(payload["content"]), 20_000)

    def test_a_result_that_is_all_metadata_still_returns_json(self):
        result = ToolResult("odd", {"success": True, "n": 1, "m": 2})
        self.assertTrue(json.loads(result.content_within(10))["success"])

    def test_an_unserialisable_payload_is_reported_not_raised(self):
        result = ToolResult("odd", {"success": True, "thing": object()})
        # default=str handles it; the contract is that it never raises.
        self.assertIsInstance(result.content_within(3_000), str)


class ToolResultBudgetTests(unittest.TestCase):
    """The cap is a share of the model's own context, not a fixed number."""

    def test_the_budget_scales_with_the_context(self):
        small = dataclasses.replace(Config(), model_context=4096)
        large = dataclasses.replace(Config(), model_context=32768)
        self.assertLess(small.max_tool_result_chars, large.max_tool_result_chars)

    def test_a_quarter_of_the_context_by_default(self):
        config = dataclasses.replace(Config(), model_context=4096)
        # 4096 / 4 = 1024 tokens, at the measured 3.27 characters per token.
        self.assertEqual(config.max_tool_result_chars, int(1024 * 3.27))

    def test_a_tiny_context_still_leaves_a_usable_floor(self):
        config = dataclasses.replace(Config(), model_context=512)
        self.assertGreaterEqual(config.max_tool_result_chars, 256)

    def test_the_share_is_configurable(self):
        half = dataclasses.replace(Config(), model_context=4096, tool_result_share=0.5)
        quarter = dataclasses.replace(Config(), model_context=4096)
        self.assertGreater(half.max_tool_result_chars, quarter.max_tool_result_chars)
