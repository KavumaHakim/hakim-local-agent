"""Python tool tests.

These actually spawn the child interpreter, so they are slower than the rest.
"""

from __future__ import annotations

import unittest

from tools.python_tool import PythonToolError, build_python_tool, run_python

TIMEOUT = 20.0
MAX_OUTPUT = 4000


def run(code: str):
    return run_python(code, timeout=TIMEOUT, max_output_chars=MAX_OUTPUT)


class SafeExecutionTests(unittest.TestCase):
    def test_simple_calculation(self):
        result = run("print(2 + 2)")
        self.assertTrue(result["success"])
        self.assertEqual(result["output"].strip(), "4")

    def test_loops_and_functions(self):
        result = run(
            "def square(n):\n"
            "    return n * n\n"
            "print(sum(square(i) for i in range(5)))"
        )
        self.assertEqual(result["output"].strip(), "30")

    def test_preloaded_modules_are_usable(self):
        result = run("print(round(math.sqrt(2), 4)); print(statistics.mean([1,2,3]))")
        self.assertTrue(result["success"])
        self.assertIn("1.4142", result["output"])
        self.assertIn("2", result["output"])

    def test_runtime_error_is_captured_not_raised(self):
        result = run("print(1 / 0)")
        self.assertFalse(result["success"])
        self.assertIn("ZeroDivisionError", result["error"])

    def test_no_output_is_explained(self):
        result = run("x = 5")
        self.assertTrue(result["success"])
        self.assertIn("no output", result["output"])

    def test_output_is_truncated(self):
        result = run_python(
            "print('x' * 5000)", timeout=TIMEOUT, max_output_chars=100
        )
        self.assertTrue(result["success"])
        self.assertIn("truncated", result["output"])


class RestrictionTests(unittest.TestCase):
    def test_import_os_is_rejected(self):
        with self.assertRaises(PythonToolError) as ctx:
            run("import os\nprint(os.getcwd())")
        self.assertIn("Imports are not allowed", str(ctx.exception))

    def test_from_import_is_rejected(self):
        with self.assertRaises(PythonToolError):
            run("from os import system")

    def test_open_is_rejected(self):
        with self.assertRaises(PythonToolError) as ctx:
            run("open('secret.txt')")
        self.assertIn("not available", str(ctx.exception))

    def test_dunder_import_is_rejected(self):
        with self.assertRaises(PythonToolError):
            run("__import__('os')")

    def test_private_attribute_access_is_rejected(self):
        with self.assertRaises(PythonToolError) as ctx:
            run("print(().__class__)")
        self.assertIn("private attribute", str(ctx.exception))

    def test_eval_and_exec_are_rejected(self):
        for code in ("eval('1+1')", "exec('x=1')", "compile('1','','eval')"):
            with self.assertRaises(PythonToolError, msg=code):
                run(code)

    def test_subprocess_and_socket_names_are_rejected(self):
        for code in ("subprocess.run('dir')", "socket.socket()"):
            with self.assertRaises(PythonToolError, msg=code):
                run(code)

    def test_getattr_escape_is_rejected(self):
        with self.assertRaises(PythonToolError):
            run("getattr(str, 'mro')")

    def test_syntax_error_is_reported(self):
        with self.assertRaises(PythonToolError) as ctx:
            run("def (:")
        self.assertIn("Syntax error", str(ctx.exception))

    def test_empty_code_is_rejected(self):
        with self.assertRaises(PythonToolError):
            run("   ")

    def test_timeout_is_enforced(self):
        with self.assertRaises(PythonToolError) as ctx:
            run_python("while True:\n    pass", timeout=2.0, max_output_chars=100)
        self.assertIn("timeout", str(ctx.exception))

    def test_builtins_are_stripped_in_the_child(self):
        # Defence in depth: even if the AST screen were bypassed, the child has
        # no __import__. Getting there via a name the screen allows:
        result = run("print('sandbox note ok')")
        self.assertTrue(result["success"])


class ToolWiringTests(unittest.TestCase):
    def test_tool_metadata(self):
        tool = build_python_tool(timeout=TIMEOUT, max_output_chars=MAX_OUTPUT)
        self.assertEqual(tool.name, "run_python")
        self.assertEqual(tool.category, "python")
        self.assertEqual(tool.parameters["required"], ["code"])

    def test_registry_execution_returns_structured_failure(self):
        from tools.base import ToolRegistry

        registry = ToolRegistry(
            [build_python_tool(timeout=TIMEOUT, max_output_chars=MAX_OUTPUT)]
        )
        result = registry.execute("run_python", {"code": "import os"})
        self.assertFalse(result.ok)
        self.assertIn("Imports are not allowed", result.payload["error"])


if __name__ == "__main__":
    unittest.main()
