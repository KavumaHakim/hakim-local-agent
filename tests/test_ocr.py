"""OCR tool tests: validation, request shape, and response handling.

The HTTP layer is stubbed. These prove the tool builds the right request and
handles every failure sensibly; they cannot prove GLM-OCR accepts that request,
which needs a running server with a projector.
"""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

import requests

from config import Config
from tools.filesystem import WorkspaceFiles
from tools.ocr_tool import DEFAULT_PROMPT, OcrClient, OcrError


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Records requests and replays scripted answers."""

    def __init__(self):
        self.posts: list[dict] = []
        self.post_response = FakeResponse(
            payload={"choices": [{"message": {"content": "INVOICE 4127"}}]}
        )
        self.post_error: Exception | None = None
        self.props = FakeResponse(payload={"modalities": {"vision": True}})

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json})
        if self.post_error:
            raise self.post_error
        return self.post_response

    def get(self, url, timeout=None):
        if url.endswith("/props"):
            return self.props
        return FakeResponse(payload={})


def build(tmp: Path, **config_kwargs):
    config_kwargs.setdefault("ocr_max_image_bytes", 1_000_000)
    config = Config(workspace=tmp, **config_kwargs)
    client = OcrClient(config, WorkspaceFiles(tmp))
    client._session = FakeSession()
    return client


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        (self.tmp / "scan.png").write_bytes(b"\x89PNG fake bytes")
        (self.tmp / "notes.txt").write_text("not an image", encoding="utf-8")
        (self.tmp / "empty.png").write_bytes(b"")
        self.client = build(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_valid_image(self):
        self.assertEqual(self.client.validate("scan.png").name, "scan.png")

    def test_wrong_extension(self):
        with self.assertRaises(OcrError) as ctx:
            self.client.validate("notes.txt")
        self.assertIn("Unsupported file type", str(ctx.exception))

    def test_missing_file(self):
        with self.assertRaises(OcrError):
            self.client.validate("nope.png")

    def test_empty_file(self):
        with self.assertRaises(OcrError):
            self.client.validate("empty.png")

    def test_workspace_escape(self):
        with self.assertRaises(OcrError) as ctx:
            self.client.validate("../outside.png")
        self.assertIn("outside the workspace", str(ctx.exception))

    def test_oversized_image(self):
        big = build(self.tmp, ocr_max_image_bytes=4)
        with self.assertRaises(OcrError) as ctx:
            big.validate("scan.png")
        self.assertIn("over the", str(ctx.exception))


class RequestShapeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.raw = b"\x89PNG pretend image"
        (self.tmp / "scan.png").write_bytes(self.raw)
        self.client = build(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_payload_has_text_and_image_parts(self):
        payload = self.client.build_payload(self.tmp / "scan.png", "read it")
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "read it"})
        self.assertEqual(content[1]["type"], "image_url")

    def test_image_is_a_base64_data_uri(self):
        payload = self.client.build_payload(self.tmp / "scan.png", "x")
        url = payload["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        encoded = url.split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), self.raw)

    def test_mime_type_follows_the_extension(self):
        (self.tmp / "photo.jpg").write_bytes(b"jpegdata")
        payload = self.client.build_payload(self.tmp / "photo.jpg", "x")
        url = payload["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))

    def test_temperature_is_low(self):
        payload = self.client.build_payload(self.tmp / "scan.png", "x")
        self.assertLessEqual(payload["temperature"], 0.2)

    def test_default_prompt_is_used(self):
        self.client.ocr_image("scan.png")
        sent = self.client._session.posts[0]["json"]
        self.assertEqual(sent["messages"][0]["content"][0]["text"], DEFAULT_PROMPT)

    def test_custom_prompt_overrides(self):
        self.client.ocr_image("scan.png", prompt="just the table")
        sent = self.client._session.posts[0]["json"]
        self.assertEqual(sent["messages"][0]["content"][0]["text"], "just the table")

    def test_posts_to_chat_completions(self):
        self.client.ocr_image("scan.png")
        self.assertTrue(
            self.client._session.posts[0]["url"].endswith("/v1/chat/completions")
        )


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        (self.tmp / "scan.png").write_bytes(b"\x89PNG data")
        self.client = build(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_successful_extraction(self):
        result = self.client.ocr_image("scan.png")
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "INVOICE 4127")
        self.assertEqual(result["characters"], len("INVOICE 4127"))

    def test_text_is_stripped(self):
        self.client._session.post_response = FakeResponse(
            payload={"choices": [{"message": {"content": "  spaced  "}}]}
        )
        self.assertEqual(self.client.ocr_image("scan.png")["text"], "spaced")

    def test_empty_reply_is_an_error(self):
        self.client._session.post_response = FakeResponse(
            payload={"choices": [{"message": {"content": "   "}}]}
        )
        with self.assertRaises(OcrError) as ctx:
            self.client.ocr_image("scan.png")
        self.assertIn("no text", str(ctx.exception))

    def test_malformed_reply(self):
        self.client._session.post_response = FakeResponse(payload={"nope": 1})
        with self.assertRaises(OcrError) as ctx:
            self.client.ocr_image("scan.png")
        self.assertIn("Could not read", str(ctx.exception))

    def test_http_error_mentions_mmproj(self):
        self.client._session.post_response = FakeResponse(
            status_code=400, text="image input not supported"
        )
        with self.assertRaises(OcrError) as ctx:
            self.client.ocr_image("scan.png")
        self.assertIn("--mmproj", str(ctx.exception))

    def test_connection_error(self):
        self.client._session.post_error = requests.ConnectionError("refused")
        with self.assertRaises(OcrError) as ctx:
            self.client.ocr_image("scan.png")
        self.assertIn("Could not reach", str(ctx.exception))

    def test_timeout(self):
        self.client._session.post_error = requests.Timeout("slow")
        with self.assertRaises(OcrError) as ctx:
            self.client.ocr_image("scan.png")
        self.assertIn("within", str(ctx.exception))


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        (self.tmp / "scan.png").write_bytes(b"\x89PNG data")
        self.client = build(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_vision_reported_as_dict(self):
        self.client._session.props = FakeResponse(
            payload={"modalities": {"vision": True}}
        )
        self.assertTrue(self.client.supports_images())

    def test_vision_reported_as_list(self):
        self.client._session.props = FakeResponse(payload={"modalities": ["vision"]})
        self.assertTrue(self.client.supports_images())

    def test_text_only_server_is_refused_before_sending(self):
        self.client._session.props = FakeResponse(
            payload={"modalities": {"vision": False}}
        )
        with self.assertRaises(OcrError) as ctx:
            self.client.ocr_image("scan.png")
        self.assertIn("no vision support", str(ctx.exception))
        self.assertEqual(self.client._session.posts, [])  # nothing was sent

    def test_unknown_capability_still_attempts(self):
        # An unrecognised /props shape must not block a working server.
        self.client._session.props = FakeResponse(payload={"something": "else"})
        self.assertIsNone(self.client.supports_images())
        self.assertTrue(self.client.ocr_image("scan.png")["success"])

    def test_unreachable_props_still_attempts(self):
        self.client._session.props = FakeResponse(status_code=404)
        self.assertIsNone(self.client.supports_images())


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_absent_unless_enabled(self):
        from tools.registry import build_default_registry

        registry, disabled = build_default_registry(Config(workspace=self.tmp))
        self.assertNotIn("ocr_image", registry.names())
        self.assertIn("ocr", {item.category for item in disabled})

    def test_registered_when_enabled(self):
        from tools.registry import build_default_registry

        registry, disabled = build_default_registry(
            Config(workspace=self.tmp, ocr_enabled=True)
        )
        self.assertIn("ocr_image", registry.names())
        self.assertNotIn("ocr", {item.category for item in disabled})

    def test_tool_metadata(self):
        client = build(self.tmp)
        tool = client.tool()
        self.assertEqual(tool.category, "ocr")
        self.assertEqual(tool.parameters["required"], ["path"])
        self.assertIn("prompt", tool.parameters["properties"])


if __name__ == "__main__":
    unittest.main()
