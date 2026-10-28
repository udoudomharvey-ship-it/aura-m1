"""Run: python -m unittest aura_sdk_tests.test_voice_thinking -v"""
import sys, os, json, io, unittest, urllib.error
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aura_sdk import voice


class Resp:
    def __init__(self, p): self.p = p
    def read(self): return json.dumps(self.p).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False

OK = {"content": [{"type": "text", "text": "Hello there."}]}


class TestThinkingOff(unittest.TestCase):
    def test_thinking_disabled_in_request(self):
        seen = []
        def opener(req, timeout):
            seen.append(json.loads(req.data)); return Resp(OK)
        self.assertEqual(voice.call_claude("S", [], key="K", opener=opener), "Hello there.")
        self.assertEqual(seen[0]["thinking"], {"type": "disabled"})
    def test_400_retries_without_thinking(self):
        seen = []
        def opener(req, timeout):
            b = json.loads(req.data); seen.append(b)
            if "thinking" in b:
                raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"{}"))
            return Resp(OK)
        self.assertEqual(voice.call_claude("S", [], key="K", opener=opener), "Hello there.")
        self.assertEqual(len(seen), 2); self.assertNotIn("thinking", seen[1])
    def test_other_http_errors_are_not_retried(self):
        n = []
        def opener(req, timeout):
            n.append(1); raise urllib.error.HTTPError("u", 401, "no", {}, io.BytesIO(b"{}"))
        with self.assertRaises(urllib.error.HTTPError):
            voice.call_claude("S", [], key="K", opener=opener)
        self.assertEqual(len(n), 1)
    def test_thinking_only_reply_still_reports_cause(self):
        opener = lambda req, timeout: Resp({"content": [{"type": "thinking"}], "stop_reason": "max_tokens"})
        with self.assertRaises(ValueError) as cm:
            voice.call_claude("S", [], key="K", opener=opener)
        self.assertIn("thinking", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
