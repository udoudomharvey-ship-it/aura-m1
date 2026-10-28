import unittest
from aura_sdk import listen as L


class FakeCoach:
    enabled = False; muted = False
    def __init__(self): self.q = None; self.calls = []
    def set_quiet(self, on): self.q = on
    def converse(self, text, speak=True, extra=None):
        self.calls.append((text, extra)); return "Try restarting the server."
    def _speak(self, t): pass


class ListenLogic(unittest.TestCase):
    def test_wake(self):
        self.assertTrue(L.is_wake("hey aura"))
        self.assertTrue(L.is_wake("hey aura set a timer"))
        self.assertFalse(L.is_wake("play the aura album"))
        self.assertEqual(L.strip_wake("hey aura set a timer for five minutes"), "set a timer for five minutes")

    def test_timer(self):
        self.assertEqual(L.parse_timer("set a timer for 10 minutes"), (600, ""))
        self.assertEqual(L.parse_timer("remind me in five minutes to stretch"), (300, "stretch"))
        self.assertEqual(L.parse_timer("set a timer for an hour"), (3600, ""))
        self.assertIsNone(L.parse_timer("set a timer for 2 seconds"))
        self.assertIsNone(L.parse_timer("set a timer for 9 hours"))
        self.assertIsNone(L.parse_timer("how do I make a timer in python"))

    def test_classify(self):
        self.assertEqual(L.classify("go quiet")[0], "quiet_on")
        self.assertEqual(L.classify("you can talk again")[0], "quiet_off")
        self.assertEqual(L.classify("that's all")[0], "end")
        self.assertEqual(L.classify("what's wrong with this")[0], "clipboard_chat")
        self.assertEqual(L.classify("why is my loop slow")[0], "chat")

    def test_fillers_and_wake_fragments_are_not_requests(self):
        for t in ("hey or", "yes", "the", "hey aura", "uh"):
            self.assertTrue(L.is_filler(L.strip_wake(t)) or L.strip_wake(t) == "", t)
        self.assertEqual(L.strip_wake("hey or"), "")
        self.assertFalse(L.is_filler("why is my loop slow"))

    def test_clipboard_wrapping_is_bounded_and_marked_as_data(self):
        out = L.with_clipboard("look at this", "x" * 9000)
        self.assertIn("<clipboard>", out); self.assertIn("[truncated]", out)
        self.assertLess(len(out), 4300)
        self.assertIn("empty", L.with_clipboard("look at this", None))


class ListenerFlow(unittest.TestCase):
    def mk(self, clip="Traceback: boom"):
        self.said = []; self.coach = FakeCoach()
        return L.Listener(self.coach, say=self.said.append, log=lambda s: None, clip_fn=lambda: clip)

    def test_chat_uses_only_real_host_actions(self):
        ls = self.mk(); self.assertTrue(ls.handle("why is my loop slow"))
        self.assertEqual(self.coach.calls[0][1], {"available_integrations": L.HOST_ACTIONS})
        self.assertNotIn("<clipboard>", self.coach.calls[0][0])

    def test_clipboard_only_read_on_request(self):
        ls = self.mk(); ls.handle("look at this error")
        self.assertIn("Traceback: boom", self.coach.calls[0][0])
        ls.handle("thanks")
        self.assertNotIn("Traceback", self.coach.calls[1][0])

    def test_quiet_ends_conversation_and_sets_flag(self):
        ls = self.mk(); self.assertFalse(ls.handle("go quiet")); self.assertTrue(self.coach.q)

    def test_bare_duration_after_timer_question(self):
        self.assertEqual(L.parse_duration("just one minutes"), 60)
        self.assertIsNone(L.parse_duration("why is it slow"))
        ls = self.mk(); ls.awaiting_timer = True
        self.assertTrue(ls.handle("just one minute"))
        self.assertIn("Timer set", self.said[0])
        for t in ls.timers: t.cancel()
        self.assertEqual(self.coach.calls, [])

    def test_echo_of_her_own_voice_is_ignored(self):
        sp = [(100.0, "Timer set for 1 minute.")]
        self.assertTrue(L.is_echo("timer set for", sp, 103.0))
        self.assertTrue(L.is_echo("timer set for one minute", sp, 103.0))
        self.assertFalse(L.is_echo("why is my loop slow", sp, 103.0))
        self.assertFalse(L.is_echo("timer set for", sp, 140.0))      # too old
        self.assertFalse(L.is_echo("set a timer for ten minutes please", sp, 103.0))

    def test_online_stt_optin_and_fallback(self):
        import os, sys, types
        os.environ.pop("AURA_STT", None)
        self.assertFalse(L.online_stt_enabled())
        os.environ["AURA_STT"] = "online"
        self.assertTrue(L.online_stt_enabled())
        os.environ.pop("AURA_STT")
        sys.modules.pop("speech_recognition", None)
        orig = sys.modules.get("speech_recognition")
        fake = types.ModuleType("speech_recognition")
        class AD:
            def __init__(self, *a): pass
        class Rec:
            def recognize_google(self, audio, language=None): return " set a timer for one minute "
        fake.AudioData, fake.Recognizer = AD, Rec
        sys.modules["speech_recognition"] = fake
        try:
            self.assertEqual(L.transcribe_online(b"\x00\x00"), "set a timer for one minute")
            class Bad:
                def recognize_google(self, audio, language=None): raise RuntimeError("offline")
            fake.Recognizer = Bad
            self.assertIsNone(L.transcribe_online(b"\x00\x00"))
        finally:
            sys.modules.pop("speech_recognition", None)

    def test_timer_fires(self):
        ls = self.mk(); self.assertTrue(ls.handle("set a timer for 10 minutes"))
        self.assertIn("Timer set for 10 minutes", self.said[0])
        for t in ls.timers: t.cancel()


if __name__ == "__main__":
    unittest.main()
