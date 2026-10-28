"""Run: python -m unittest aura_sdk_tests.test_voice -v"""
import sys, os, json, time, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from types import SimpleNamespace as NS
from aura_sdk import voice
from aura_sdk.recommend import ProblemClass


def rec(problem=ProblemClass.SPATIAL_RESTLESSNESS, why="You've been shifting position a lot.",
        actions=("Take a 60-second stretch break",)):
    return NS(why=why, suggested_actions=list(actions), problem_class=problem, state=2)


class FakeResp:
    def __init__(self, payload): self.p = payload
    def read(self): return json.dumps(self.p).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False


class TestHelpers(unittest.TestCase):
    def test_fallback_combines(self):
        self.assertEqual(voice.fallback_line("You look tired.", ["Take a walk"]), "You look tired. Take a walk")
    def test_clean_bounds_length(self):
        self.assertLessEqual(len(voice.clean_line("word " * 200)), voice.MAX_SPOKEN_CHARS + 1)
    def test_master_prompt_is_the_full_spec(self):
        t = voice.load_master_prompt()
        for marker in ("ULTIMATE MASTER SYSTEM PROMPT", "SKIP", "proactive_nudge", "Time for a short reset."):
            self.assertIn(marker, t)
        self.assertGreater(len(t), 20000)
    def test_time_of_day(self):
        import datetime as d
        f = lambda h: voice.time_of_day(d.datetime(2026, 1, 1, h))
        self.assertEqual([f(6), f(13), f(18), f(23), f(2)], ["morning", "afternoon", "evening", "late night", "late night"])
    def test_route_mode(self):
        self.assertEqual(voice.route_mode("Create a reminder for my report"), "authorised_action")
        self.assertEqual(voice.route_mode("I'm struggling to organise this project"), "active_problem_solving")
        self.assertEqual(voice.route_mode("Thanks, that helped a lot"), "conversational_assistance")


class TestValidate(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(voice.validate_nudge("Perhaps a quick stretch would help."), "Perhaps a quick stretch would help.")
    def test_skip(self):
        self.assertEqual(voice.validate_nudge(" SKIP "), "SKIP")
    def test_two_sentences_rejected(self):
        self.assertIsNone(voice.validate_nudge("Stand up. Then stretch."))
    def test_too_long_rejected(self):
        self.assertIsNone(voice.validate_nudge(" ".join(["word"] * 23) + "."))
    def test_exactly_22_words_ok(self):
        self.assertIsNotNone(voice.validate_nudge(" ".join(["word"] * 22) + "."))
    def test_markdown_rejected(self):
        self.assertIsNone(voice.validate_nudge("**Stretch** now."))
    def test_empty_rejected(self):
        self.assertIsNone(voice.validate_nudge(""))


class TestCallClaude(unittest.TestCase):
    def test_request_shape(self):
        seen = {}
        def opener(req, timeout):
            seen["h"] = dict(req.header_items()); seen["b"] = json.loads(req.data)
            return FakeResp({"content": [{"type": "text", "text": "Hello."}]})
        out = voice.call_claude("SYS", [{"role": "user", "content": "{}"}], key="K", opener=opener)
        self.assertEqual(out, "Hello.")
        self.assertIn("X-api-key", seen["h"])
        blk = seen["b"]["system"][0]
        self.assertEqual(blk["text"], "SYS"); self.assertEqual(blk["cache_control"], {"type": "ephemeral"})
    def test_bearer(self):
        seen = {}
        def opener(req, timeout):
            seen["h"] = dict(req.header_items()); return FakeResp({"content": [{"type": "text", "text": "ok"}]})
        voice.call_claude("S", [], key="K", auth="bearer", opener=opener)
        self.assertIn("Authorization", seen["h"])
    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            voice.call_claude("S", [], key="K", opener=lambda r, timeout: FakeResp({"content": []}))


class Base(unittest.TestCase):
    def setUp(self):
        self.spoken, self.logs, self.calls = [], [], []
        self.t = [1000.0]
        self.qf = os.path.join(tempfile.mkdtemp(), "quiet")
        os.environ["AURA_ANTHROPIC_KEY"] = "KEY123"
    def tearDown(self):
        os.environ.pop("AURA_ANTHROPIC_KEY", None)
    def mk(self, reply="Perhaps a quick stretch would help.", **kw):
        def call(system, messages, **k):
            self.calls.append((system, messages, k))
            if isinstance(reply, Exception): raise reply
            return reply
        kw.setdefault("enabled", True)
        return voice.VoiceCoach(log=self.logs.append, speak=lambda x: self.spoken.append(x) or True,
                                call_fn=call, clock=lambda: self.t[0], quiet_file=self.qf, **kw)
    def wait(self, n=1):
        end = time.time() + 3
        while len(self.spoken) < n and time.time() < end: time.sleep(0.01)
        time.sleep(0.05)


class TestNudges(Base):
    def test_disabled_by_default(self):
        os.environ.pop("AURA_VOICE", None)
        self.assertFalse(voice.VoiceCoach(log=lambda s: None).announce(rec()))
    def test_payload_matches_prompt_schema(self):
        c = self.mk(); c.announce(rec()); self.wait()
        sys_, msgs, kw = self.calls[0]
        p = json.loads(msgs[0]["content"])
        for k in ("interaction_mode", "problem", "workload_state_1_to_5", "why", "suggested_action",
                  "episode_number", "minutes_into_session", "time_of_day", "quiet_mode", "recent_interventions"):
            self.assertIn(k, p)
        self.assertEqual(p["interaction_mode"], "proactive_nudge")
        self.assertEqual(p["episode_number"], 1)
        self.assertFalse(p["quiet_mode"])
        self.assertEqual(sys_, voice.load_master_prompt())
        self.assertEqual(self.spoken, ["Perhaps a quick stretch would help."])
        self.assertEqual(c.last_source, "claude")
    def test_payload_has_no_body_or_video_data(self):
        c = self.mk(); c.announce(rec()); self.wait()
        blob = self.calls[0][1][0]["content"].lower()
        for banned in ("frame", "joint", "h_sp", "h_te", "h_in", "entropy", "q_hci", "image", "pixel"):
            self.assertNotIn(banned, blob)
    def test_recent_interventions_and_episode_progress(self):
        c = self.mk(); c.announce(rec()); self.wait(1)
        self.t[0] += 200; c.announce(rec()); self.wait(2)
        p = json.loads(self.calls[1][1][0]["content"])
        self.assertEqual(p["episode_number"], 2)
        self.assertEqual(p["recent_interventions"], ["Perhaps a quick stretch would help."])
    def test_cooldown(self):
        c = self.mk(); self.assertTrue(c.announce(rec())); self.wait()
        self.t[0] += 10; self.assertFalse(c.announce(rec()))
        self.t[0] += 100; self.assertTrue(c.announce(rec()))
    def test_quiet_flag_never_calls_api(self):
        c = self.mk(); c.quiet_flag = True
        self.assertFalse(c.announce(rec())); time.sleep(0.1)
        self.assertEqual(self.calls, [])
    def test_quiet_file(self):
        c = self.mk(); open(self.qf, "w").close()
        self.assertFalse(c.announce(rec()))
        os.remove(self.qf); self.assertTrue(c.announce(rec()))
    def test_skip_reply_is_silent(self):
        c = self.mk(reply="SKIP"); c.announce(rec()); time.sleep(0.3)
        self.assertEqual(self.spoken, [])
    def test_invalid_reply_falls_back_to_builtin(self):
        c = self.mk(reply="Stand up. Then stretch."); c.announce(rec()); self.wait()
        self.assertEqual(c.last_source, "built-in")
        self.assertIn("Take a 60-second stretch break", self.spoken[0])
    def test_api_failure_falls_back_and_never_logs_key(self):
        c = self.mk(reply=OSError("net down KEY123")); c.announce(rec()); self.wait()
        self.assertEqual(c.last_source, "built-in")
        self.assertFalse(any("KEY123" in l for l in self.logs))
    def test_missing_action_uses_exact_fallback_without_api(self):
        c = self.mk(); c.announce(rec(actions=())); self.wait()
        self.assertEqual(self.spoken, ["Time for a short reset."]); self.assertEqual(self.calls, [])
    def test_declines_stretch_cooldown(self):
        c = self.mk(); c.announce(rec()); self.wait()
        c.mark_declined()
        self.t[0] += 100; self.assertFalse(c.announce(rec()))     # 90 s x2 = 180 s
        self.t[0] += 100; self.assertTrue(c.announce(rec()))
    def test_done_suppresses_same_problem_only(self):
        c = self.mk(); c.announce(rec()); self.wait(); c.mark_done()
        self.t[0] += 200
        self.assertFalse(c.announce(rec()))
        self.assertTrue(c.announce(rec(problem=ProblemClass.COGNITIVE_OVERLOAD)))
    def test_no_key_uses_builtin(self):
        os.environ.pop("AURA_ANTHROPIC_KEY", None)
        c = self.mk(); c.announce(rec()); self.wait()
        self.assertEqual(self.calls, []); self.assertEqual(c.last_source, "built-in")


class TestConversation(Base):
    def test_modes_and_history(self):
        c = self.mk(reply="Let's make it manageable. Which section needs the most work?")
        r = c.converse("I'm struggling to organise this project")
        self.assertTrue(r)
        p = json.loads(self.calls[0][1][-1]["content"])
        self.assertEqual(p["interaction_mode"], "active_problem_solving")
        self.assertEqual(p["user_message"], "I'm struggling to organise this project")
        self.assertEqual(p["available_integrations"], [])
        c.converse("The methodology")
        msgs = self.calls[1][1]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user"])
    def test_history_is_capped(self):
        c = self.mk(reply="ok")
        for i in range(20): c.converse(f"message {i}")
        self.assertLessEqual(len(c.history), 2 * c.MAX_TURNS)
        self.assertEqual(c.history[0]["role"], "user")
    def test_authorised_action_mode(self):
        c = self.mk(reply="I can't set reminders yet.")
        c.converse("Create a reminder for my report")
        self.assertEqual(json.loads(self.calls[0][1][-1]["content"])["interaction_mode"], "authorised_action")
    def test_speaks_reply_when_enabled(self):
        c = self.mk(reply="Sure."); c.converse("hello"); self.assertEqual(self.spoken, ["Sure."])
    def test_failure_returns_empty_and_keeps_history_clean(self):
        c = self.mk(reply=OSError("x")); self.assertEqual(c.converse("hi"), ""); self.assertEqual(c.history, [])
    def test_empty_input(self):
        self.assertEqual(self.mk().converse("   "), "")


if __name__ == "__main__":
    unittest.main()
