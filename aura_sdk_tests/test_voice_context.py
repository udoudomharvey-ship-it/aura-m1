"""Run: python -m unittest aura_sdk_tests.test_voice_context -v"""
import sys, os, json, time, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from aura_sdk import voice_context as vc
from aura_sdk import voice


class Ctx(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.sp, self.vp = self.d / "dashboard_state.json", self.d / "voice_state.json"
    def live(self, **kw):
        st = {"mode": "live", "bnn_state": 2, "alert": False, "q_hci": 0.31,
              "entropy": {"h_sp": 0.7, "h_te": 0.4, "h_in": 0.2}, "events": []}
        st.update(kw); self.sp.write_text(json.dumps(st))
    def read(self, **kw):
        return vc.read_context(self.sp, self.vp, **kw)

    def test_live_reading_gives_workload(self):
        self.live()
        c = self.read()
        self.assertEqual(c["workload_state_1_to_5"], 2)
        self.assertTrue(any("estimate, not a diagnosis" in n for n in c["available_context"]))
    def test_alert_noted(self):
        self.live(alert=True)
        self.assertTrue(any("alert" in n for n in self.read()["available_context"]))
    def test_nothing_sensitive_leaks(self):
        self.live()
        blob = json.dumps(self.read()).lower()
        for banned in ("h_sp", "h_te", "h_in", "q_hci", "entropy", "0.31", "0.7"):
            self.assertNotIn(banned, blob)
    def test_stale_file_means_not_running(self):
        self.live()
        c = self.read(now=lambda: time.time() + 60)
        self.assertNotIn("workload_state_1_to_5", c)
        self.assertTrue(any("not running" in n for n in c["available_context"]))
    def test_missing_file_means_not_running(self):
        c = self.read()
        self.assertTrue(any("not running" in n for n in c["available_context"]))
    def test_demo_mode_is_ignored(self):
        self.live(mode="demo")
        self.assertNotIn("workload_state_1_to_5", self.read())
    def test_garbled_state_is_survived(self):
        self.sp.write_text('{"mode": "live", "bnn_st')
        self.assertNotIn("workload_state_1_to_5", self.read())
    def test_out_of_range_state_ignored(self):
        self.live(bnn_state=9)
        self.assertNotIn("workload_state_1_to_5", self.read())
    def test_voice_state_roundtrip(self):
        self.assertTrue(vc.write_voice_state(["a", "b"], "spatial_restlessness", 2, path=self.vp))
        c = self.read()
        self.assertEqual(c["recent_interventions"], ["a", "b"])
        self.assertTrue(any("spatial restlessness" in n for n in c["available_context"]))
    def test_old_nudges_are_not_recent(self):
        vc.write_voice_state(["old"], "x", 1, path=self.vp, now=lambda: time.time() - 7200)
        self.assertNotIn("recent_interventions", self.read())
    def test_write_never_raises(self):
        self.assertFalse(vc.write_voice_state(["a"], "p", 1, path=self.d / "nope" / "x.json"))


class CoachBridge(unittest.TestCase):
    def setUp(self):
        os.environ["AURA_ANTHROPIC_KEY"] = "KEY"
        self.calls = []
        self.qf = os.path.join(tempfile.mkdtemp(), "quiet")
    def tearDown(self):
        os.environ.pop("AURA_ANTHROPIC_KEY", None)
    def coach(self, ctx):
        def call(system, messages, **k):
            self.calls.append(messages); return "Sure."
        return voice.VoiceCoach(enabled=True, log=lambda s: None, speak=lambda t: True,
                                call_fn=call, context_fn=lambda: ctx, quiet_file=self.qf)
    def test_chat_request_carries_live_context(self):
        c = self.coach({"workload_state_1_to_5": 1, "available_context": ["note"], "recent_interventions": ["x"]})
        c.converse("I'm swamped")
        p = json.loads(self.calls[0][-1]["content"])
        self.assertEqual(p["workload_state_1_to_5"], 1)
        self.assertEqual(p["available_context"], ["note"])
        self.assertEqual(p["recent_interventions"], ["x"])
    def test_context_failure_does_not_break_chat(self):
        def boom(): raise RuntimeError("x")
        def call(system, messages, **k): return "Fine."
        c = voice.VoiceCoach(enabled=True, log=lambda s: None, speak=lambda t: True,
                             call_fn=call, context_fn=boom, quiet_file=self.qf)
        self.assertEqual(c.converse("hello"), "Fine.")
    def test_set_quiet_is_shared_through_the_file(self):
        a, b = self.coach({}), self.coach({})
        a.set_quiet(True)
        self.assertTrue(os.path.exists(self.qf)); self.assertTrue(b.is_quiet())
        a.set_quiet(False)
        self.assertFalse(b.is_quiet())


if __name__ == "__main__":
    unittest.main()
