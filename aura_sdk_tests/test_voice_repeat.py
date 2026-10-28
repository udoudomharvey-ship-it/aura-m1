"""Run: python -m unittest aura_sdk_tests.test_voice_repeat -v"""
import sys, os, json, time, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from types import SimpleNamespace as NS
from aura_sdk import voice
from aura_sdk.recommend import ProblemClass


def rec():
    return NS(why="You've been shifting position a lot.", suggested_actions=["Take a 60-second stretch break"],
              problem_class=ProblemClass.SPATIAL_RESTLESSNESS, state=2)


class Rig(unittest.TestCase):
    def setUp(self):
        os.environ["AURA_ANTHROPIC_KEY"] = "KEY123"
        self.spoken, self.logs, self.payloads = [], [], []
        d = tempfile.mkdtemp()
        self.kw = dict(enabled=True, cooldown_s=0, log=self.logs.append,
                       speak=lambda t: self.spoken.append(t) or True,
                       quiet_file=os.path.join(d, "q"), voice_state_path=os.path.join(d, "v.json"))
    def tearDown(self):
        os.environ.pop("AURA_ANTHROPIC_KEY", None)
    def coach(self, replies):
        it = iter(replies)
        def call(system, messages, **k):
            self.payloads.append(json.loads(messages[0]["content"]))
            return next(it)
        return voice.VoiceCoach(call_fn=call, **self.kw)
    def fire(self, c, n):
        c.announce(rec())
        end = time.time() + 3
        while len(self.spoken) < n and time.time() < end: time.sleep(0.01)
        time.sleep(0.05)


class TestRepeat(Rig):
    def test_repeat_triggers_one_retry_with_instruction(self):
        c = self.coach(["Perhaps a quick stretch would help.", "Perhaps a quick stretch would help.", "A little movement might be nice."])
        self.fire(c, 1); self.fire(c, 2)
        self.assertEqual(self.spoken, ["Perhaps a quick stretch would help.", "A little movement might be nice."])
        self.assertNotIn("avoid_repeating", self.payloads[0]); self.assertIn("avoid_repeating", self.payloads[2])
    def test_repeat_ignores_case_and_punctuation(self):
        c = self.coach(["Take a stretch.", "take a stretch", "Stand up for a moment."])
        self.fire(c, 1); self.fire(c, 2)
        self.assertEqual(self.spoken[1], "Stand up for a moment.")
    def test_still_repeating_falls_back_to_builtin(self):
        c = self.coach(["Take a stretch.", "Take a stretch.", "Take a stretch."])
        self.fire(c, 1); self.fire(c, 2)
        self.assertIn("Take a 60-second stretch break", self.spoken[1])
        self.assertTrue(any("repeated an earlier line" in l for l in self.logs))
    def test_no_retry_when_not_a_repeat(self):
        c = self.coach(["One line.", "A different line."])
        self.fire(c, 1); self.fire(c, 2)
        self.assertEqual(len(self.payloads), 2)


class TestRejectedLogging(Rig):
    def test_rejected_reply_text_is_logged(self):
        c = self.coach(["Stand up. Then stretch."])
        self.fire(c, 1)
        self.assertTrue(any("broke the nudge rules" in l and "Stand up. Then stretch." in l for l in self.logs))
        self.assertIn("Take a 60-second stretch break", self.spoken[0])
    def test_key_never_logged(self):
        c = self.coach(["Stand up. Then stretch."]); self.fire(c, 1)
        self.assertFalse(any("KEY123" in l for l in self.logs))
    def test_skip_still_silent(self):
        c = self.coach(["SKIP"]); c.announce(rec()); time.sleep(0.3)
        self.assertEqual(self.spoken, [])


if __name__ == "__main__":
    unittest.main()
