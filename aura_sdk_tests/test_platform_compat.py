"""Windows/macOS support, checked without a Windows machine: platform is faked, subprocess is captured."""
import os, sys, tempfile, unittest
from unittest import mock
from aura_sdk import platform_compat as C, envfile, voice


class WindowsPaths(unittest.TestCase):
    def test_config_and_data_dirs(self):
        with mock.patch.object(sys, "platform", "win32"), mock.patch.dict(os.environ, {"APPDATA": r"C:\Users\h\AppData\Roaming", "LOCALAPPDATA": r"C:\Users\h\AppData\Local"}):
            self.assertTrue(C.config_dir().startswith(r"C:\Users\h\AppData\Roaming"))
            self.assertTrue(C.config_dir().endswith("aura-m1"))
            self.assertIn("Local", C.vosk_model_dir())

    def test_linux_paths_unchanged(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(C.config_dir(), os.path.expanduser("~/.config/aura-m1"))


class WindowsSpeechAndClipboard(unittest.TestCase):
    def test_speech_goes_through_powershell_on_stdin(self):
        calls = []
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch("shutil.which", side_effect=lambda n: "C:\\ps.exe" if n == "powershell" else None), \
             mock.patch("subprocess.run", side_effect=lambda cmd, **kw: calls.append((cmd, kw))):
            self.assertTrue(voice._default_speak("Time for a short reset."))
        cmd, kw = calls[0]
        self.assertEqual(cmd[0], "C:\\ps.exe")
        self.assertIn("System.Speech", cmd[-1])
        self.assertEqual(kw["input"], b"Time for a short reset.")
        self.assertNotIn("Time for a short reset.", " ".join(cmd))   # text never put on the command line

    def test_no_powershell_means_no_speech_not_a_crash(self):
        with mock.patch.object(sys, "platform", "win32"), mock.patch("shutil.which", return_value=None):
            self.assertFalse(voice._default_speak("hello"))

    def test_clipboard_uses_get_clipboard(self):
        with mock.patch.object(sys, "platform", "win32"), mock.patch("shutil.which", return_value="C:\\ps.exe"):
            cmds = C.clipboard_commands()
        self.assertIn("Get-Clipboard -Raw", cmds[0][-1])

    def test_clipboard_on_linux_tries_wayland_then_x11(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual([c[0] for c in C.clipboard_commands()], ["wl-paste", "xclip", "xsel"])

    def test_wav_plays_with_winsound_on_windows(self):
        fake = mock.MagicMock(SND_FILENAME=2)
        with mock.patch.object(sys, "platform", "win32"), mock.patch.dict(sys.modules, {"winsound": fake}):
            self.assertTrue(C.play_wav("x.wav"))
        fake.PlaySound.assert_called_once_with("x.wav", 2)


class EnvFile(unittest.TestCase):
    def test_parses_bash_and_windows_styles(self):
        d = envfile.parse('# c\nexport AURA_ANTHROPIC_KEY="abc"\nset AURA_VOICE_ENGINE=piper\nexport AURA_PIPER_MODEL=$HOME/v.onnx\nbad line\n')
        self.assertEqual(d["AURA_ANTHROPIC_KEY"], "abc")
        self.assertEqual(d["AURA_VOICE_ENGINE"], "piper")
        self.assertEqual(d["AURA_PIPER_MODEL"], os.path.expanduser("~") + "/v.onnx")

    def test_does_not_override_existing_and_never_raises(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "a.env"); open(p, "w").write("export AURA_ANTHROPIC_KEY=new\nexport AURA_X=1\n")
            env = {"AURA_ANTHROPIC_KEY": "keep"}
            self.assertEqual(envfile.load_env_file(p, env), ["AURA_X"])
            self.assertEqual(env["AURA_ANTHROPIC_KEY"], "keep")
            self.assertEqual(envfile.load_env_file(os.path.join(t, "missing.env"), {}), [])


class Launcher(unittest.TestCase):
    def test_plan(self):
        import aura
        script, extra, url, banner = aura.plan(["demo", "--voice"], root="/r", env={})
        self.assertTrue(script[0].endswith("dashboard_server.py"))
        self.assertEqual(extra["AURA_VOICE"], "1")
        self.assertTrue(extra["AURA_BNN_MODEL"].endswith("demo_placeholder_bnn.json"))
        self.assertTrue(url.endswith("/aura.html"))
        self.assertNotIn("AURA_BNN_MODEL", aura.plan(["live"], root="/r", env={})[1])
        self.assertIsNone(aura.plan(["help"], root="/r", env={})[0])
        self.assertEqual(aura.plan(["test"], root="/r", env={})[0][:2], ["-m", "unittest"])


if __name__ == "__main__":
    unittest.main()
