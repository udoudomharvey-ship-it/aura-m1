#!/usr/bin/env python3
"""Aura-M1 command line for Windows, macOS and Linux.   python aura.py help   (Windows: aura.bat help)"""
import os, subprocess, sys, threading, time, webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from aura_sdk import envfile
from aura_sdk import platform_compat as compat

HELP = """aura demo [--voice]   scripted demo, no camera needed
aura live [--voice]   real webcam session (needs: install --live)
aura voice            hear the four spoken nudges without any camera
aura chat             type to Aura, hear replies (needs an API key)
aura listen           say "Hey Aura" and talk (needs: install --listen)
aura doctor           check speaker and voice setup
aura test             run the automated tests
Add --no-browser to demo/live to skip opening the dashboard page."""


def plan(argv, root=ROOT, env=None):
    """Returns (script_and_args, extra_env, url_or_None, banner_lines). Pure, so it can be tested."""
    env = dict(os.environ if env is None else env)
    args = [a for a in argv if a not in ("--no-browser",)]
    cmd = args[0] if args else "help"
    voice = "--voice" in args
    extra, banner, url = {}, [], None
    if voice:
        extra["AURA_VOICE"] = "1"
    if cmd == "demo":
        extra["AURA_BNN_MODEL"] = os.path.join(root, "models", "demo_placeholder_bnn.json")
        banner = ["DEMO MODE: scripted scenarios, placeholder model. Not a reading of a real person."]
        url = "http://localhost:8765/aura.html"
        return [os.path.join(root, "dashboard_server.py")], extra, url, banner
    if cmd == "live":
        banner = ["LIVE MODE: webcam. Detection is NOT validated on real people yet."]
        url = "http://localhost:8765/aura.html"
        return [os.path.join(root, "run_live.py")], extra, url, banner
    if cmd == "voice":
        extra["AURA_VOICE"] = "1"
        return [os.path.join(root, "demo_voice.py")], extra, None, banner
    if cmd == "chat":
        return [os.path.join(root, "tools", "talk.py")], extra, None, banner
    if cmd == "listen":
        return [os.path.join(root, "tools", "listen.py")], extra, None, banner
    if cmd == "doctor":
        return [os.path.join(root, "tools", "voice_selftest.py"), "--no-claude"], extra, None, banner
    if cmd == "test":
        return ["-m", "unittest", "discover", "-s", "aura_sdk_tests", "-t", "."], extra, None, banner
    return None, extra, None, banner


def main(argv):
    script, extra, url, banner = plan(argv)
    if script is None:
        print(HELP); return 0
    envfile.load_env_file()
    env = dict(os.environ); env.update(extra)
    for line in banner:
        print(line)
    if url:
        print(f"Open {url}   (classic view: /dashboard.html)")
        if "--no-browser" not in argv:
            threading.Timer(3.0, lambda: webbrowser.open(url)).start()
    try:
        return subprocess.call([sys.executable] + script, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
