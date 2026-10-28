#!/usr/bin/env python3
"""
Check the voice setup end to end, without the camera.
Run:  python tools/voice_selftest.py            (uses the key from your environment)
      python tools/voice_selftest.py --no-claude (speaker test only)
Prints which auth style works and whether a speaker is installed.
The key itself is never printed.
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aura_sdk import envfile; envfile.load_env_file()
from aura_sdk import voice

ap = argparse.ArgumentParser(); ap.add_argument("--no-claude", action="store_true")
a = ap.parse_args()

key = os.environ.get("AURA_ANTHROPIC_KEY", "").strip()
print(f"key present: {'yes (%d chars)' % len(key) if key else 'NO'}")
spk = voice.detect_speaker()
print(f"speaker: {spk[0] if spk else 'NONE -- install one:  sudo apt install espeak-ng'}")

line = voice.fallback_line("Your movement has been restless for a while", ["Stand up and stretch for one minute"])
if not a.no_claude:
    if not key:
        print("Claude: skipped (no AURA_ANTHROPIC_KEY). Run:  source ~/.bashrc")
    else:
        for auth in ("x-api-key", "bearer"):
            try:
                line = voice.claude_line("Your movement has been restless for a while",
                                         ["Stand up and stretch for one minute"],
                                         "spatial_restlessness", 2, key=key, auth=auth)
                print(f"Claude: OK with auth={auth}")
                if auth != "x-api-key":
                    print("  -> add this to ~/.config/aura-m1/anthropic.env:  export AURA_ANTHROPIC_AUTH=bearer")
                break
            except Exception as e:
                print(f"Claude: auth={auth} failed ({type(e).__name__} {getattr(e, 'code', '')})")
        else:
            print("Claude: both auth styles failed -> built-in sentences will be used")
print(f"line: {line}")
if spk:
    print("speaking...", "done" if voice._default_speak(line) else "failed")

# --- full master prompt, end to end ------------------------------------------
print()
mp = voice.load_master_prompt()
print(f"master prompt: {len(mp)} chars ({'full spec' if len(mp) > 20000 else 'SHORT FALLBACK - voice_prompt.txt missing'})")
if key and not a.no_claude:
    import json as _j
    payload = {"interaction_mode": "proactive_nudge", "problem": "spatial_restlessness",
               "workload_state_1_to_5": 2, "why": "You've been shifting position a lot.",
               "suggested_action": "Take a 60-second stretch break", "episode_number": 1,
               "minutes_into_session": 42, "time_of_day": voice.time_of_day(),
               "recent_interventions": [], "quiet_mode": False, "available_integrations": []}
    auth = os.environ.get("AURA_ANTHROPIC_AUTH", "x-api-key")
    try:
        raw = voice.call_claude(mp, [{"role": "user", "content": _j.dumps(payload)}], key=key, auth=auth)
        v = voice.validate_nudge(raw)
        print(f"nudge via master prompt: {raw!r}")
        print("validator:", "PASS" if v and v != "SKIP" else ("SKIP" if v == "SKIP" else "REJECTED (would use built-in line)"))
        payload["quiet_mode"] = True
        raw2 = voice.call_claude(mp, [{"role": "user", "content": _j.dumps(payload)}], key=key, auth=auth)
        print(f"quiet-mode check (expect SKIP): {raw2!r}")
    except Exception as e:
        print(f"master prompt call failed ({type(e).__name__} {getattr(e, 'code', '')})")
