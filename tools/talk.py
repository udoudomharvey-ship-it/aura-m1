#!/usr/bin/env python3
"""
Talk to Aura-M1 by typing; it answers out loud (Piper / espeak) using the master prompt.
Run:  python tools/talk.py
Commands:  /status  /quiet  /loud  /mute  /unmute  /help  /quit
What is sent to the Anthropic API: what you type, the last few turns, time of day, minutes into the
session and, if the live app is running, the workload number (1-5), whether an alert is active and the
last things Aura said. Nothing from the camera, screen or files. /status shows exactly the live part.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from aura_sdk import envfile; envfile.load_env_file()
from aura_sdk.voice import VoiceCoach
from aura_sdk import voice_context

coach = VoiceCoach(enabled=True, context_fn=voice_context.read_context)
if not coach._key:
    print("No AURA_ANTHROPIC_KEY loaded (run: source ~/.bashrc). Replies will be a placeholder.")
print("Aura-M1 is listening (typing). /help for commands, /quit to leave.\n")
while True:
    try:
        text = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break
    if not text:
        continue
    cmd = text.lower()
    if cmd in ("/quit", "/exit"):
        break
    if cmd == "/help":
        print("/status shows what live context Aura can see; /quiet silences spoken nudges; /loud brings them back; /mute stops speech; /unmute resumes; /quit exits.")
    elif cmd == "/status":
        c = voice_context.read_context()
        print(json.dumps(c, indent=2) if c else "no live reading and no recent nudges found")
        print("quiet mode:", "ON" if coach.is_quiet() else "off")
    elif cmd == "/quiet":
        coach.set_quiet(True); print("quiet mode on for all of Aura-M1, including the live app. You can still talk to me.")
    elif cmd == "/loud":
        coach.set_quiet(False); print("quiet mode off")
    elif cmd == "/mute":
        coach.muted = True; print("speech muted (text replies continue)")
    elif cmd == "/unmute":
        coach.muted = False; print("speech on")
    else:
        reply = coach.converse(text)
        print(f"aura> {reply or '(no reply, check your connection)'}\n")
