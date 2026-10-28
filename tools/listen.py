#!/usr/bin/env python3
"""Hands-free Aura. Say "Hey Aura". Local speech recognition; see aura_sdk/listen.py for exactly what is kept or sent.
Run:  python tools/listen.py        (needs: bash install_listen.sh once)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from aura_sdk import envfile; envfile.load_env_file()
from aura_sdk.voice import VoiceCoach
from aura_sdk import voice_context
from aura_sdk.listen import Listener

coach = VoiceCoach(enabled=True, context_fn=voice_context.read_context)
if not coach._key:
    print("No AURA_ANTHROPIC_KEY loaded: replies will be a placeholder.")
Listener(coach).run()
