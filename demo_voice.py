#!/usr/bin/env python3
"""
Hear Aura-M1's voice coaching without the camera or any alert.
Feeds each problem type through the real VoiceCoach (Claude line + your speaker).
Run:  AURA_VOICE=1 python demo_voice.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("AURA_VOICE", "1")
from aura_sdk.recommend.rules import (ACTION_LIBRARY, WHY_TEMPLATES, ProblemClass,
                                      RecommendationEvent)
from aura_sdk import envfile; envfile.load_env_file()
from aura_sdk.voice import VoiceCoach

coach = VoiceCoach(cooldown_s=0)
for i, pc in enumerate(ProblemClass, 1):
    ev = RecommendationEvent(frame_idx=i, problem_class=pc, state=2, q_hci=0.3,
                             why=WHY_TEMPLATES[pc], suggested_actions=list(ACTION_LIBRARY[pc]),
                             rationale="demo", sustained_frames=30, episode_count=i)
    print(f"\n[{pc.value}]")
    coach.announce(ev)
    time.sleep(0.5)
    while not coach._q.empty():
        time.sleep(0.1)
    time.sleep(6)          # let Claude + speech finish before the next one
