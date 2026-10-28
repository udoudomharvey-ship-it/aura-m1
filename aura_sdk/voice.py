"""
aura_sdk/voice.py
-----------------
Aura-M1's voice: spoken nudges and short conversations, structured around the
master system prompt in aura_sdk/voice_prompt.txt.

Division of labour (as the prompt requires)
  * The HOST (this file) enforces everything that must be deterministic:
    quiet mode, cooldowns, declined/completed handling, output validation,
    speech scheduling. Quiet mode never even reaches the API.
  * Claude writes the words, as a JSON request -> one reply, in one of four
    modes: proactive_nudge, conversational_assistance, active_problem_solving,
    authorised_action. No integrations exist yet, so authorised_action is told
    that none are available.
  * Speech is produced by a local program (Piper or espeak-ng). The API
    makes text, not audio.

What leaves the machine (only when a key is set and AURA_VOICE_CLAUDE != 0)
  * Nudges: problem label, workload number, the reason and suggested action,
    time of day, minutes into session, the last few spoken lines.
  * Conversation: the same, plus what the user types to Aura and the last
    few turns. Never video, joint data, entropy values, screen content or files.

Switches (environment)
  AURA_VOICE=1              spoken nudges on (default: off)
  AURA_VOICE_CLAUDE=0       never call the API; built-in sentences only
  AURA_VOICE_MUTE=1         silence without changing anything else
  ~/.config/aura-m1/quiet   if this file exists, quiet mode is on (touch / rm)
  AURA_ANTHROPIC_KEY        the API key (never logged or printed)
  AURA_ANTHROPIC_AUTH       x-api-key (default) or bearer
  AURA_VOICE_MODEL          default claude-haiku-5-5
  AURA_VOICE_COOLDOWN_S     base seconds between spoken nudges (default 90)
  AURA_VOICE_ENGINE         espeak (default) or piper
  AURA_PIPER_MODEL          path to a Piper .onnx voice file
  AURA_ESPEAK_ARGS          espeak-ng options, default '-v en-us+f3 -s 150 -p 55'
"""
from __future__ import annotations

import collections
import datetime
import json
import os
import queue
import re
import shutil
import subprocess
from . import platform_compat as _compat
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

from aura_sdk import voice_context

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-5-5"
MAX_SPOKEN_CHARS = 220

SYSTEM_PROMPT = (
    "You write ONE short spoken coaching sentence for a focus and posture "
    "assistant. Maximum 22 words. Calm, friendly, second person, plain words. "
    "Suggest the given action. Do not diagnose, do not make medical claims, "
    "do not mention cameras, sensors, scores or numbers. Output only the sentence."
)


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def fallback_line(why: str, actions) -> str:
    """Built-in sentence, no network needed."""
    action = (actions[0] if actions else "").strip()
    why = (why or "").strip().rstrip(".")
    if why and action:
        return f"{why}. {action}"
    return action or why or "Time for a short reset."


def clean_line(text: str, limit: int = MAX_SPOKEN_CHARS) -> str:
    """Make model output safe to speak: one line, bounded length."""
    t = " ".join(str(text or "").split())
    t = t.strip("\"'` ")
    if len(t) > limit:
        t = t[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "."
    return t


def _espeak_cmd() -> Optional[list]:
    if not shutil.which("espeak-ng"):
        return None
    args = os.environ.get("AURA_ESPEAK_ARGS", "-v en-us+f3 -s 150 -p 55").split()
    return ["espeak-ng"] + args


def detect_speaker() -> Optional[list]:
    """Return a command prefix for a local TTS program, or None."""
    if _compat.is_windows():
        return _compat.sapi_cmd()       # Windows' built-in speech, text is sent on stdin
    cmd = _espeak_cmd()
    if cmd:
        return cmd
    for cmd in (["spd-say", "-w"], ["say"]):
        if shutil.which(cmd[0]):
            return cmd
    return None


def _player() -> Optional[list]:
    for cmd in (["paplay"], ["aplay", "-q"], ["afplay"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            return cmd
    return None


def _piper_speak(text: str) -> bool:
    """Neural voice via Piper. Needs: pip install piper-tts, a .onnx voice, AURA_PIPER_MODEL."""
    model = os.environ.get("AURA_PIPER_MODEL", "").strip()
    if not model or not os.path.isfile(model) or not _compat.can_play():
        return False
    import tempfile
    exe = shutil.which("piper")
    cmd = [exe] if exe else [sys.executable, "-m", "piper"]
    fd, wav = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    try:
        r = subprocess.run(cmd + ["--model", model, "--output_file", wav], input=text.encode(),
                           timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0 or os.path.getsize(wav) < 100:
            return False
        return _compat.play_wav(wav)
    except Exception:
        return False
    finally:
        try: os.remove(wav)
        except OSError: pass


def _default_speak(text: str) -> bool:
    if os.environ.get("AURA_VOICE_ENGINE", "espeak").strip().lower() == "piper":
        if _piper_speak(text):
            return True            # else fall through to espeak
    if _compat.is_windows():
        return _compat.speak_windows(text)
    cmd = detect_speaker()
    if not cmd:
        return False
    try:
        subprocess.run(cmd + [text], timeout=30, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def claude_line(why: str, actions, problem: str, state: int, *,
                key: str, model: str = DEFAULT_MODEL, auth: str = "x-api-key",
                timeout: float = 5.0,
                opener: Callable = urllib.request.urlopen) -> str:
    """Ask Claude for one spoken sentence. Raises on any failure."""
    user = json.dumps({
        "problem": problem,
        "workload_state_1_to_5": state,
        "why": why,
        "suggested_action": (actions[0] if actions else ""),
    })
    body = json.dumps({
        "model": model, "max_tokens": 80, "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user}],
    }).encode()
    headers = {"content-type": "application/json",
               "anthropic-version": "2023-06-01"}
    if auth == "bearer":
        headers["authorization"] = f"Bearer {key}"
    else:
        headers["x-api-key"] = key
    req = urllib.request.Request(API_URL, data=body, headers=headers, method="POST")
    with opener(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    line = clean_line("".join(parts))
    if not line:
        raise ValueError("empty reply")
    return line


# ── master prompt, validation, routing ───────────────────────────────────────

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_prompt.txt")
QUIET_FILE = os.path.join(_compat.config_dir(), "quiet")
MAX_NUDGE_WORDS = 22
FALLBACK_NUDGE = "Time for a short reset."


def load_master_prompt() -> str:
    """The full master prompt. Falls back to the short legacy prompt if missing."""
    try:
        with open(PROMPT_PATH, encoding="utf-8") as f:
            t = f.read().strip()
        if len(t) > 500:
            return t
    except OSError:
        pass
    return SYSTEM_PROMPT


def time_of_day(now: Optional[datetime.datetime] = None) -> str:
    h = (now or datetime.datetime.now()).hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 21:
        return "evening"
    return "late night"


_ACTION_WORDS = re.compile(r"\b(remind(er)?|set (a |an )?(timer|alarm)|create (a )?(task|reminder)|open|schedule|add (it )?to)\b", re.I)
_SOLVE_WORDS = re.compile(r"\b(help|organi[sz]e|plan|stuck|struggl\w+|how (can|do|should)|where (do|should)|break (it|this) down|workflow|next step|what should i)\b", re.I)


def route_mode(text: str) -> str:
    """Host-side mode choice for something the user said."""
    if _ACTION_WORDS.search(text or ""):
        return "authorised_action"
    if _SOLVE_WORDS.search(text or ""):
        return "active_problem_solving"
    return "conversational_assistance"


def validate_nudge(text: str):
    """Return 'SKIP', the cleaned sentence, or None if it breaks the nudge rules."""
    t = clean_line(text)
    if t.upper() == "SKIP":
        return "SKIP"
    if not t or any(c in t for c in "*#`_[]{}<>|"):
        return None
    if len([p for p in re.split(r"(?<=[.!?])\s+", t) if p]) != 1:
        return None
    if len(t.split()) > MAX_NUDGE_WORDS:
        return None
    return t


def call_claude(system_text: str, messages: list, *, key: str, model: str = DEFAULT_MODEL,
                auth: str = "x-api-key", max_tokens: int = 80, timeout: float = 15.0,
                opener: Callable = urllib.request.urlopen) -> str:
    """One Messages API call. The long system prompt is marked cacheable so
    repeat calls are cheaper and faster. Raises on any failure."""
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if auth == "bearer":
        headers["authorization"] = f"Bearer {key}"
    else:
        headers["x-api-key"] = key

    def _send(extra: dict) -> dict:
        body = json.dumps({
            "model": model, "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            "messages": messages, **extra,
        }).encode()
        req = urllib.request.Request(API_URL, data=body, headers=headers, method="POST")
        with opener(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    # Short spoken replies do not need internal reasoning. With it on, the model can use
    # the whole token budget thinking and return no text at all.
    try:
        data = _send({"thinking": {"type": "disabled"}})
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        data = _send({})        # this model rejected the switch: retry without it
    out = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
    if not out:
        kinds = [b.get("type") for b in data.get("content", [])]
        raise ValueError(f"empty reply (stop_reason={data.get('stop_reason')}, blocks={kinds})")
    return out


# ── the coach ────────────────────────────────────────────────────────────────

class VoiceCoach:
    """
    Non-blocking for the camera loop: announce() returns immediately.
    converse() is synchronous (used by tools/talk.py).
    """

    MAX_TURNS = 8            # conversation turns kept for continuity
    DONE_SUPPRESS_S = 600    # after "done", leave that problem alone this long
    MAX_DECLINE_FACTOR = 8   # declined nudges stretch the cooldown up to 8x

    def __init__(self, enabled: Optional[bool] = None, use_claude: Optional[bool] = None,
                 cooldown_s: Optional[float] = None, log: Callable[[str], None] = print,
                 speak: Optional[Callable[[str], bool]] = None,
                 call_fn: Optional[Callable[..., str]] = None,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], datetime.datetime] = datetime.datetime.now,
                 quiet_file: Optional[str] = None,
                 context_fn: Optional[Callable[[], dict]] = None,
                 voice_state_path: Optional[str] = None):
        env = os.environ
        self.enabled = _truthy(env.get("AURA_VOICE")) if enabled is None else enabled
        self.muted = _truthy(env.get("AURA_VOICE_MUTE"))
        self.use_claude = (env.get("AURA_VOICE_CLAUDE", "1") != "0") if use_claude is None else use_claude
        self.cooldown_s = float(env.get("AURA_VOICE_COOLDOWN_S", "90")) if cooldown_s is None else cooldown_s
        self._key = env.get("AURA_ANTHROPIC_KEY", "").strip()
        self._auth = env.get("AURA_ANTHROPIC_AUTH", "x-api-key").strip().lower()
        self._model = env.get("AURA_VOICE_MODEL", DEFAULT_MODEL).strip()
        self._log, self._speak = log, speak or _default_speak
        self._call, self._clock, self._wall = call_fn or call_claude, clock, wall
        self._quiet_file = QUIET_FILE if quiet_file is None else quiet_file
        self._context_fn = context_fn
        self._voice_state_path = voice_state_path
        self._system = load_master_prompt()
        self.quiet_flag = False
        self._start = clock()
        self._last_t: Optional[float] = None
        self.episode = 0
        self.recent: "collections.deque" = collections.deque(maxlen=5)
        self.history: list = []
        self.declines: dict = collections.defaultdict(int)
        self.done_until: dict = {}
        self.last_problem: Optional[str] = None
        self.last_line = ""
        self.last_source = ""
        self._q: "queue.Queue" = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    # state the user (or a hotkey / talk.py) can change
    def is_quiet(self) -> bool:
        return self.quiet_flag or bool(self._quiet_file and os.path.exists(self._quiet_file))

    def set_quiet(self, on: bool):
        """Quiet mode shared with every Aura process (the live app checks the same file)."""
        self.quiet_flag = False
        if not self._quiet_file:
            self.quiet_flag = on
            return
        try:
            if on:
                os.makedirs(os.path.dirname(self._quiet_file), exist_ok=True)
                open(self._quiet_file, "a").close()
            elif os.path.exists(self._quiet_file):
                os.remove(self._quiet_file)
        except OSError:
            self.quiet_flag = on

    def mark_declined(self):
        if self.last_problem:
            self.declines[self.last_problem] = min(self.declines[self.last_problem] + 1, 3)

    def mark_done(self):
        if self.last_problem:
            self.done_until[self.last_problem] = self._clock() + self.DONE_SUPPRESS_S

    def _minutes(self) -> int:
        return int((self._clock() - self._start) // 60)

    def _base_context(self) -> dict:
        ctx = {"minutes_into_session": self._minutes(),
               "time_of_day": time_of_day(self._wall()),
               "recent_interventions": list(self.recent),
               "available_integrations": []}
        if self._context_fn:
            try:
                extra = self._context_fn() or {}
            except Exception:
                extra = {}
            for k in ("workload_state_1_to_5", "recent_interventions", "available_context"):
                if extra.get(k):
                    ctx[k] = extra[k]
        return ctx

    # proactive nudges ---------------------------------------------------------
    def announce(self, rec) -> bool:
        """Queue a spoken nudge for a RecommendationEvent. True if accepted."""
        if not self.enabled or self.muted or rec is None:
            return False
        if self.is_quiet():                       # host-enforced, never reaches the API
            return False
        problem = rec.problem_class.value
        now = self._clock()
        if self.done_until.get(problem, 0) > now:
            return False
        factor = min(2 ** self.declines[problem], self.MAX_DECLINE_FACTOR)
        if self._last_t is not None and now - self._last_t < self.cooldown_s * factor:
            return False
        try:
            self._q.put_nowait(rec)
        except queue.Full:
            return False
        self._last_t = now
        return True

    def build_nudge_payload(self, rec) -> dict:
        p = {"interaction_mode": "proactive_nudge",
             "problem": rec.problem_class.value,
             "workload_state_1_to_5": rec.state,
             "why": rec.why,
             "suggested_action": (rec.suggested_actions[0] if rec.suggested_actions else ""),
             "episode_number": self.episode + 1,
             "quiet_mode": False}
        p.update(self._base_context())
        return p

    def _worker(self):
        while True:
            rec = self._q.get()
            try:
                self._handle(rec)
            except Exception as e:
                self._log(f"voice: error {type(e).__name__}")

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", str(text).lower()).strip()

    def _is_repeat(self, line: str) -> bool:
        n = self._norm(line)
        return any(n == self._norm(r) for r in self.recent)

    def _ask(self, payload: dict):
        """One model call for a nudge. Returns (validated_line_or_SKIP_or_None, raw_reply)."""
        raw = self._call(self._system, [{"role": "user", "content": json.dumps(payload)}],
                         key=self._key, model=self._model, auth=self._auth, max_tokens=300)
        return validate_nudge(raw), raw

    def _handle(self, rec):
        payload = self.build_nudge_payload(rec)
        built_in = (fallback_line(rec.why, rec.suggested_actions)
                    if payload["suggested_action"] else FALLBACK_NUDGE)
        line, source = built_in, "built-in"
        if self.use_claude and self._key and payload["suggested_action"]:
            try:
                checked, raw = self._ask(payload)
                if checked == "SKIP":
                    self._log("voice: model chose silence (SKIP)")
                    return
                if checked and self._is_repeat(checked):
                    retry = dict(payload, avoid_repeating="Do not reuse the wording of any line in recent_interventions.")
                    again, raw = self._ask(retry)
                    if again == "SKIP":
                        self._log("voice: model chose silence (SKIP)")
                        return
                    checked = again if (again and not self._is_repeat(again)) else None
                    if checked is None:
                        self._log(f"voice: reply repeated an earlier line ({raw[:140]!r}), using built-in line")
                elif checked is None:
                    self._log(f"voice: reply broke the nudge rules ({raw[:140]!r}), using built-in line")
                if checked:
                    line, source = checked, "claude"
            except Exception as e:   # key is never in the message
                detail = str(e)[:120] if isinstance(e, ValueError) else getattr(e, "code", "")
                self._log(f"voice: Claude unavailable ({type(e).__name__} {detail}), using built-in line")
        self.episode += 1
        self.last_problem = payload["problem"]
        self.recent.append(line)
        self.last_line, self.last_source = line, source
        voice_context.write_voice_state(list(self.recent), self.last_problem, self.episode,
                                        path=self._voice_state_path)
        ok = self._speak(line)
        self._log(f"voice[{source}]{'' if ok else ' (no speaker)'}: {line}")

    # conversation -------------------------------------------------------------
    def converse(self, user_text: str, speak: bool = True, extra: Optional[dict] = None) -> str:
        """Answer something the user said. Returns the reply text ('' on failure)."""
        user_text = (user_text or "").strip()
        if not user_text:
            return ""
        mode = route_mode(user_text)
        payload = {"interaction_mode": mode, "user_message": user_text, "quiet_mode": False}
        payload.update(self._base_context())
        if extra:
            payload.update(extra)
        msgs = list(self.history) + [{"role": "user", "content": json.dumps(payload)}]
        if not (self.use_claude and self._key):
            reply = "I can't reach my thinking service right now. Is there something small we can sort out together?"
            source = "built-in"
        else:
            try:
                reply = clean_line(self._call(self._system, msgs, key=self._key, model=self._model,
                                              auth=self._auth, max_tokens=600, timeout=20.0), limit=700)
                source = "claude"
            except Exception as e:
                detail = str(e)[:120] if isinstance(e, ValueError) else getattr(e, "code", "")
                self._log(f"voice: Claude unavailable ({type(e).__name__} {detail})")
                return ""
        if reply.upper() == "SKIP":
            return ""
        self.history.append({"role": "user", "content": json.dumps(payload)})
        self.history.append({"role": "assistant", "content": reply})
        del self.history[:-2 * self.MAX_TURNS]
        while self.history and self.history[0]["role"] != "user":
            self.history.pop(0)
        self.last_line, self.last_source = reply, source
        if speak and self.enabled and not self.muted:
            self._speak(reply)
        return reply
