"""
Small adapters so Aura runs on Windows and macOS as well as Linux. Nothing here talks to the network.

  config_dir()  where the quiet-mode file and anthropic.env live
  data_dir()    where the offline speech model is stored
  speak_windows(text) / play_wav(path)   speech and sound without extra installs on Windows
  read_clipboard()                       the clipboard, only ever called when the user asks
"""
import os, shutil, subprocess, sys
from typing import Optional


def is_windows() -> bool:
    return sys.platform.startswith("win")


def is_mac() -> bool:
    return sys.platform == "darwin"


def config_dir() -> str:
    if is_windows():
        return os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "aura-m1")
    return os.path.expanduser("~/.config/aura-m1")


def data_dir() -> str:
    if is_windows():
        return os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "aura-m1")
    return os.path.expanduser("~/.local/share")


def vosk_model_dir() -> str:
    return os.path.join(data_dir(), "vosk", "vosk-model-small-en-us-0.15")


def powershell() -> Optional[str]:
    return shutil.which("powershell") or shutil.which("pwsh")


# ---- speech ------------------------------------------------------------------------------------
_SAPI = ("[Console]::InputEncoding=[Text.Encoding]::UTF8;"
         "Add-Type -AssemblyName System.Speech;"
         "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
         "try{$s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female)}catch{};"
         "$s.Rate=0;$s.Speak([Console]::In.ReadToEnd())")


def sapi_cmd() -> Optional[list]:
    ps = powershell()
    return [ps, "-NoProfile", "-NonInteractive", "-Command", _SAPI] if ps else None


def speak_windows(text: str, timeout: float = 40.0) -> bool:
    cmd = sapi_cmd()
    if not cmd:
        return False
    try:
        subprocess.run(cmd, input=text.encode("utf-8"), timeout=timeout, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _unix_player() -> Optional[list]:
    for cmd in (["paplay"], ["aplay", "-q"], ["afplay"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            return cmd
    return None


def can_play() -> bool:
    return True if is_windows() else _unix_player() is not None


def play_wav(path: str, timeout: float = 60.0) -> bool:
    try:
        if is_windows():
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)      # blocks until finished
            return True
        player = _unix_player()
        if not player:
            return False
        subprocess.run(player + [path], timeout=timeout, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


# ---- clipboard ---------------------------------------------------------------------------------
def clipboard_commands() -> list:
    if is_windows():
        ps = powershell()
        return [[ps, "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard -Raw"]] if ps else []
    if is_mac():
        return [["pbpaste"]]
    return [["wl-paste", "-n"], ["xclip", "-selection", "clipboard", "-o"], ["xsel", "-b", "-o"]]


def read_clipboard() -> Optional[str]:
    for cmd in clipboard_commands():
        if cmd and shutil.which(cmd[0]):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if out.returncode == 0 and out.stdout.strip():
                    return out.stdout
            except Exception:
                pass
    return None
