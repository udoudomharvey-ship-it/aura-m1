# Aura-M1 demo package

Aura-M1 reads posture and movement from a webcam to spot when focus is sliding, then suggests a
short, quiet reset. This is an early prototype.

## Quick start (about 2 minutes)
    ./install.sh                # once  (use ./install.sh --live --voice --listen for everything)
    ./aura demo                 # then open http://localhost:8765/dashboard.html
    ./aura demo --voice         # same, with spoken nudges (Linux: sudo apt install espeak-ng)

## Windows (native, no WSL needed)
Install Python 3.10 to 3.12 from python.org (tick "Add python.exe to PATH"), then in the unzipped folder:
    install.bat                 # once   (install.bat --live --voice --listen for everything)
    aura.bat demo               # opens http://localhost:8765/aura.html
    aura.bat live               # webcam; Windows asks permission to use the camera and microphone
Spoken output uses Windows' built-in voice, with no extra installs. Your API key goes in
`%APPDATA%\aura-m1\anthropic.env` as a line: `set AURA_ANTHROPIC_KEY=your-key`. The same `python aura.py ...` works on macOS.
Windows support is new and has not been tested on a real Windows machine yet.
Full command list and troubleshooting: WINDOWS_COMMANDS.md

## Needs
Python 3.10+ with `python3 -m venv` (Ubuntu: `sudo apt install python3-venv`). Linux is tested; macOS/Windows are not.
Optional: `espeak-ng` (basic voice), `alsa-utils` + `xclip` (hands-free listening), a webcam (live mode).

## What each mode is
- **demo**: scripted scenarios sent through the real pipeline with placeholder model weights, so you
  can see alerts, recommendations and escalation without a camera. It does NOT read a real person.
- **live**: your webcam, processed on your machine. Needs `./install.sh --live`. Detection is not
  validated on real people yet; treat results as a prototype, not a measurement.
- **listen**: hands-free. Say "Hey Aura". Speech is recognised offline on your machine (Vosk), never saved or sent;
  only your words as text go to the chat. Aura can set timers, go quiet, and read your clipboard once when you say
  "look at this error". It cannot run commands or touch files.
- **voice / chat**: spoken nudges, and typed conversation with spoken replies. These use an
  Anthropic API key (`export AURA_ANTHROPIC_KEY=...` in `~/.config/aura-m1/anthropic.env`).
  With no key, built-in sentences are used. Set `AURA_VOICE_CLAUDE=0` to send nothing anywhere.

## Privacy
Video stays in memory on your machine and is never saved or uploaded. The dashboard listens on
127.0.0.1 only. Voice requests to the API carry the problem label, a 1-5 workload number, time of
day and recent spoken lines, plus what you type in chat. Never video, body data or screen content.

## Known limits
- Detection is unvalidated; there is no accuracy figure.
- The model is calibrated to one person (the founder). Others will see rough results.
- Alerts are voice and dashboard (aura.html has an optional edge glow). No grace periods or battery mode yet.
- Hands-free listening is new and lightly tested; wake-word accuracy depends on your microphone and accent.
- Linux tested. Windows and macOS support is new and untested on real machines. Escalation emails are a dry run unless AURA_SEND_REAL_ESCALATIONS is set.
