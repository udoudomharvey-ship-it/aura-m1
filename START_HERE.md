# Aura-M1 demo package

Aura-M1 reads posture and movement from a webcam to spot when focus is sliding, then suggests a
short, quiet reset. This is an early prototype.

## Quick start (about 2 minutes)
    ./install.sh                # once
    ./aura demo                 # then open http://localhost:8765/dashboard.html
    ./aura demo --voice         # same, with spoken nudges (Linux: sudo apt install espeak-ng)

## What each mode is
- **demo**: scripted scenarios sent through the real pipeline with placeholder model weights, so you
  can see alerts, recommendations and escalation without a camera. It does NOT read a real person.
- **live**: your webcam, processed on your machine. Needs `./install.sh --live`. Detection is not
  validated on real people yet; treat results as a prototype, not a measurement.
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
- Alerts are voice and dashboard only. No screen-edge glow, grace periods or battery mode yet.
- Linux tested. Escalation emails are a dry run unless AURA_SEND_REAL_ESCALATIONS is set.
