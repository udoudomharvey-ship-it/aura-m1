# Aura-M1 MVP prototype

Webcam-only focus-decline detector. Everything runs on your machine.

## Run
    conda activate base          # the env with numpy, opencv, mediapipe
    ./start_aura.sh              # dashboard at http://localhost:8765/dashboard.html
    ./start_aura.sh --voice      # same, with spoken coaching

## Voice coaching
- Off by default. `--voice` (or `AURA_VOICE=1`) turns it on; `AURA_VOICE_MUTE=1` silences it.
- When an alert has been sustained, Aura-M1 speaks one short sentence, at most once per 90 seconds.
- Claude writes the sentence from the problem label and suggested action only. No video, no body
  data, no identity is sent. Without a key or network, a built-in sentence is used.
- Speech comes from a local program (`sudo apt install espeak-ng`).
- Check the setup any time: `python tools/voice_selftest.py`

## Privacy
- Video is processed in memory on your machine and is not saved. The dashboard binds to 127.0.0.1 only.
- Escalation emails are a dry run unless `AURA_SEND_REAL_ESCALATIONS` is set.
- The only outbound call is the optional voice-line request to the Anthropic API.

## Tests
    python -m unittest discover -s aura_sdk_tests -t .      # 134 tests

## Known limits (read before showing anyone)
- Detection is NOT validated on real people yet. The validation run needs labelled focused and
  distracted sessions across several days; until then, quote no accuracy figure.
- Alerts are voice and dashboard only. There is no screen-edge glow, grace-period filter, battery
  mode or CPU monitor yet (all requested in interviews).
- Rust core and firmware are untested here and do not apply the per-feature rescaling.
- Single user, single camera, Linux tested.
