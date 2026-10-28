# Aura-M1

**A private way to notice when your focus is slipping, using only the webcam you already own.**
Built by Harvey Udoudom, GreyMatter Inc. (iDICE Startup Bridge Founders Lab).

**Try it in your browser:** open the simulation page (GitHub Pages, from the `docs/` folder of this repository). It replays sessions recorded from the real pipeline. It uses no camera and no microphone.

## What it does

A webcam feed goes through MediaPipe pose tracking. Joint movement is turned into three entropy features (HEMM). A small 1-bit neural network scores interaction quality, a threshold detector decides when a decline is sustained, and a recommender picks one gentle reset. Only short text, never video or body data, goes to an AI model for wording. Optional voice coaching and a hands-free "Hey Aura" mode are included.

## Status (honest)

- Working prototype: runs end to end, 202 automated tests pass.
- **Detection accuracy is not yet measured.** No accuracy claim is made. Held-out validation on labelled sessions is the next step.
- The simulation page and `./aura demo` use a **placeholder model** and scripted scenarios. Only "cognitive overload" alerts reliably; the other two scenarios are marked as known limitations in the dashboard.
- The Windows version is written but has not been tested on a real Windows machine.
- 10 customer interviews done, 0 pilot users so far.

## Run it yourself

Download the install package from the repository's **Releases** page (or clone this repo), then, on Linux or macOS:

```bash
cd aura-m1
./install.sh            # add --live for the webcam, --listen for "Hey Aura"
./aura demo             # scripted demo, no camera needed
./aura live --voice     # real webcam (detection not validated on people yet)
```

Open http://localhost:8765/aura.html. On Windows use `aura.bat` and see `WINDOWS_COMMANDS.md`. More detail: `START_HERE.md`.

Voice coaching needs your own Anthropic API key in `~/.config/aura-m1/anthropic.env`. Without a key, Aura uses built-in lines.

## Privacy

Video is never saved or uploaded; frames stay in memory. The wake word is matched on your computer. The optional online speech mode (`AURA_STT=online`) sends each spoken request to Google for transcription, and is off by default.

## Layout

| Path | What it is |
| --- | --- |
| `aura_sdk/` | Pipeline: pose, entropy features, BNN, recommender, voice |
| `aura_core/`, `aura_firmware/` | Rust core and firmware sketch |
| `aura.html`, `dashboard_server.py` | Dashboard and local server |
| `docs/` | Static simulation page for GitHub Pages |
| `aura_sdk_tests/`, `tests/` | Automated tests |

Contact: Harvey Udoudom, GreyMatter Inc.
