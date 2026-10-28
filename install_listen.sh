#!/usr/bin/env bash
# One-time setup for hands-free Aura: offline speech recognition (Vosk) + a small English model (~40 MB).
set -e
cd "$(dirname "$0")"
python3 -m pip install --quiet vosk SpeechRecognition
command -v arecord >/dev/null || echo "Tip: sudo apt install alsa-utils   (for the microphone)"
command -v xclip >/dev/null || command -v wl-paste >/dev/null || echo "Tip: sudo apt install xclip   (only needed for 'look at this error')"
python3 tools/setup_listen.py
