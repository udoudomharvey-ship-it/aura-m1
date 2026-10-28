#!/usr/bin/env bash
# Creates a local virtual environment and installs what Aura-M1 needs.
#   ./install.sh            demo + tests (numpy only)
#   ./install.sh --live     also webcam support (opencv, mediapipe)
#   ./install.sh --voice    also the Piper neural voice
#   ./install.sh --listen   also hands-free "Hey Aura" (offline speech recognition, ~40 MB model)
cd "$(dirname "$0")" || exit 1
command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
python3 -m venv .venv || { echo "On Ubuntu run: sudo apt install python3-venv"; exit 1; }
. .venv/bin/activate && pip install --quiet --upgrade pip && pip install --quiet numpy
for a in "$@"; do
  [ "$a" = "--live" ] && pip install --quiet opencv-python mediapipe
  [ "$a" = "--voice" ] && pip install --quiet piper-tts
  [ "$a" = "--listen" ] && { bash ./install_listen.sh || echo "listen setup incomplete: see message above"; }
done
echo "Installed. Try:  ./aura demo     (and ./aura test)"
echo "For spoken output on Linux:  sudo apt install espeak-ng"
