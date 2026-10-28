#!/usr/bin/env bash
# Aura-M1 one-command launcher.
#   ./start_aura.sh            run with the dashboard, voice off
#   ./start_aura.sh --voice    run with spoken coaching on
# Run it from the conda environment you use for the project (base).
cd "$(dirname "$0")" || exit 1
ENVF="$HOME/.config/aura-m1/anthropic.env"
[ -f "$ENVF" ] && . "$ENVF"            # loads AURA_ANTHROPIC_KEY, never printed
[ "$1" = "--voice" ] && export AURA_VOICE=1

echo "== Aura-M1 preflight =="
python - <<'PY' || { echo "Missing Python packages. Activate your conda base env and retry."; exit 1; }
import importlib, sys
bad = [m for m in ("numpy", "cv2", "mediapipe") if importlib.util.find_spec(m) is None]
if bad: print("missing:", ", ".join(bad)); sys.exit(1)
print("python packages: ok")
PY
ls /dev/video* >/dev/null 2>&1 && echo "camera: found" || echo "camera: NOT FOUND (plug in / allow access)"
[ -n "$AURA_ANTHROPIC_KEY" ] && echo "API key: loaded" || echo "API key: not loaded (built-in voice lines only)"
command -v espeak-ng >/dev/null || command -v spd-say >/dev/null || echo "speaker: none -- sudo apt install espeak-ng"
echo
echo "Dashboard:  http://localhost:8765/dashboard.html"
echo "Labelling:  http://localhost:8765/label_control.html"
echo "Stop with Ctrl+C"
echo
exec python run_live.py
