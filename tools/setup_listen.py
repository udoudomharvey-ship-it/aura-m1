#!/usr/bin/env python3
"""Downloads the small offline English speech model (about 40 MB) used by "Hey Aura". Safe to re-run."""
import os, sys, urllib.request, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from aura_sdk import platform_compat as compat

NAME = "vosk-model-small-en-us-0.15"
URL = f"https://alphacephei.com/vosk/models/{NAME}.zip"
dest = os.path.dirname(compat.vosk_model_dir())
os.makedirs(dest, exist_ok=True)
if os.path.isdir(os.path.join(dest, NAME)):
    print("Speech model already installed."); sys.exit(0)
zpath = os.path.join(dest, "m.zip")
print("Downloading the speech model (about 40 MB) ...")
try:
    urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
finally:
    if os.path.exists(zpath):
        os.remove(zpath)
print("Ready. Run:  python tools/listen.py   (Windows: aura.bat listen)  and say 'Hey Aura'.")
