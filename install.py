#!/usr/bin/env python3
"""Sets up Aura-M1 in a local virtual environment (.venv). Works on Windows, macOS and Linux.
  python install.py              demo + tests (numpy only)
  python install.py --live       also webcam support (opencv, mediapipe)
  python install.py --voice      also the Piper neural voice
  python install.py --listen     also hands-free "Hey Aura" (offline speech model, about 40 MB)"""
import os, subprocess, sys, venv

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")


def venv_python():
    return os.path.join(VENV, "Scripts" if sys.platform.startswith("win") else "bin",
                        "python.exe" if sys.platform.startswith("win") else "python")


def pip(*pkgs):
    return subprocess.call([venv_python(), "-m", "pip", "install", "--quiet", *pkgs])


def main(argv):
    if sys.version_info < (3, 9):
        print("Python 3.9 or newer is required."); return 1
    if not os.path.isfile(venv_python()):
        print("Creating .venv ...")
        try:
            venv.EnvBuilder(with_pip=True).create(VENV)
        except Exception as e:
            print(f"Could not create a virtual environment ({e}). On Ubuntu: sudo apt install python3-venv"); return 1
    pip("--upgrade", "pip")
    if pip("numpy") != 0:
        print("Could not install numpy. Check your internet connection."); return 1
    if "--live" in argv and pip("opencv-python", "mediapipe") != 0:
        print("Webcam packages failed to install (mediapipe needs Python 3.9 to 3.12 on some systems).")
    if "--voice" in argv:
        pip("piper-tts")
    if "--listen" in argv:
        if pip("vosk", "sounddevice", "SpeechRecognition") != 0:
            print("Listening packages failed to install.")
        else:
            subprocess.call([venv_python(), os.path.join(ROOT, "tools", "setup_listen.py")])
    run = "aura.bat" if sys.platform.startswith("win") else "./aura"
    print(f"Installed. Try:  {run} demo     (and {run} test)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
