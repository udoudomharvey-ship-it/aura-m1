# Aura-M1 on Windows: commands to run

Copy each block into **PowerShell** (Start menu, type "PowerShell"). In PowerShell, run the .bat files with `.\` in front, as shown.
Windows support is new and not yet tested on a real Windows machine. If anything fails, copy the red text and send it back.

## 1. One-time setup

Check Python (you need 3.10 to 3.12; 3.12 is the safest for the webcam package):
```powershell
py -0
python --version
```
No Python? Install it (or download from python.org and tick "Add python.exe to PATH"):
```powershell
winget install -e --id Python.Python.3.12
```
Close and reopen PowerShell after installing, then unzip and enter the folder:
```powershell
Expand-Archive -Path "$HOME\Downloads\aura-m1-demo-20261025.zip" -DestinationPath "$HOME\Aura"
cd "$HOME\Aura\aura-m1-demo-20261025"
```

## 2. Install

Basic (demo and tests only):
```powershell
.\install.bat
```
Everything (webcam, voice, hands-free "Hey Aura"):
```powershell
.\install.bat --live --voice --listen
```
If you have several Pythons (for example 3.13 as well as 3.12) the webcam package may fail to install. Pick 3.12 explicitly:
```powershell
py -3.12 install.py --live --voice --listen
```

## 3. Check it works
```powershell
.\aura.bat test
.\aura.bat doctor
```
`test` should end with `OK`. `doctor` checks your speaker and voice.

## 4. Run
```powershell
.\aura.bat demo                  # scripted demo, no camera. Opens http://localhost:8765/aura.html
.\aura.bat demo --voice          # same, with spoken nudges
.\aura.bat live                  # your webcam (needs install --live)
.\aura.bat live --voice
.\aura.bat chat                  # type to Aura, hear replies (needs API key)
.\aura.bat listen                # say "Hey Aura" (needs install --listen and API key)
.\aura.bat voice                 # hear the four spoken nudges
```
Stop anything with **Ctrl+C**. Add `--no-browser` to demo or live if you do not want the page to open by itself.

## 5. API key (for chat, listening and smart voice lines)

Never paste the key into a chat or a screen recording. Create the settings file and open it in Notepad:
```powershell
New-Item -ItemType Directory -Force "$env:APPDATA\aura-m1" | Out-Null
notepad "$env:APPDATA\aura-m1\anthropic.env"
```
In Notepad, put this one line, with your own key, then save:
```
set AURA_ANTHROPIC_KEY=your-key-here
```
Without a key Aura still runs, using built-in sentences. To send nothing anywhere, add the line `set AURA_VOICE_CLAUDE=0`.

## 6. Quiet mode
```powershell
New-Item -ItemType File -Force "$env:APPDATA\aura-m1\quiet" | Out-Null     # quiet on
Remove-Item "$env:APPDATA\aura-m1\quiet"                                   # quiet off
```
(Inside `chat` you can also type `/quiet` and `/loud`, or say "go quiet" to Aura.)

## 7. When something goes wrong

**"python is not recognized"**: reopen PowerShell after installing Python, or use `py` instead of `python`.

**Port 8765 already in use** (an earlier run is still going):
```powershell
netstat -ano | findstr :8765
taskkill /PID <the number at the end of the line> /F
```

**Camera not found or black.** Windows: Settings > Privacy & security > Camera, and allow desktop apps. Close Zoom, Teams or OBS if they hold the camera. Then:
```powershell
.\.venv\Scripts\python.exe diagnose_camera.py
```

**Microphone not heard.** Settings > Privacy & security > Microphone, and allow desktop apps. List your devices:
```powershell
.\.venv\Scripts\python.exe -c "import sounddevice as sd; print(sd.query_devices())"
```

**No sound from Aura.** Check Windows volume and the default output device, then run `.\aura.bat doctor`.

**Webcam package fails to install.** Use Python 3.12: `py -3.12 install.py --live`. Delete the half-made environment first if needed:
```powershell
Remove-Item -Recurse -Force .venv
```

**Speech model will not download** (for "Hey Aura"): check your internet connection, then
```powershell
.\.venv\Scripts\python.exe tools\setup_listen.py
```

**Aura hears herself.** Use headphones. It is the most reliable fix.

## 8. Privacy reminders
- Video is processed in memory on your computer and is not saved or uploaded.
- The dashboard only listens on your own computer (`localhost`).
- "Hey Aura" recognises speech offline. Audio is not saved or sent; only your words as text go to the chat.
- Your clipboard is read once, only when you say "look at this error" or "what's wrong with this".
- Detection is an early prototype and is not validated on real people yet.

## 9. Remove it completely
```powershell
cd ..
Remove-Item -Recurse -Force "$HOME\Aura"
Remove-Item -Recurse -Force "$env:APPDATA\aura-m1" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\aura-m1" -ErrorAction SilentlyContinue
```
