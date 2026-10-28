@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto venv
where py >nul 2>nul
if %errorlevel%==0 goto pylauncher
python aura.py %*
goto end
:pylauncher
py -3 aura.py %*
goto end
:venv
".venv\Scripts\python.exe" aura.py %*
:end
endlocal
