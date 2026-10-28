@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 goto pylauncher
python install.py %*
goto end
:pylauncher
py -3 install.py %*
:end
endlocal
