@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run 1_install first.
  pause
  exit /b 1
)
echo Checking libraries...
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt
echo Starting...
".venv\Scripts\python.exe" whisper_assistant.py
if errorlevel 1 pause
