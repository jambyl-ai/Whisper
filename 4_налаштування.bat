@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run 1_install first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" whisper_assistant.py --setup
if errorlevel 1 pause
