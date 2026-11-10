@echo off
cd /d "%~dp0"
echo === Creating virtual environment ===
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
)
if not exist ".venv\Scripts\python.exe" (
  echo Python not found. Install Python 3.10-3.14 from python.org and tick "Add to PATH".
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Library install failed. Check the internet connection and run 1_install again.
  pause
  exit /b 1
)
if not exist "config.json" copy /y "config.example.json" "config.json" >nul

echo.
where nvidia-smi >nul 2>nul
if errorlevel 1 goto nogpu
echo === NVIDIA GPU found: installing CUDA libraries (about 1.3 GB, one time only) ===
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check "nvidia-cublas-cu12==12.4.*" "nvidia-cudnn-cu12==9.10.2.21"
if errorlevel 1 ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check "nvidia-cublas-cu12>=12.4,<13" "nvidia-cudnn-cu12==9.10.2.21"
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check "nvidia-cuda-nvrtc-cu12==12.4.*" 2>nul
echo Speech recognition will run on the GPU (model large-v3-turbo).
goto done

:nogpu
echo No NVIDIA GPU (integrated or AMD graphics): that is fine.
echo With a Groq key speech is recognized in the Groq cloud (large-v3-turbo),
echo without a key - on the processor (model small).

:done
echo.
echo ================================================================
echo  Done. Get a FREE key at https://console.groq.com/keys
echo  and paste it into config.json:  "keys": { "groq": "gsk_..." }
echo  Then start 2_run.
echo ================================================================
pause
