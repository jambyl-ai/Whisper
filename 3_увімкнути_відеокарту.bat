@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run 1_install first.
  pause
  exit /b 1
)
echo ================================================================
echo  Installing NVIDIA CUDA libraries for faster-whisper on the GPU
echo  (cuBLAS 12 + cuDNN 9.10, about 1.3 GB download, one time only)
echo ================================================================
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check "nvidia-cublas-cu12==12.4.*" "nvidia-cudnn-cu12==9.10.2.21"
if errorlevel 1 (
  echo.
  echo First attempt failed, trying the newest cuBLAS 12...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check "nvidia-cublas-cu12>=12.4,<13" "nvidia-cudnn-cu12==9.10.2.21"
)
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check "nvidia-cuda-nvrtc-cu12==12.4.*" 2>nul
echo.
echo Checking the GPU...
".venv\Scripts\python.exe" -c "import whisper_assistant as w; import ctranslate2 as c; print(\"CUDA libs:\", w.CUDA_LIBS); print(\"GPUs found:\", c.get_cuda_device_count())"
echo.
echo Done. Now start 2_run. The first start downloads the large-v3-turbo model (~1.6 GB).
pause
