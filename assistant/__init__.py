"""Whisper-асистент: слухає звук ПК, розпізнає мову (Whisper) і підказує відповіді (LLM).

Модулі:
    system       — Windows: IPv4 першим, DLL NVIDIA, soundcard, COM (імпортується першим)
    config       — config.json, ключі, варіанти моделей
    textproc     — питання / обірване речення / галюцинації
    knowledge    — профіль і папка «знання»
    audio        — запис звуку ПК і нарізка на фрази
    transcriber  — Whisper: відеокарта, процесор або хмара Groq
    llm          — моделі відповідей і запасна модель
    pipeline     — фонові потоки обробки
    app          — вікно програми
"""
import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

from . import system  # noqa: E402,F401  — до будь-якого імпорту faster-whisper (шляхи до DLL CUDA)
