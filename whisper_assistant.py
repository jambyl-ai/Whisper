"""
Whisper-асистент: слухає звук, що відтворюється на ПК (Zoom, Discord, YouTube),
розпізнає мову через Whisper (відеокарта NVIDIA, процесор або хмара Groq)
і відповідає на почуті питання через LLM (Groq / Gemini / OpenAI / Claude / Ollama).

Запуск: 2_запустити.bat (або: python whisper_assistant.py)
Налаштування: 4_налаштування.bat (або python whisper_assistant.py --setup).
Код: тека assistant/.
"""
import sys

from assistant.system import CUDA_LIBS  # noqa: F401  — перевіряє 3_увімкнути_відеокарту.bat

if __name__ == "__main__":
    if "--setup" in sys.argv:            # 4_налаштування.bat: лише вікно налаштувань
        from assistant.settings import SettingsWindow
        SettingsWindow().run()
    else:
        from assistant.app import App
        App().run()
