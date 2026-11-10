"""
Whisper-асистент: слухає звук, що відтворюється на ПК (Zoom, Discord, YouTube),
розпізнає мову через Whisper (відеокарта NVIDIA, процесор або хмара Groq)
і відповідає на почуті питання через LLM (Groq / Gemini / OpenAI / Claude / Ollama).

Запуск: 2_запустити.bat (або: python whisper_assistant.py)
Код: тека assistant/. Налаштування: config.json поруч із цим файлом.
"""
from assistant.system import CUDA_LIBS  # noqa: F401  — перевіряє 3_увімкнути_відеокарту.bat
from assistant.app import App

if __name__ == "__main__":
    App().run()
