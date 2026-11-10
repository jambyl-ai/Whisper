"""Налаштування: значення за замовчуванням, config.json, ключі, варіанти моделей."""
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent   # тека проєкту (де config.json, про_мене.md, знання)


DEFAULT_CONFIG = {
    "user_name": "",                   # як звати користувача (підказки — від його імені)
    "provider": "groq",
    "api_key": "",
    "keys": {},                        # ключі для кожного сервісу: {"groq": "...", "gemini": "..."}
    "model": "openai/gpt-oss-120b",
    "reasoning_effort": "low",
    "fallback": {"provider": "gemini", "model": "gemini-3.8-flash"},
    "base_url": "",
    "whisper_backend": "auto",         # auto: NVIDIA → локально; інакше хмара Groq (якщо є ключ) або процесор
    "cloud_whisper_model": "whisper-large-v3-turbo",
    "whisper_device": "auto",          # auto | cuda | cpu
    "whisper_model": "auto",           # auto: large-v3-turbo на відеокарті, small на процесорі
    "whisper_model_gpu": "large-v3-turbo",
    "whisper_model_cpu": "small",
    "whisper_compute_type": "int8",    # для процесора
    "gpu_compute_type": "float16",
    "language": None,                  # фіксована мова, напр. "en"
    "languages": ["en", "uk", "ru"],   # якщо language = null — вибір лише з цих
    "whisper_prompt": "",
    "answer_mode": "questions",
    "answer_instructions": "",
    "context": "",
    "silence_seconds": 0.6,
    "max_phrase_seconds": 25,
    "partial_seconds": 0,              # як часто показувати живий текст (0 = авто: 1,2 с GPU, 3 с CPU)
    "turn_gap_seconds": 1.2,           # тиша, після якої репліка закінчена (якщо речення виглядає завершеним)
    "unfinished_gap_seconds": 2.5,     # те саме, якщо речення обірване («What do you think about…»)
    "slow_speech_wps": 1.8,            # повільніше за стільки слів/с — паузи чекаємо довше (до ×1,8)
    "min_rms": 0.006,
    "history_phrases": 20,
    "save_log": True,
    "output_device": "",
    "profile_file": "про_мене.md",
    "knowledge_dir": "знання",
}

OPENAI_COMPAT_BASES = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
           "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}

def load_config():
    path = HERE / "config.json"
    cfg = dict(DEFAULT_CONFIG)
    if path.exists():
        with open(path, encoding="utf-8-sig") as f:
            cfg.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    else:
        example = HERE / "config.example.json"
        if example.exists():
            import shutil
            shutil.copyfile(example, path)
            with open(path, encoding="utf-8-sig") as f:
                cfg.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
    prof = HERE / cfg["profile_file"]
    prof_example = HERE / "про_мене.example.md"
    if not prof.exists() and prof_example.exists():
        import shutil
        shutil.copyfile(prof_example, prof)
    if not cfg["api_key"]:
        cfg["api_key"] = key_for(cfg, cfg["provider"])
    return cfg


def key_for(cfg, prov):
    """Ключ для провайдера: config.json → keys → <provider>, інакше змінна середовища."""
    keys = cfg.get("keys") or {}
    return keys.get(prov) or os.environ.get(KEY_ENV.get(prov, ""), "")


# Варіанти для меню «Модель» у вікні: (назва в меню, provider, model)
MODEL_CHOICES = [
    ("Groq · gpt-oss-120b — безкоштовно, швидко", "groq", "openai/gpt-oss-120b"),
    ("Gemini · 3.8 Flash — безкоштовно", "gemini", "gemini-3.8-flash"),
    ("OpenAI · GPT-6 Luna — платно", "openai", "gpt-6-luna"),
]


def save_model_choice(cfg):
    """Записати вибір моделі в config.json, не чіпаючи решту полів і коментарів."""
    path = HERE / "config.json"
    try:
        with open(path, encoding="utf-8-sig") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return
    keys = raw.setdefault("keys", {})
    if raw.get("api_key") and raw.get("provider") and not keys.get(raw["provider"]):
        keys[raw["provider"]] = raw["api_key"]       # старий ключ не губимо
    raw["provider"], raw["model"], raw["api_key"] = cfg["provider"], cfg["model"], ""
    raw["fallback"] = cfg.get("fallback") or {}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)


def pick_fallback(cfg, prov):
    """Запасна модель: інша безкоштовна, для якої є ключ (платну запасною не ставимо)."""
    for _, p, m in MODEL_CHOICES:
        if p != prov and p != "openai" and key_for(cfg, p):
            return {"provider": p, "model": m}
    return {}
