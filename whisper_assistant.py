"""
Whisper-асистент: слухає звук, що відтворюється на ПК (Zoom, Discord, YouTube),
розпізнає мову через faster-whisper (локально, на відеокарті NVIDIA або процесорі)
і відповідає на почуті питання через LLM (Gemini / Groq / Ollama / Claude / OpenAI).

Запуск: 2_запустити.bat (або: python whisper_assistant.py)
Відеокарта: один раз запустити 3_увімкнути_відеокарту.bat
Налаштування: config.json поруч із цим файлом.
"""
import ctypes
import html
import json
import os
import queue
import re
import socket
import sys
import threading
import time
import tkinter as tk
import types
import warnings
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

HERE = Path(__file__).resolve().parent
SR = 16000            # частота дискретизації для Whisper
BLOCK = 1600          # 0,1 с

# Спершу IPv4: якщо IPv6 у мережі «зламаний», кожна спроба з'єднання інакше
# висить до тайм-ауту, і відповідь приходить через хвилини.
_orig_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_ipv4_first(*args, **kwargs):
    res = _orig_getaddrinfo(*args, **kwargs)
    return sorted(res, key=lambda r: 0 if r[0] == socket.AF_INET else 1)


socket.getaddrinfo = _getaddrinfo_ipv4_first


# --------------------------------------------------------------------------- CUDA DLL
def setup_cuda_dlls():
    """Додати теки з DLL бібліотек NVIDIA (pip-пакети nvidia-*) у пошук Windows."""
    found = []
    if sys.platform != "win32":
        return found
    for sp in {p for p in sys.path if p and "site-packages" in p}:
        base = Path(sp) / "nvidia"
        if not base.is_dir():
            continue
        for b in sorted(base.glob("*/bin")):
            try:
                os.add_dll_directory(str(b))
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
            found.append(b.parent.name)
    return found


CUDA_LIBS = setup_cuda_dlls()


# --------------------------------------------------------------------------- soundcard
def _load_soundcard():
    """Імпорт soundcard з виправленням для Python 3.12+ на Windows.

    soundcard при імпорті викликає platform.win32_ver(), який у нових Python сам
    ініціалізує COM, після чого CoInitializeEx повертає S_FALSE (1) — «вже
    ініціалізовано». Це код успіху, але soundcard вважає його помилкою
    («Error 0x100000001»). Тут модуль завантажується з одним виправленням:
    S_FALSE приймається як успіх.
    """
    if sys.platform != "win32":
        import soundcard
        return soundcard
    import importlib.util
    spec = importlib.util.find_spec("soundcard")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("бібліотеку soundcard не встановлено (запустіть 1_встановити.bat)")
    pkg_dir = Path(list(spec.submodule_search_locations)[0])
    src = (pkg_dir / "mediafoundation.py").read_text(encoding="utf-8")
    marker = "self.check_error(hr)"          # перша поява — у _COMLibrary.__init__
    if marker not in src:                     # інша версія soundcard — звичайний імпорт
        import soundcard
        return soundcard
    src = src.replace(marker, "(hr == 1) or self.check_error(hr)", 1)

    pkg = types.ModuleType("soundcard")
    pkg.__path__ = [str(pkg_dir)]
    pkg.__file__ = str(pkg_dir / "__init__.py")
    pkg.__package__ = "soundcard"
    mf = types.ModuleType("soundcard.mediafoundation")
    mf.__file__ = str(pkg_dir / "mediafoundation.py")
    mf.__package__ = "soundcard"
    sys.modules["soundcard"] = pkg
    sys.modules["soundcard.mediafoundation"] = mf
    try:
        exec(compile(src, mf.__file__, "exec"), mf.__dict__)
    except Exception:
        sys.modules.pop("soundcard", None)
        sys.modules.pop("soundcard.mediafoundation", None)
        raise
    for name in dir(mf):
        if not name.startswith("_"):
            setattr(pkg, name, getattr(mf, name))
    pkg.mediafoundation = mf
    return pkg


try:
    sc = _load_soundcard()
    SC_ERROR = None
except Exception as _e:  # noqa: BLE001
    sc = None
    SC_ERROR = _e


def com_init():
    """COM у фоновому потоці (той самий багатопотоковий режим, що й у soundcard)."""
    if sys.platform == "win32":
        try:
            ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        except Exception:
            pass


# --------------------------------------------------------------------------- config
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
    "merge_gap_seconds": 2.0,          # пауза, після якої незакінчене речення все одно вважається завершеним
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

QUESTION_WORDS = {
    # en
    "what", "why", "how", "when", "where", "who", "whom", "whose", "which", "can", "could",
    "would", "will", "do", "does", "did", "is", "are", "was", "were", "should", "have", "has",
    "tell", "explain", "describe", "name", "give", "say",
    # uk / ru
    "що", "чому", "як", "коли", "де", "хто", "який", "яка", "яке", "які", "скільки", "навіщо",
    "чи", "поясніть", "поясни", "розкажіть", "розкажи", "назвіть", "назви", "опишіть", "скажіть",
    "что", "почему", "как", "когда", "где", "кто", "какой", "зачем", "сколько", "расскажи",
}

# типові «галюцинації» Whisper на тиші й музиці
HALLUCINATIONS = ("amara.org", "dimatorzok", "субтитр", "продолжение следует",
                  "thanks for watching", "thank you for watching", "дякую за перегляд",
                  "please subscribe", "підписуйтесь", "подписывайтесь")


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


def looks_like_question(text):
    t = text.strip().lower()
    if "?" in t:
        return True
    words = re.findall(r"[\w'’-]+", t)
    return bool(words) and words[0] in QUESTION_WORDS


def sentence_done(text):
    """Схоже на завершене речення чи питання."""
    t = text.rstrip(' "»”)\'')
    return t.endswith((".", "!", "?", "…", "。")) or looks_like_question(text)


def is_hallucination(text):
    t = text.strip().lower()
    if len(t) < 2 or t.strip(" .!") in ("you", "thank you", "дякую", "bye", "спасибо"):
        return True
    return any(h in t for h in HALLUCINATIONS)


def read_docx(path):
    """Текст із .docx без сторонніх бібліотек."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab/>", "\t", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    xml = re.sub(r"</w:tc>", " | ", xml)
    text = html.unescape(re.sub(r"<[^>]+>", "", xml))
    return re.sub(r"\n{3,}", "\n\n", text)


def read_any(path):
    """Текст із .txt/.md/.docx/.pdf (інші типи пропускаємо)."""
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".docx":
        return read_docx(path)
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("для PDF потрібна бібліотека pypdf (запустіть 1_встановити.bat)") from None
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)
    return ""


def load_profile_and_knowledge(cfg):
    prof_path = HERE / cfg["profile_file"]
    profile = prof_path.read_text(encoding="utf-8") if prof_path.exists() else ""
    docs, errors = [], []
    kdir = HERE / cfg["knowledge_dir"]
    kdir.mkdir(exist_ok=True)
    for p in sorted(kdir.rglob("*")):
        if p.is_file() and not p.name.startswith("~$") and p.name.lower() != "readme.md":
            try:
                t = read_any(p).strip()
            except Exception as e:  # noqa: BLE001
                t = ""
                errors.append(f"{p.name}: {e}")
            if t:
                docs.append((p.name, t))
    return profile, docs, errors


# --------------------------------------------------------------------------- audio
class LoopbackCapture(threading.Thread):
    """Записує те, що грає в навушниках/динаміках (WASAPI loopback, спільний режим).

    Loopback не забирає пристрій: звук грає як завжди, мікрофон гарнітури
    не відкривається (Bluetooth-навушники не перемикаються в Hands-Free).
    """

    def __init__(self, out_q, status, error, device_hint=""):
        super().__init__(daemon=True)
        self.out_q, self.status, self.error = out_q, status, error
        self.device_hint = (device_hint or "").lower()
        self.stop_flag = threading.Event()
        self.paused = threading.Event()

    def pick_speaker(self):
        if self.device_hint:
            for sp in sc.all_speakers():
                if self.device_hint in sp.name.lower():
                    return sp
            self.error(f"Пристрій «{self.device_hint}» не знайдено — слухаю пристрій за замовчуванням.")
            self.device_hint = ""
        return sc.default_speaker()

    def run(self):
        com_init()
        if sc is None:
            self.status("Запис звуку не працює")
            self.error(f"Не вдалося завантажити soundcard: {SC_ERROR!r}")
            return
        warnings.filterwarnings("ignore", message=".*discontinuity.*")
        last_err = None
        while not self.stop_flag.is_set():
            try:
                speaker = self.pick_speaker()
                mic = sc.get_microphone(id=speaker.id, include_loopback=True)
                self.status(f"Слухаю: {speaker.name}")
                last_err = None
                last_check = time.monotonic()
                with mic.recorder(samplerate=SR, blocksize=BLOCK) as rec:
                    while not self.stop_flag.is_set():
                        data = rec.record(numframes=BLOCK)
                        if not self.paused.is_set():
                            mono = data.mean(axis=1) if data.ndim == 2 else data
                            self.out_q.put(mono.astype(np.float32))
                        if not self.device_hint and time.monotonic() - last_check > 2:
                            last_check = time.monotonic()
                            if sc.default_speaker().id != speaker.id:
                                self.error("Змінився пристрій виводу — перепідключаюсь.")
                                break
            except Exception as e:  # noqa: BLE001
                msg = f"Помилка аудіо: {e!r}"
                if msg != last_err:
                    self.error(msg + " Повтор через 2 с…")
                    last_err = msg
                self.status("Помилка аудіо, повтор…")
                time.sleep(2)


class Segmenter:
    """Нарізає потік на фрази за паузами (простий енергетичний VAD)."""

    def __init__(self, cfg):
        self.silence_blocks = max(1, int(round(cfg["silence_seconds"] * 10)))
        self.max_blocks = int(cfg["max_phrase_seconds"] * 10)
        self.min_rms = cfg["min_rms"]
        self.noise = self.min_rms / 3
        self.pre = deque(maxlen=3)
        self.buf, self.silent, self.active = [], 0, False

    def feed(self, block, rms):
        thr = max(self.min_rms, self.noise * 3)
        speech = rms > thr
        if not speech:
            self.noise = 0.95 * self.noise + 0.05 * rms
        if not self.active:
            self.pre.append(block)
            if speech:
                self.active, self.buf, self.silent = True, list(self.pre), 0
            return None
        self.buf.append(block)
        self.silent = 0 if speech else self.silent + 1
        if self.silent >= self.silence_blocks or len(self.buf) >= self.max_blocks:
            seg = np.concatenate(self.buf)
            self.active, self.buf = False, []
            self.pre.clear()
            if len(seg) >= SR * 0.5:
                return seg
        return None

    def current(self):
        """Звук поточної фрази, що ще звучить (для живого тексту), або None."""
        return np.concatenate(self.buf) if self.active and self.buf else None


# --------------------------------------------------------------------------- Whisper
LANG_CODES = {"english": "en", "ukrainian": "uk", "russian": "ru", "polish": "pl", "german": "de",
              "french": "fr", "spanish": "es", "italian": "it", "portuguese": "pt", "czech": "cs",
              "slovak": "sk", "belarusian": "be", "japanese": "ja", "chinese": "zh", "korean": "ko"}


class Transcriber:
    def __init__(self, cfg, post):
        self.cfg, self.post = cfg, post
        self.cloud, self.local, self.cloud_off_until, self.last_any = None, None, 0.0, ""
        self.model, self.device, self.name = self.load()
        self.beam = 5 if self.device == "cuda" else 1
        self.last_text = {}

    def _use_cloud(self):
        import openai
        self.cloud = openai.OpenAI(api_key=key_for(self.cfg, "groq"), base_url=OPENAI_COMPAT_BASES["groq"],
                                   timeout=30, max_retries=0)
        return None, "cloud", self.cfg.get("cloud_whisper_model") or "whisper-large-v3-turbo"

    def _load_cpu(self):
        from faster_whisper import WhisperModel
        cfg = self.cfg
        name = cfg["whisper_model"] if cfg["whisper_model"] != "auto" else cfg["whisper_model_cpu"]
        self.post("info", f"Завантажую Whisper «{name}» на процесор…")
        threads = max(4, (os.cpu_count() or 8) // 2)
        m = WhisperModel(name, device="cpu", compute_type=cfg["whisper_compute_type"] or "int8",
                         cpu_threads=threads)
        return m, "cpu", name

    def load(self):
        cfg = self.cfg
        backend = (cfg.get("whisper_backend") or "auto").lower()
        dev = (cfg.get("whisper_device") or "auto").lower()
        has_groq = bool(key_for(cfg, "groq"))
        if backend in ("groq", "cloud"):
            if has_groq:
                return self._use_cloud()
            self.post("info", "Для хмарного Whisper потрібен ключ Groq (config.json → keys → groq). "
                              "Розпізнаю на цьому комп'ютері.")
        from faster_whisper import WhisperModel
        reason = ""
        if dev in ("auto", "cuda", "gpu"):
            n = 0
            try:
                import ctranslate2
                n = ctranslate2.get_cuda_device_count()
            except Exception as e:  # noqa: BLE001
                reason = repr(e)
            if n > 0:
                name = cfg["whisper_model"] if cfg["whisper_model"] != "auto" else cfg["whisper_model_gpu"]
                try:
                    self.post("info", f"Завантажую Whisper «{name}» на відеокарту "
                                      "(перший раз качає ~1,6 ГБ — кілька хвилин)…")
                    m = WhisperModel(name, device="cuda", compute_type=cfg["gpu_compute_type"])
                    # пробний прогін: саме тут підвантажуються cuBLAS/cuDNN
                    noise = (np.random.default_rng(0).standard_normal(SR) * 1e-3).astype(np.float32)
                    list(m.transcribe(noise, language="en", beam_size=1)[0])
                    return m, "cuda", name
                except Exception as e:  # noqa: BLE001
                    reason = str(e)
            elif not reason:
                reason = "CUDA-відеокарту не знайдено (немає драйвера NVIDIA?)"
            hint = (" Запустіть 3_увімкнути_відеокарту.bat."
                    if ("cublas" in reason.lower() or "cudnn" in reason.lower()) and not CUDA_LIBS else "")
            if backend == "auto" and dev == "auto" and has_groq:
                self.post("info", f"Відеокарта NVIDIA не використовується: {reason[:200]}.{hint} "
                                  "Розпізнаю в хмарі Groq (whisper-large-v3-turbo) — точно й швидко на будь-якому ПК.")
                return self._use_cloud()
            self.post("info", f"Відеокарта не використовується: {reason[:300]}.{hint} Працюю на процесорі.")
        return self._load_cpu()

    def _run(self, audio, lang, beam=None, model=None):
        prompt_parts = []
        if self.cfg.get("whisper_prompt") and lang in (None, "en"):
            prompt_parts.append(self.cfg["whisper_prompt"])
        if lang and self.last_text.get(lang):
            prompt_parts.append(self.last_text[lang][-200:])
        segs, info = (model or self.model).transcribe(
            audio, language=lang, beam_size=beam or self.beam, vad_filter=self.cfg.get("whisper_vad", True),
            condition_on_previous_text=False,
            initial_prompt=" ".join(prompt_parts) or None)
        return segs, info

    def _cloud_transcribe(self, audio):
        import io
        import wave
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        fixed = self.cfg.get("language") or None
        kw = {"model": self.name, "file": ("audio.wav", buf.getvalue()), "response_format": "verbose_json"}
        if fixed:
            kw["language"] = fixed
        prompt = " ".join(x for x in (self.cfg.get("whisper_prompt") or "", self.last_any[-200:]) if x)
        if prompt:
            kw["prompt"] = prompt[:800]
        r = self.cloud.audio.transcriptions.create(**kw)
        text = (getattr(r, "text", "") or "").strip()
        raw = str(getattr(r, "language", "") or "").lower()
        lang = fixed or LANG_CODES.get(raw, raw[:2] or "?")
        return text, lang

    def transcribe(self, audio, partial=False):
        if self.device == "cloud":
            if time.monotonic() >= self.cloud_off_until:
                try:
                    text, lang = self._cloud_transcribe(audio)
                    if text and not partial:
                        self.last_any = text
                    return text, lang
                except Exception as e:  # noqa: BLE001
                    self.cloud_off_until = time.monotonic() + 60
                    if not partial:
                        self.post("info", f"Хмарний Whisper не відповів ({str(e)[:160]}) — "
                                          "хвилину розпізнаю на цьому комп'ютері.")
            if partial:
                return "", ""
            if self.local is None:
                self.local = self._load_cpu()[0]
            return self._local_transcribe(audio, False, self.local)
        return self._local_transcribe(audio, partial, None)

    def _local_transcribe(self, audio, partial, model):
        beam = 1 if (partial or model is not None) else None
        fixed = self.cfg.get("language") or None
        allowed = [l for l in (self.cfg.get("languages") or []) if l]
        if fixed or not allowed:
            segs, info = self._run(audio, fixed, beam, model)
            lang = fixed or info.language
        else:
            segs, info = self._run(audio, None, beam, model)   # сегменти «ліниві», тут лише визначення мови
            lang = info.language
            if lang not in allowed:
                probs = dict(info.all_language_probs or [])
                lang = max(allowed, key=lambda l: probs.get(l, 0.0))
                segs, info = self._run(audio, lang, beam, model)
        text = " ".join(s.text.strip() for s in segs).strip()
        if text and not partial:
            self.last_text[lang] = text
        return text, lang


# --------------------------------------------------------------------------- LLM
class GeminiNative:
    """Рідний REST API Gemini (працює і зі старими AIza-, і з новими AQ.-ключами).

    Без додаткових бібліотек: лише urllib. Відповідь приходить потоком (SSE).
    """
    BASE = "https://generativelanguage.googleapis.com/v1beta"
    THINKING = [{"thinkingLevel": "minimal"}, {"thinkingLevel": "low"}, {"thinkingBudget": 0}, None]

    def __init__(self, key, model, notify):
        self.key, self.model, self.notify = key, model, notify
        self.think_idx = 0

    def _open(self, url, body=None, timeout=25):
        import urllib.error
        import urllib.request
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                     headers={"x-goog-api-key": self.key,
                                              "Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:400]
            raise RuntimeError(f"Error code: {e.code} - {detail}") from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            raise RuntimeError(f"Connection error / timeout: {e}") from None

    def pick_model(self):
        """Знайти актуальну Flash-модель, якщо вказаної вже немає."""
        with self._open(self.BASE + "/models?pageSize=200") as r:
            res = json.loads(r.read().decode("utf-8"))
        names = []
        for m in res.get("models", []):
            name = m.get("name", "").split("/")[-1]
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            if "flash" in name and not any(x in name for x in
                                           ("lite", "image", "tts", "live", "audio", "embedding", "thinking", "exp")):
                names.append(name)
        if not names:
            raise RuntimeError("Error code: 404 - не знайдено жодної доступної Flash-моделі")
        stable = [n for n in names if "preview" not in n] or names
        return sorted(stable, reverse=True)[0]

    def stream(self, system, user, max_tokens=1024):
        attempt, transient = 0, 0
        while True:
            attempt += 1
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"maxOutputTokens": max_tokens}}
            tc = self.THINKING[self.think_idx]
            if tc:
                body["generationConfig"]["thinkingConfig"] = tc
            url = f"{self.BASE}/models/{self.model}:streamGenerateContent?alt=sse"
            try:
                resp = self._open(url, body)
            except RuntimeError as e:
                msg = str(e)
                if attempt >= 6:
                    raise
                if "404" in msg:
                    new = self.pick_model()
                    if new == self.model:
                        raise
                    self.notify(f"Модель «{self.model}» недоступна — використовую «{new}».")
                    self.model = new
                    continue
                if "400" in msg and "thinking" in msg.lower() and self.think_idx < len(self.THINKING) - 1:
                    self.think_idx += 1
                    continue
                if transient < 2 and any(c in msg for c in ("500", "502", "503", "504", "Connection", "timeout")):
                    transient += 1
                    time.sleep(1.0)
                    continue
                raise
            with resp:
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        obj = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    for c in (obj.get("candidates") or [])[:1]:
                        for p in c.get("content", {}).get("parts", []):
                            if p.get("text") and not p.get("thought"):
                                yield p["text"]
            return


class Answerer:
    def __init__(self, cfg, notify=print):
        self.cfg = cfg
        self.notify = notify
        self.reload()
        self.main = self._make(cfg)
        self.backup = None
        fb = cfg.get("fallback") or {}
        if isinstance(fb, dict) and fb.get("provider"):
            fcfg = dict(cfg)
            fcfg.update({k: v for k, v in fb.items() if not k.startswith("_")})
            fcfg["api_key"] = fb.get("api_key") or key_for(cfg, fb["provider"])
            fcfg["base_url"] = fb.get("base_url")
            self.backup = self._make(fcfg)
        if self.main and self.backup and hasattr(self.main[0], "with_options"):
            # є запасний — не чекаємо повторів основного (напр. 429 ліміт), одразу йдемо на запасний
            self.main = (self.main[0].with_options(max_retries=0), self.main[1])
        self.client = self.main or self.backup

    def _make(self, cfg):
        client = self._new_client(cfg)
        return (client, cfg) if client else None

    def _new_client(self, cfg):
        prov = cfg["provider"]
        if prov == "gemini" and not cfg.get("base_url"):
            return GeminiNative(cfg["api_key"], cfg["model"], self.notify) if cfg["api_key"] else None
        if prov == "anthropic":
            if not cfg["api_key"]:
                return None
            import anthropic
            return anthropic.Anthropic(api_key=cfg["api_key"])
        import openai
        key = cfg["api_key"] or ("ollama" if prov == "ollama" else "")
        if not key:
            return None
        base = cfg.get("base_url") or OPENAI_COMPAT_BASES.get(prov)
        return openai.OpenAI(api_key=key, base_url=base, timeout=30, max_retries=1) \
            if base else openai.OpenAI(api_key=key, timeout=30, max_retries=1)

    def reload(self):
        self.profile, self.docs, self.load_errors = load_profile_and_knowledge(self.cfg)
        return len(self.docs)

    def system_prompt(self):
        name = (self.cfg.get("user_name") or "").strip()
        who = f"користувача, якого звати {name}" if name else "користувача"
        s = (f"Ти — підказувач для {who}. "
             "Ти чуєш розмову, що звучить на його комп'ютері (пара в Zoom, дзвінок, відео): тобі дають "
             "розшифровку останніх фраз (Whisper, можливі помилки розпізнавання — враховуй схожі за "
             "звучанням слова). Склади відповідь на ОСТАННЄ питання так, щоб користувач міг одразу сказати "
             "її вголос від першої особи («I…», «Я…»).\n"
             "Правила:\n"
             "1. Спершу шукай відповідь у МАТЕРІАЛАХ нижче (його тексти, роботи, конспекти). Якщо питання "
             "про текст чи тему роботи — відповідай за текстом, його словами й термінами.\n"
             "2. Питання про нього самого (навчання, спеціальність, досвід, плани) — бери з ПРОФІЛЮ. "
             "Не вигадуй фактів про нього: якщо в профілі немає потрібного, дай нейтральну правдоподібну "
             "відповідь і познач її в кінці «(уточни)».\n"
             "3. Якщо в матеріалах відповіді немає — відповідай із загальних знань.\n"
             "4. Відповідь коротка: 1–4 речення, простими словами. Англійською — якщо питали англійською; "
             "якщо питали українською чи російською — відповідай українською. Без вступів.\n"
             "5. В історії є мітки: «[питання вручну]» — питання, яке користувач написав тобі сам; "
             "«[твоя підказка]» — що ти відповів раніше. Короткі уточнення («why?», «а чому?», «more») "
             "стосуються попереднього питання і твоєї попередньої підказки — продовжуй ту саму тему.\n"
             "6. Якщо в останній фразі немає питання чи прохання, на яке треба відповісти "
             "(наприклад, говорить хтось інший про своє, вітання, технічні фрази), напиши рівно: NONE.")
        if self.cfg.get("answer_instructions"):
            s += "\n\nДодаткові вимоги до відповіді: " + self.cfg["answer_instructions"]
        if self.cfg.get("context"):
            s += "\n\nДодатковий контекст: " + self.cfg["context"]
        if self.profile:
            s += "\n\n=== ПРОФІЛЬ (про користувача) ===\n" + self.profile
        for name, text in self.docs:
            s += f"\n\n=== МАТЕРІАЛ: {name} ===\n{text}"
        return s

    def stream(self, history, force=False):
        """Генератор шматків відповіді."""
        convo = "\n".join(f"- {h}" for h in history[:-1]) or "(немає)"
        user = f"Попередні фрази:\n{convo}\n\nОстання фраза:\n{history[-1]}"
        if force:
            user += "\n\nКористувач сам попросив відповідь: дай найкращу відповідь або коментар, не NONE."
        system = self.system_prompt()
        first = self.main or self.backup
        got = False
        try:
            for t in self._stream(first, system, user):
                got = True
                yield t
            return
        except Exception as e:  # noqa: BLE001
            if got or not self.backup or first is self.backup:
                raise
            why = str(e).replace("\n", " ")[:220]
        self.notify(f"Основний сервер ({first[1]['model']}) не відповів — відповідаю через запасний "
                    f"({self.backup[1]['model']}). Причина: {why}")
        yield from self._stream(self.backup, system, user)

    @staticmethod
    def _stream(pair, system, user):
        client, cfg = pair
        if isinstance(client, GeminiNative):
            yield from client.stream(system, user)
        elif cfg["provider"] == "anthropic":
            with client.messages.stream(
                    model=cfg["model"], max_tokens=500,
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": user}]) as st:
                yield from st.text_stream
        else:
            kw = {"max_tokens": 1024}
            if cfg["provider"] == "openai":
                kw = {"max_completion_tokens": 2048}
            if cfg["provider"] in ("openai", "groq") and cfg.get("reasoning_effort"):
                kw["reasoning_effort"] = cfg["reasoning_effort"]
            resp = client.chat.completions.create(
                model=cfg["model"], stream=True, **kw,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            for ch in resp:
                if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content:
                    yield ch.choices[0].delta.content

def friendly_llm_error(e):
    s = str(e)
    if "429" in s or "quota" in s.lower() or "RESOURCE_EXHAUSTED" in s or "rate limit" in s.lower():
        return "Ліміт безкоштовного тарифу: зачекайте хвилину (або перемкніться на інший provider)."
    if "401" in s or "403" in s or "API key" in s or ("invalid" in s.lower() and "key" in s.lower()):
        return f"Ключ не підходить: перевірте api_key і provider у config.json. (Деталі: {s[:200]})"
    if "404" in s:
        return f"Модель не знайдено: змініть \"model\" у config.json. (Деталі: {s[:200]})"
    if "503" in s or "overloaded" in s.lower():
        return "Сервер відповідей перевантажений (503). Спробуйте ще раз за кілька секунд."
    if "Connection" in s or "timeout" in s.lower() or "timed out" in s.lower():
        return "Немає зв'язку з сервером відповідей або він не відповів вчасно (інтернет?)."
    return f"Помилка відповіді: {s[:300]}"


# --------------------------------------------------------------------------- app
BG, FG, MUTED, ACCENT, WARN = "#1e1f22", "#e6e6e6", "#9aa0a6", "#7ee787", "#e3b341"


class App:
    def __init__(self):
        self.cfg = load_config()
        self.ui_q = queue.Queue()
        self.audio_q = queue.Queue()
        self.phrase_q = queue.Queue()
        self.answer_q = queue.Queue()
        self.history = deque(maxlen=self.cfg["history_phrases"])
        self.hist_lock = threading.Lock()
        self.mode = self.cfg["answer_mode"]
        self.answerer = None
        self.capture = None
        self.thinking_since = None
        self.log_path = None
        if self.cfg["save_log"]:
            (HERE / "logs").mkdir(exist_ok=True)
            self.log_path = HERE / "logs" / f"{datetime.now():%Y-%m-%d_%H-%M}.txt"
        self.build_ui()
        threading.Thread(target=self.worker, daemon=True).start()
        self.root.after(100, self.poll)

    # ---- UI (створюється й змінюється лише в головному потоці)
    def build_ui(self):
        self.root = tk.Tk()
        self.root.title("Whisper-асистент")
        self.root.geometry("640x520+40+40")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)

        bar = tk.Frame(self.root, bg=BG)
        bar.pack(side="top", fill="x", padx=4, pady=4)
        btn = dict(bg="#2b2d31", fg=FG, activebackground="#3a3d42", activeforeground=FG,
                   relief="flat", padx=8, pady=3, bd=0, highlightthickness=0)
        tk.Button(bar, text="Відповісти зараз", command=self.force_answer, **btn).pack(side="left", padx=2)
        self.pause_btn = tk.Button(bar, text="Пауза", command=self.toggle_pause, **btn)
        self.pause_btn.pack(side="left", padx=2)
        tk.Button(bar, text="Очистити", command=self.clear, **btn).pack(side="left", padx=2)
        tk.Button(bar, text="Оновити знання", command=self.reload_knowledge, **btn).pack(side="left", padx=2)
        self.top_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Поверх", variable=self.top_var, bg=BG, fg=FG, selectcolor="#2b2d31",
                       activebackground=BG, activeforeground=FG,
                       command=lambda: self.root.attributes("-topmost", self.top_var.get())).pack(side="left", padx=4)
        self.mode_var = tk.StringVar(value=self.mode)
        self.mode_var.trace_add("write", lambda *_: setattr(self, "mode", self.mode_var.get()))
        om = tk.OptionMenu(bar, self.mode_var, "questions", "all", "manual")
        om.configure(bg="#2b2d31", fg=FG, activebackground="#3a3d42", activeforeground=FG,
                     relief="flat", highlightthickness=0, bd=0)
        om["menu"].configure(bg="#2b2d31", fg=FG)
        om.pack(side="right")
        tk.Label(bar, text="режим:", bg=BG, fg=MUTED).pack(side="right")

        mrow = tk.Frame(self.root, bg=BG)
        mrow.pack(side="top", fill="x", padx=4, pady=(0, 4))
        tk.Label(mrow, text="Модель:", bg=BG, fg=MUTED).pack(side="left")
        self.model_var = tk.StringVar(value=self.current_choice_label())
        self.model_var.trace_add("write", lambda *_: self.switch_model(self.model_var.get()))
        mm = tk.OptionMenu(mrow, self.model_var, *[c[0] for c in MODEL_CHOICES])
        mm.configure(bg="#2b2d31", fg=FG, activebackground="#3a3d42", activeforeground=FG,
                     relief="flat", highlightthickness=0, bd=0, anchor="w")
        mm["menu"].configure(bg="#2b2d31", fg=FG)
        mm.pack(side="left", padx=4)
        self.fb_var = tk.StringVar(value=self.fallback_text())
        tk.Label(mrow, textvariable=self.fb_var, bg=BG, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")

        sb = tk.Frame(self.root, bg="#17181a")
        sb.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="Запуск…")
        self.level_var = tk.StringVar(value="")
        tk.Label(sb, textvariable=self.level_var, anchor="e", font=("Consolas", 9),
                 bg="#17181a", fg=ACCENT, width=18).pack(side="right")
        tk.Label(sb, textvariable=self.status_var, anchor="w", font=("Segoe UI", 9),
                 bg="#17181a", fg=MUTED).pack(side="left", fill="x")

        ask = tk.Frame(self.root, bg=BG)
        ask.pack(side="bottom", fill="x", padx=4, pady=(0, 4))
        tk.Label(ask, text="Своє питання:", bg=BG, fg=MUTED).pack(side="left")
        self.ask_var = tk.StringVar()
        ent = tk.Entry(ask, textvariable=self.ask_var, bg="#2b2d31", fg=FG, insertbackground=FG,
                       relief="flat", font=("Segoe UI", 10))
        ent.pack(side="left", fill="x", expand=True, padx=4, ipady=3)
        ent.bind("<Return>", lambda _e: self.ask_manual())
        tk.Button(ask, text="Спитати", command=self.ask_manual, **btn).pack(side="right")

        panes = tk.PanedWindow(self.root, orient="vertical", bg=BG, sashwidth=6, bd=0)
        panes.pack(fill="both", expand=True, padx=4)
        self.answer_box = tk.Text(panes, wrap="word", height=6, bg="#14261b", fg=ACCENT,
                                  font=("Segoe UI", 13, "bold"), relief="flat", padx=8, pady=6,
                                  insertbackground=FG)
        self.answer_box.insert("end", "Тут з'явиться відповідь.")
        panes.add(self.answer_box, minsize=60, height=150)
        frame = tk.Frame(panes, bg=BG)
        scroll = tk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        self.text = tk.Text(frame, wrap="word", font=("Segoe UI", 10), height=10, bg=BG, fg=FG,
                            relief="flat", padx=6, pady=4, yscrollcommand=scroll.set, insertbackground=FG)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.text.yview)
        panes.add(frame, minsize=80)
        self.text.tag_config("heard", foreground="#c9ccd1")
        self.text.tag_config("partial", foreground=MUTED, font=("Segoe UI", 10, "italic"))
        self.text.tag_config("lang", foreground=MUTED, font=("Consolas", 8))
        self.text.tag_config("answer", foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        self.text.tag_config("info", foreground=WARN, font=("Segoe UI", 9))
        self.text.tag_config("mine", foreground="#79c0ff")

    def current_choice_label(self):
        for label, prov, model in MODEL_CHOICES:
            if prov == self.cfg["provider"] and model == self.cfg["model"]:
                return label
        return f'{self.cfg["provider"]} · {self.cfg["model"]}'

    def fallback_text(self):
        fb = self.cfg.get("fallback") or {}
        return f"запасна: {fb['provider']} · {fb.get('model', '')}" if fb.get("provider") else "без запасної"

    def switch_model(self, label):
        choice = next((c for c in MODEL_CHOICES if c[0] == label), None)
        if not choice or (choice[1], choice[2]) == (self.cfg["provider"], self.cfg["model"]):
            return
        _, prov, model = choice
        key = key_for(self.cfg, prov)
        if not key and prov == self.cfg["provider"]:
            key = self.cfg["api_key"]
        if not key:
            self.add_line(f"Немає ключа для {prov}: впишіть його в config.json → \"keys\" → \"{prov}\" "
                          "і перезапустіть.", "info")
            self.root.after(0, lambda: self.model_var.set(self.current_choice_label()))
            return
        keys = self.cfg.setdefault("keys", {})
        if self.cfg.get("api_key") and not keys.get(self.cfg["provider"]):
            keys[self.cfg["provider"]] = self.cfg["api_key"]
        # запасна — інша безкоштовна модель, для якої є ключ (платну як запасну не ставимо)
        fb = next(({"provider": p, "model": m} for _, p, m in MODEL_CHOICES
                   if p != prov and p != "openai" and key_for(self.cfg, p)), {})
        self.cfg.update(provider=prov, model=model, api_key=key, fallback=fb)
        try:
            new = Answerer(self.cfg, notify=lambda m: self.post("info", m))
        except Exception as e:  # noqa: BLE001
            self.add_line(f"Не вдалося підключити {prov}: {e!r}", "info")
            return
        if self.answerer is not None:
            self.answerer = new          # answer_loop підхопить нову модель з наступного питання
        self.fb_var.set(self.fallback_text())
        save_model_choice(self.cfg)
        self.add_line(f"Модель: {label}. {self.fallback_text().capitalize()}.", "info")

    def clear(self):
        self.text.delete("1.0", "end")
        self.answer_box.delete("1.0", "end")

    def post(self, kind, *payload):
        self.ui_q.put((kind, payload))

    def log(self, line):
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now():%H:%M:%S}] {line}\n")

    def add_line(self, text, tag, prefix=""):
        self.text.insert("end", prefix + text + "\n\n", tag)
        self.text.see("end")

    def show_live(self, text, lang):
        """Живий рядок «що чую зараз»: оновлюється на місці, доки фраза не закінчиться."""
        rng = self.text.tag_ranges("live")
        if rng:
            pos = str(rng[0])
            self.text.delete(rng[0], rng[-1])
        else:
            pos = self.text.index("end-1c")
        self.text.insert(pos, f"[{lang} …] ", ("lang", "live"), "🗣 " + text + " …\n\n", ("partial", "live"))
        self.text.see("end")

    def clear_live(self):
        rng = self.text.tag_ranges("live")
        if rng:
            self.text.delete(rng[0], rng[-1])

    def poll(self):
        try:
            for _ in range(300):
                kind, p = self.ui_q.get_nowait()
                if kind == "status":
                    self.status_var.set(p[0])
                elif kind == "level":
                    self.level_var.set(p[0])
                elif kind == "partial":
                    self.show_live(*p)
                elif kind == "live_off":
                    self.clear_live()
                elif kind == "heard":
                    text, lang, dt = p
                    self.clear_live()
                    self.text.insert("end", f"[{lang} {dt:.1f}с] ", "lang")
                    self.add_line(text, "heard", "🗣 ")
                    self.log(f"🗣 [{lang} {dt:.1f}s] {text}")
                elif kind == "mine":
                    self.add_line(p[0], "mine", "❓ ")
                    self.log(f"❓ {p[0]}")
                elif kind == "info":
                    self.add_line(p[0], "info", "ℹ ")
                    self.log(f"ℹ {p[0]}")
                elif kind == "ans_start":
                    self.answer_box.delete("1.0", "end")
                elif kind == "ans_delta":
                    self.answer_box.insert("end", p[0])
                    self.answer_box.see("end")
                elif kind == "ans_end":
                    text, dt = p
                    self.add_line(f"{text}  ({dt:.1f} с)", "answer", "💬 ")
                    self.log(f"💬 ({dt:.1f}s) {text}")
        except queue.Empty:
            pass
        if self.thinking_since is not None:
            self.status_var.set(f"Думаю над відповіддю… {time.monotonic() - self.thinking_since:.0f} с")
        self.root.after(80, self.poll)

    def toggle_pause(self):
        if not self.capture:
            return
        if self.capture.paused.is_set():
            self.capture.paused.clear()
            self.pause_btn.config(text="Пауза")
            self.post("status", "Слухаю…")
        else:
            self.capture.paused.set()
            self.pause_btn.config(text="Продовжити")
            self.post("status", "Пауза")

    def reload_knowledge(self):
        if not self.answerer:
            return
        n = self.answerer.reload()
        self.post("info", f"Знання: профіль {'є' if self.answerer.profile else 'немає'}, "
                          f"матеріалів — {n}" + (": " + ", ".join(d[0] for d in self.answerer.docs) if n else "."))
        for e in self.answerer.load_errors:
            self.post("info", "Не прочитав " + e)

    def force_answer(self):
        if not self.answerer:
            return
        with self.hist_lock:
            hist = list(self.history)
        while hist and hist[-1].startswith("[твоя підказка]"):
            hist.pop()           # «Відповісти зараз» — на останню почуту фразу, а не на власну підказку
        if not hist:
            self.post("info", "Ще нічого не почуто.")
            return
        self.answer_q.put((hist, True))

    def ask_manual(self):
        q = self.ask_var.get().strip()
        if not q or not self.answerer:
            return
        self.ask_var.set("")
        with self.hist_lock:
            self.history.append(f"[питання вручну] {q}")
            hist = list(self.history)
        self.post("mine", q)
        self.answer_q.put((hist, True))

    # ---- потоки
    def answer_loop(self):
        while True:
            hist, force = self.answer_q.get()
            try:  # якщо поки думали, прийшли новіші запити — беремо лише останній
                while True:
                    hist, force = self.answer_q.get_nowait()
            except queue.Empty:
                pass
            if not self.answerer.client:
                self.post("info", "⚠ Немає API-ключа: впишіть його в config.json → \"keys\" і перезапустіть.")
                continue
            t0 = time.monotonic()
            self.thinking_since = t0
            buf, started = "", False
            try:
                for chunk in self.answerer.stream(hist, force):
                    buf += chunk
                    if not started:
                        head = buf.strip().upper()
                        if not head or "NONE".startswith(head) or head.rstrip(". ") == "NONE":
                            continue       # ще може виявитись «NONE»
                        started = True
                        self.thinking_since = None
                        self.post("ans_start")
                        self.post("ans_delta", buf.lstrip())
                    else:
                        self.post("ans_delta", chunk)
                if not started and buf.strip() and buf.strip().upper().rstrip(". ") != "NONE":
                    started = True
                    self.post("ans_start")
                    self.post("ans_delta", buf.strip())
                if started:
                    self.post("ans_end", buf.strip(), time.monotonic() - t0)
                    with self.hist_lock:
                        self.history.append(f"[твоя підказка] {buf.strip()}")
            except Exception as e:  # noqa: BLE001
                self.post("info", friendly_llm_error(e))
            finally:
                self.thinking_since = None
                self.post("status", "Слухаю…")

    def transcribe_loop(self, tr):
        pending, pending_lang, t_first = "", "", 0.0   # незакінчене речення, що чекає продовження
        last_item = time.monotonic()
        gap = float(self.cfg.get("merge_gap_seconds") or 2.0)
        while True:
            try:
                items = [self.phrase_q.get(timeout=0.3)]
            except queue.Empty:
                items = []
            try:
                while True:
                    items.append(self.phrase_q.get_nowait())
            except queue.Empty:
                pass
            now = time.monotonic()
            if items:
                last_item = now
            elif pending and now - last_item > gap:     # довго тиша — віддаємо, що є
                self.finish_phrase(pending, pending_lang, now - t_first)
                pending = ""
                continue
            if not items:
                continue
            finals = [a for k, a in items if k == "final"]
            live = items[-1][1] if items[-1][0] == "partial" else None

            if finals:
                audio = np.concatenate(finals) if len(finals) > 1 else finals[0]
                t0 = time.monotonic()
                try:
                    self.post("status", "Розпізнаю…")
                    text, lang = tr.transcribe(audio)
                except Exception as e:  # noqa: BLE001
                    self.post("info", f"Помилка розпізнавання: {e!r}")
                    text, lang = "", ""
                finally:
                    if self.thinking_since is None:
                        self.post("status", "Слухаю…")
                if text and not is_hallucination(text):
                    if not pending:
                        t_first = t0
                    pending = (pending + " " + text).strip() if pending else text
                    pending_lang = lang
                    if sentence_done(pending) or len(pending) > 400:
                        self.finish_phrase(pending, pending_lang, time.monotonic() - t0)
                        pending = ""
                    else:
                        self.post("partial", pending, pending_lang)
                elif not pending:
                    self.post("live_off")

            if live is not None and len(live) >= SR * 0.8:
                try:
                    text, lang = tr.transcribe(live, partial=True)
                except Exception:  # noqa: BLE001
                    continue
                if text and not is_hallucination(text):
                    self.post("partial", (pending + " " + text).strip(), lang)

    def finish_phrase(self, text, lang, dt):
        with self.hist_lock:
            self.history.append(text)
            hist = list(self.history)
        self.post("heard", text, lang, dt)
        mode = self.mode
        if mode == "all" or (mode == "questions" and looks_like_question(text)):
            self.answer_q.put((hist, False))

    def worker(self):
        com_init()
        # 1) звук
        if sc is None:
            self.post("info", f"Запис звуку не працює: {SC_ERROR!r}")
        else:
            try:
                names = [sp.name for sp in sc.all_speakers()]
                self.post("info", "Пристрої виводу: " + "; ".join(names)
                          + f"\nЗа замовчуванням: {sc.default_speaker().name}")
            except Exception as e:  # noqa: BLE001
                self.post("info", f"Не вдалося отримати список пристроїв: {e!r}")
        self.capture = LoopbackCapture(self.audio_q, lambda s: self.post("status", s),
                                       lambda s: self.post("info", s), self.cfg.get("output_device", ""))
        self.capture.start()

        # 2) відповіді
        try:
            self.answerer = Answerer(self.cfg, notify=lambda m: self.post("info", m))
        except Exception as e:  # noqa: BLE001
            self.post("info", f"Не вдалося підключити {self.cfg['provider']}: {e!r}")
        if self.answerer:
            self.reload_knowledge()
            if not self.answerer.client:
                self.post("info", "API-ключа немає: розпізнавання працює, відповіді — ні. Безкоштовний ключ: "
                                  "console.groq.com/keys → впишіть у config.json → \"keys\" → \"groq\" "
                                  "і перезапустіть.")
            threading.Thread(target=self.answer_loop, daemon=True).start()

        # 3) Whisper
        self.post("status", "Завантажую Whisper…")
        try:
            tr = Transcriber(self.cfg, self.post)
        except Exception as e:  # noqa: BLE001
            self.post("info", f"Whisper не запустився: {e!r}")
            return
        where = {"cuda": "відеокарті", "cloud": "сервері Groq"}.get(tr.device, "процесорі")
        threading.Thread(target=self.transcribe_loop, args=(tr,), daemon=True).start()
        self.post("info", f"Готово: Whisper «{tr.name}» на {where}. Слухаю звук ПК — "
                          "справа внизу рівень гучності.")
        self.post("status", "Слухаю…")

        # 4) нарізка звуку на фрази (у реальному часі, не чекає на Whisper)
        try:  # звук, що накопичився під час завантаження моделі, не потрібен
            while True:
                self.audio_q.get_nowait()
        except queue.Empty:
            pass
        seg = Segmenter(self.cfg)
        # живий текст: у хмарі за замовчуванням вимкнено (безкоштовний ліміт Groq — 20 запитів/хв)
        raw_every = self.cfg.get("partial_seconds")
        every = float(raw_every) if raw_every else {"cuda": 1.2, "cpu": 3.0}.get(tr.device, 0.0)
        silence = np.zeros(BLOCK, dtype=np.float32)
        n, since_partial = 0, 0
        while True:
            try:
                block = self.audio_q.get(timeout=0.15)
            except queue.Empty:
                # коли нічого не грає (пауза у відео, пауза в програмі), Windows не дає навіть тиші —
                # підставляємо її самі, інакше фраза ніколи не закінчиться
                block = silence
            rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
            n += 1
            if n % 3 == 0:
                bars = min(10, int(rms / 0.01))
                self.post("level", "рівень " + "▮" * bars + "·" * (10 - bars))
            phrase = seg.feed(block, rms)
            if phrase is not None:
                self.phrase_q.put(("final", phrase))
                since_partial = 0
            elif seg.active and every > 0:
                since_partial += 1
                if since_partial >= every * 10:
                    since_partial = 0
                    cur = seg.current()
                    if cur is not None:
                        self.phrase_q.put(("partial", cur))
            else:
                since_partial = 0

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
