"""Розпізнавання мови: faster-whisper на відеокарті/процесорі або хмара Groq."""
import os
import time

import numpy as np

from .audio import SR
from .config import OPENAI_COMPAT_BASES, key_for
from .system import CUDA_LIBS


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
