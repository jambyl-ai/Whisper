"""Фонові потоки: звук → фрази → розпізнавання → репліки → відповіді."""
import queue
import threading
import time

import numpy as np

from .audio import BLOCK, LoopbackCapture, SR, Segmenter
from .llm import Answerer, friendly_llm_error
from .system import SC_ERROR, com_init, sc
from .textproc import is_hallucination, looks_like_question, looks_unfinished
from .transcriber import Transcriber


class Pipeline:
    """Домішка до App: потоки обробки. Спілкується з вікном лише через self.post()."""

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

    def turn_gap(self, text, rate):
        """Скільки тиші чекати, перш ніж вважати репліку закінченою."""
        cfg = self.cfg
        gap = float(cfg.get("unfinished_gap_seconds") or 2.5) if looks_unfinished(text) \
            else float(cfg.get("turn_gap_seconds") or 1.2)
        slow = float(cfg.get("slow_speech_wps") or 1.8)
        if rate and rate < slow:            # повільний мовець робить довгі паузи посеред речення
            gap *= min(1.8, slow / rate)
        return gap

    def transcribe_loop(self, tr):
        """Фрази (шматки між короткими паузами) склеюються в одну репліку, доки мовець не замовкне
        по-справжньому. Лише тоді репліка йде в історію і, якщо це питання, — на відповідь."""
        pending, pending_lang, last_dt = "", "", 0.0
        speaking, last_end, rate = False, time.monotonic(), None
        silence_s = float(self.cfg.get("silence_seconds") or 0.6)
        while True:
            try:
                items = [self.phrase_q.get(timeout=0.1)]
            except queue.Empty:
                items = []
            try:
                while True:
                    items.append(self.phrase_q.get_nowait())
            except queue.Empty:
                pass
            for kind, _a, t in items:
                if kind == "start":
                    speaking = True
                elif kind in ("final", "end"):
                    speaking, last_end = False, t
            finals = [a for k, a, _t in items if k == "final"]
            live = items[-1][1] if items and items[-1][0] == "partial" else None

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
                last_dt = time.monotonic() - t0
                if text and not is_hallucination(text):
                    speech_s = max(0.6, len(audio) / SR - silence_s - 0.3)
                    r = len(text.split()) / speech_s
                    rate = r if rate is None else 0.6 * rate + 0.4 * r
                    pending = (pending + " " + text).strip() if pending else text
                    pending_lang = lang
                    self.post("partial", pending, pending_lang)
                elif not pending:
                    self.post("live_off")

            if live is not None and len(live) >= SR * 0.8:
                try:
                    text, lang = tr.transcribe(live, partial=True)
                except Exception:  # noqa: BLE001
                    text = ""
                if text and not is_hallucination(text):
                    self.post("partial", (pending + " " + text).strip(), lang)

            if pending and not speaking and (time.monotonic() - last_end >= self.turn_gap(pending, rate)
                                              or len(pending) > 600):
                self.finish_phrase(pending, pending_lang, last_dt)
                pending = ""

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
        silence_s = float(self.cfg.get("silence_seconds") or 0.6)
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
            was_active = seg.active
            phrase = seg.feed(block, rms)
            now = time.monotonic()
            if seg.active and not was_active:
                self.phrase_q.put(("start", None, now))
            if phrase is not None:
                self.phrase_q.put(("final", phrase, now - silence_s))
                since_partial = 0
            elif was_active and not seg.active:            # надто короткий звук — фразу відкинуто
                self.phrase_q.put(("end", None, now - silence_s))
                since_partial = 0
            elif seg.active and every > 0:
                since_partial += 1
                if since_partial >= every * 10:
                    since_partial = 0
                    cur = seg.current()
                    if cur is not None:
                        self.phrase_q.put(("partial", cur, now))
            else:
                since_partial = 0
