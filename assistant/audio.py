"""Звук: запис того, що грає на ПК (WASAPI loopback), і нарізка на фрази за паузами."""
import threading
import time
import warnings
from collections import deque

import numpy as np

from .system import SC_ERROR, com_init, sc


SR = 16000            # частота дискретизації для Whisper
BLOCK = 1600          # 0,1 с


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
