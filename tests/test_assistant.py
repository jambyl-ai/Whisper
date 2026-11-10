"""Автотести без звуку, відеокарти й інтернету: усе зовнішнє підмінено заглушками.

Запуск з теки проєкту:  .venv\\Scripts\\python -m unittest discover tests
"""
import json
import queue
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:  # на сервері без графіки tkinter може не бути — вікно в тестах не створюється
    import tkinter  # noqa: F401
except ImportError:
    sys.modules["tkinter"] = types.ModuleType("tkinter")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant import app as app_mod, config, llm, textproc, transcriber  # noqa: E402
from assistant.app import App  # noqa: E402


def make_cfg(**over):
    cfg = dict(config.DEFAULT_CONFIG)
    cfg.update({"keys": {}, "fallback": {}})
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------- текст
class TextTests(unittest.TestCase):
    def test_question(self):
        self.assertTrue(textproc.looks_like_question("Where are you from"))
        self.assertTrue(textproc.looks_like_question("Ти звідки?"))
        self.assertFalse(textproc.looks_like_question("I like it."))

    def test_unfinished(self):
        self.assertTrue(textproc.looks_unfinished("What do you think about"))
        self.assertTrue(textproc.looks_unfinished("What do you think about..."))
        self.assertTrue(textproc.looks_unfinished("Tell me about the."))
        self.assertFalse(textproc.looks_unfinished("Where are you from?"))
        self.assertFalse(textproc.looks_unfinished("I like it."))

    def test_hallucination(self):
        self.assertTrue(textproc.is_hallucination("Thanks for watching!"))
        self.assertFalse(textproc.is_hallucination("Hello there"))


# ---------------------------------------------------------------- конфіг
class ConfigTests(unittest.TestCase):
    def test_first_run_copies_examples(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            shutil.copy(ROOT / "config.example.json", tmp)
            shutil.copy(ROOT / "про_мене.example.md", tmp)
            with mock.patch.object(config, "HERE", tmp):
                cfg = config.load_config()
            self.assertTrue((tmp / "config.json").exists())
            self.assertTrue((tmp / "про_мене.md").exists())
            self.assertEqual(cfg["provider"], "groq")
        finally:
            shutil.rmtree(tmp)

    def test_example_has_no_secrets(self):
        ex = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
        self.assertTrue(all(not v for v in ex["keys"].values()))
        self.assertEqual(ex["api_key"], "")
        self.assertEqual(ex["user_name"], "")

    def test_key_for(self):
        cfg = make_cfg(keys={"groq": "gsk_x"})
        self.assertEqual(config.key_for(cfg, "groq"), "gsk_x")
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(config.key_for(cfg, "gemini"), "")


# ---------------------------------------------------------------- Whisper
class FakeWhisperModel:
    def __init__(self, name, **kw):
        self.name = name

    def transcribe(self, audio, **kw):
        info = types.SimpleNamespace(language="en", all_language_probs=[("en", 0.9)])
        return iter([types.SimpleNamespace(text=" local text.")]), info


class FakeOpenAI:
    fail = False
    calls = []

    def __init__(self, **kw):
        self.audio = types.SimpleNamespace(transcriptions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        FakeOpenAI.calls.append(kw)
        if FakeOpenAI.fail:
            raise RuntimeError("Error code: 429 - rate limit")
        return types.SimpleNamespace(text=" Where are you from? ", language="english")


def fake_modules(gpus=0):
    ct = types.ModuleType("ctranslate2")
    ct.get_cuda_device_count = lambda: gpus
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = FakeWhisperModel
    oa = types.ModuleType("openai")
    oa.OpenAI = FakeOpenAI
    return {"ctranslate2": ct, "faster_whisper": fw, "openai": oa}


class TranscriberTests(unittest.TestCase):
    def setUp(self):
        FakeOpenAI.fail, FakeOpenAI.calls = False, []
        self.audio = np.zeros(16000 * 2, dtype=np.float32)

    def make(self, **over):
        with mock.patch.dict(sys.modules, fake_modules()):
            return transcriber.Transcriber(make_cfg(**over), lambda *a: None)

    def test_no_gpu_no_key_uses_cpu(self):
        self.assertEqual(self.make().device, "cpu")

    def test_no_gpu_with_groq_key_uses_cloud(self):
        tr = self.make(keys={"groq": "gsk_x"})
        self.assertEqual(tr.device, "cloud")
        with mock.patch.dict(sys.modules, fake_modules()):
            self.assertEqual(tr.transcribe(self.audio), ("Where are you from?", "en"))
        self.assertEqual(FakeOpenAI.calls[-1]["model"], "whisper-large-v3-turbo")

    def test_cloud_failure_falls_back_to_cpu_and_cools_down(self):
        tr = self.make(keys={"groq": "gsk_x"})
        FakeOpenAI.fail = True
        with mock.patch.dict(sys.modules, fake_modules()):
            self.assertEqual(tr.transcribe(self.audio)[0], "local text.")
            n = len(FakeOpenAI.calls)
            self.assertEqual(tr.transcribe(self.audio)[0], "local text.")
        self.assertEqual(len(FakeOpenAI.calls), n)   # хмару хвилину не смикаємо

    def test_forced_cpu_ignores_cloud(self):
        self.assertEqual(self.make(keys={"groq": "gsk_x"}, whisper_device="cpu").device, "cpu")


# ---------------------------------------------------------------- відповіді
class AnswererTests(unittest.TestCase):
    def test_fallback_when_main_fails(self):
        class Fail:
            def stream(self, system, user):
                raise RuntimeError("Error code: 503 - overloaded")
                yield  # noqa

        class Ok:
            seen = None

            def stream(self, system, user):
                Ok.seen = user
                yield "Because."

        notes = []
        with mock.patch.object(llm, "GeminiNative", object):
            a = llm.Answerer.__new__(llm.Answerer)
        a.notify, a.cfg = notes.append, make_cfg()
        a.system_prompt = lambda: ""
        with mock.patch.object(llm, "GeminiNative", (Fail, Ok)):
            a.main = (Fail(), {"provider": "gemini", "model": "main"})
            a.backup = (Ok(), {"provider": "gemini", "model": "backup"})
            hist = ["[питання вручну] why are you late", "[твоя підказка] Traffic.", "[питання вручну] why"]
            self.assertEqual(list(a.stream(hist)), ["Because."])
        self.assertIn("503", notes[0])
        self.assertIn("[твоя підказка] Traffic.", Ok.seen)   # контекст доходить до моделі

    def test_prompt_uses_user_name(self):
        a = llm.Answerer.__new__(llm.Answerer)
        a.profile, a.docs = "", []
        a.cfg = make_cfg(user_name="Іван")
        self.assertIn("Іван", a.system_prompt())
        a.cfg = make_cfg(user_name="")
        self.assertIn("для користувача.", a.system_prompt())


# ---------------------------------------------------------------- репліки
class PipelineTests(unittest.TestCase):
    def run_scenario(self, chunks):
        outs = iter([t for t, _ in chunks])

        class Tr:
            def transcribe(self, audio, partial=False):
                return next(outs), "en"

        a = App.__new__(App)
        a.cfg, a.mode, a.thinking_since = make_cfg(), "questions", None
        a.phrase_q, a.answer_q = queue.Queue(), queue.Queue()
        a.history, a.hist_lock = app_mod.deque(maxlen=20), threading.Lock()
        a.post = lambda *p: None
        threading.Thread(target=a.transcribe_loop, args=(Tr(),), daemon=True).start()
        for _text, (speech_s, pause_s) in chunks:
            a.phrase_q.put(("start", None, time.monotonic()))
            a.phrase_q.put(("final", np.zeros(int(16000 * (speech_s + 0.9)), np.float32), time.monotonic()))
            time.sleep(pause_s)
        asked = []
        while not a.answer_q.empty():
            asked.append(a.answer_q.get()[0][-1])
        return asked

    def test_fast_question(self):
        self.assertEqual(self.run_scenario([("Where are you from?", (1.5, 1.8))]), ["Where are you from?"])

    def test_slow_speaker_is_one_question(self):
        asked = self.run_scenario([("What do you.", (2.5, 1.4)), ("Think about.", (2.5, 1.4)),
                                   ("Your future job?", (3.0, 3.5))])
        self.assertEqual(asked, ["What do you. Think about. Your future job?"])

    def test_cut_sentence_waits(self):
        asked = self.run_scenario([("Can you tell me about", (1.5, 2.0)), ("your project?", (1.0, 2.0))])
        self.assertEqual(asked, ["Can you tell me about your project?"])


# ---------------------------------------------------------------- меню моделі
class SwitchModelTests(unittest.TestCase):
    def test_switch_and_missing_key(self):
        class Var:
            def __init__(self):
                self.v = ""

            def set(self, v):
                self.v = v

            def get(self):
                return self.v

        a = App.__new__(App)
        a.cfg = make_cfg(api_key="gsk_x", keys={"groq": "gsk_x", "gemini": "AQ.y"})
        a.answerer, a.model_var, a.fb_var = object(), Var(), Var()
        a.root = types.SimpleNamespace(after=lambda t, f: f())
        lines = []
        a.post, a.add_line = (lambda *p: None), (lambda t, tag, prefix="": lines.append(t))
        with mock.patch.object(app_mod, "Answerer", lambda cfg, notify=None: types.SimpleNamespace(cfg=dict(cfg))), \
                mock.patch.object(app_mod, "save_model_choice", lambda cfg: None):
            a.switch_model(config.MODEL_CHOICES[1][0])                 # Gemini
            self.assertEqual((a.cfg["provider"], a.cfg["fallback"]["provider"]), ("gemini", "groq"))
            a.switch_model(config.MODEL_CHOICES[2][0])                 # OpenAI без ключа
            self.assertEqual(a.cfg["provider"], "gemini")
            self.assertIn("Немає ключа", lines[-1])


if __name__ == "__main__":
    unittest.main()
