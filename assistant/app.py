"""Вікно програми (tkinter): відповіді, почуте, вибір моделі, кнопки."""
import queue
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime

from .config import HERE, MODEL_CHOICES, key_for, load_config, pick_fallback, save_model_choice
from .llm import Answerer
from .pipeline import Pipeline
from .settings import SettingsWindow
from .theme import ACCENT, BG, FG, MUTED, WARN




class App(Pipeline):
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
        self.settings_win = None
        self.build_ui()
        if not any((self.cfg.get("keys") or {}).values()) and not self.cfg.get("api_key") \
                and self.cfg["provider"] != "ollama":
            self.root.after(400, lambda: self.open_settings(first_run=True))
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
        tk.Button(mrow, text="⚙ Налаштування", command=self.open_settings, **btn).pack(side="right")

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
            self.add_line(f"Немає ключа для {prov}: вставте його в «⚙ Налаштування».", "info")
            self.root.after(0, lambda: self.model_var.set(self.current_choice_label()))
            return
        keys = self.cfg.setdefault("keys", {})
        if self.cfg.get("api_key") and not keys.get(self.cfg["provider"]):
            keys[self.cfg["provider"]] = self.cfg["api_key"]
        fb = pick_fallback(self.cfg, prov)
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

    def open_settings(self, first_run=False):
        if self.settings_win is not None:
            try:
                self.settings_win.win.lift()
                return
            except tk.TclError:
                pass
        self.settings_win = SettingsWindow(self.root, on_save=self.apply_settings, first_run=first_run)

    def apply_settings(self, _raw):
        """Застосувати збережені налаштування без перезапуску (крім звуку й способу розпізнавання)."""
        self.cfg.clear()
        self.cfg.update(load_config())     # той самий об'єкт — мова й терміни Whisper оновляться одразу
        self.mode_var.set(self.cfg["answer_mode"])
        if self.answerer is not None:
            try:
                self.answerer = Answerer(self.cfg, notify=lambda m: self.post("info", m))
            except Exception as e:  # noqa: BLE001
                self.add_line(f"Не вдалося підключити {self.cfg['provider']}: {e!r}", "info")
        self.model_var.set(self.current_choice_label())
        self.fb_var.set(self.fallback_text())
        self.add_line("Налаштування збережено.", "info")

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

    def run(self):
        self.root.mainloop()
