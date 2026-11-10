"""Вікно налаштувань: усе важливе з config.json — без ручного редагування JSON.

Відкривається кнопкою «⚙ Налаштування», сама при першому запуску (немає ключа)
або окремо: 4_налаштування.bat (python whisper_assistant.py --setup).
"""
import json
import os
import shutil
import sys
import threading
import tkinter as tk
import urllib.error
import urllib.request
import webbrowser

from .config import HERE, MODEL_CHOICES, pick_fallback
from .system import sc
from .theme import ACCENT, BG, ERROR, FG, FIELD, HOVER, MUTED, WARN

KEY_SERVICES = [
    ("groq", "Groq", "https://console.groq.com/keys",
     "безкоштовно, без картки. Головний ключ: і відповіді, і розпізнавання в хмарі"),
    ("gemini", "Gemini", "https://aistudio.google.com/apikey", "безкоштовно — запасна модель"),
    ("openai", "OpenAI", "https://platform.openai.com/api-keys", "платно, не обов'язково"),
]
CHECK_URLS = {
    "groq": ("https://api.groq.com/openai/v1/models", "Authorization", "Bearer {}"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1", "x-goog-api-key", "{}"),
    "openai": ("https://api.openai.com/v1/models", "Authorization", "Bearer {}"),
}
LANGS = [("Автоматично (англійська / українська / російська)", None),
         ("Тільки англійська — найточніше на парах англійської", "en"),
         ("Тільки українська", "uk")]
MODES = [("Лише на питання (рекомендовано)", "questions"), ("На кожну фразу", "all"),
         ("Тільки кнопкою «Відповісти зараз»", "manual")]
BACKENDS = [("Автоматично (рекомендовано)", "auto"), ("Тільки на цьому комп'ютері", "local"),
            ("Завжди в хмарі Groq", "groq")]
SPEEDS = [("Швидко — відповідати одразу після паузи", 0.9, 1.8),
          ("Нормально", 1.2, 2.5),
          ("Чекати довше — співрозмовник говорить повільно", 1.8, 3.5)]
DEFAULT_DEVICE = "Пристрій Windows за замовчуванням"


# ------------------------------------------------------------------ логіка (без вікна, тестується)
def config_path():
    return HERE / "config.json"


def read_raw(path=None):
    """config.json як є (з коментарями «_…»); якщо його ще немає — з config.example.json."""
    path = path or config_path()
    for p in (path, path.with_name("config.example.json")):
        if p.exists():
            with open(p, encoding="utf-8-sig") as f:
                return json.load(f)
    return {}


def save_raw(raw, path=None):
    with open(path or config_path(), "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)


def apply_values(raw, v):
    """Перенести значення з форми в config.json, решту полів і коментарі не чіпати."""
    raw = dict(raw)
    raw["user_name"] = v["user_name"].strip()
    keys = dict(raw.get("keys") or {})
    for prov, *_ in KEY_SERVICES:
        keys[prov] = (v["keys"].get(prov) or "").strip()
    raw["keys"] = keys
    raw["api_key"] = ""                      # ключі тепер лише в «keys»
    raw["provider"], raw["model"] = v["provider"], v["model"]
    raw["fallback"] = pick_fallback({"keys": keys}, v["provider"])
    raw["language"] = v["language"]
    raw["answer_mode"] = v["answer_mode"]
    raw["answer_instructions"] = v["answer_instructions"].strip()
    raw["whisper_prompt"] = v["whisper_prompt"].strip()
    raw["whisper_backend"] = v["whisper_backend"]
    raw["output_device"] = v["output_device"]
    raw["turn_gap_seconds"], raw["unfinished_gap_seconds"] = v["turn_gaps"]
    return raw


def check_key(provider, key, opener=urllib.request.urlopen):
    """(успіх, повідомлення) — простий запит списку моделей."""
    if not key.strip():
        return False, "Спершу вставте ключ."
    url, header, fmt = CHECK_URLS[provider]
    req = urllib.request.Request(url, headers={header: fmt.format(key.strip())})
    try:
        with opener(req, timeout=12):
            return True, "✓ Ключ працює"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, "✗ Ключ не підходить — скопіюйте його ще раз"
        if e.code == 429:
            return True, "✓ Ключ правильний (зараз ліміт запитів — це нормально)"
        return False, f"✗ Сервер відповів помилкою {e.code}"
    except Exception:  # noqa: BLE001
        return False, "✗ Немає зв'язку із сервером (інтернет?)"


def speed_index(raw):
    gap = float(raw.get("turn_gap_seconds") or 1.2)
    return min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i][1] - gap))


def speaker_names():
    try:
        return [s.name for s in sc.all_speakers()] if sc else []
    except Exception:  # noqa: BLE001
        return []


def open_path(path):
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606 — відкрити у Провіднику / Блокноті
    else:
        webbrowser.open(path.as_uri())


# ------------------------------------------------------------------ вікно
class SettingsWindow:
    def __init__(self, master=None, on_save=None, first_run=False):
        self.on_save, self.standalone = on_save, master is None
        self.win = tk.Tk() if self.standalone else tk.Toplevel(master)
        self.win.title("Whisper-асистент — налаштування")
        self.win.configure(bg=BG)
        self.win.geometry("700x720+80+40")
        self.win.minsize(560, 400)
        if not self.standalone:
            self.win.transient(master)
            self.win.attributes("-topmost", True)
        self.raw = read_raw()
        self.key_vars, self.key_status, self.key_entries = {}, {}, []
        self._build(first_run)

    # ---- дрібні помічники для однакового вигляду
    def _label(self, parent, text, row, hint=None):
        tk.Label(parent, text=text, bg=BG, fg=FG, font=("Segoe UI", 10), anchor="w").grid(
            row=row, column=0, sticky="nw", padx=(0, 10), pady=(8, 0))
        if hint:
            tk.Label(parent, text=hint, bg=BG, fg=MUTED, font=("Segoe UI", 8), anchor="w", justify="left",
                     wraplength=430).grid(row=row + 1, column=1, columnspan=2, sticky="w")

    def _entry(self, parent, var, row, show=None):
        e = tk.Entry(parent, textvariable=var, bg=FIELD, fg=FG, insertbackground=FG, relief="flat",
                     font=("Segoe UI", 10), show=show or "")
        e.grid(row=row, column=1, sticky="ew", pady=(8, 0), ipady=3)
        return e

    def _menu(self, parent, var, options, row, colspan=2):
        m = tk.OptionMenu(parent, var, *options)
        m.configure(bg=FIELD, fg=FG, activebackground=HOVER, activeforeground=FG, relief="flat",
                    highlightthickness=0, bd=0, anchor="w")
        m["menu"].configure(bg=FIELD, fg=FG)
        m.grid(row=row, column=1, columnspan=colspan, sticky="ew", pady=(8, 0))
        return m

    def _button(self, parent, text, cmd, **grid):
        b = tk.Button(parent, text=text, command=cmd, bg=FIELD, fg=FG, activebackground=HOVER,
                      activeforeground=FG, relief="flat", padx=8, pady=2, bd=0)
        if grid:
            b.grid(**grid)
        return b

    def _section(self, parent, text, row):
        tk.Label(parent, text=text, bg=BG, fg=ACCENT, font=("Segoe UI", 11, "bold"), anchor="w").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(16, 0))

    # ---- форма
    def _build(self, first_run):
        raw = self.raw
        bottom = tk.Frame(self.win, bg=BG)
        bottom.pack(side="bottom", fill="x", padx=12, pady=10)
        self.status = tk.Label(bottom, text="", bg=BG, fg=MUTED, anchor="w", justify="left", wraplength=470)
        self.status.pack(side="left", fill="x", expand=True)
        tk.Button(bottom, text="Зберегти", command=self.save, bg="#238636", fg="white", activebackground="#2ea043",
                  activeforeground="white", relief="flat", padx=16, pady=4, bd=0,
                  font=("Segoe UI", 10, "bold")).pack(side="right")
        self._button(bottom, "Скасувати", self.close).pack(side="right", padx=6)

        canvas = tk.Canvas(self.win, bg=BG, highlightthickness=0)
        sbar = tk.Scrollbar(self.win, command=canvas.yview)
        sbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=sbar.set)
        f = tk.Frame(canvas, bg=BG, padx=14, pady=6)
        fid = canvas.create_window((0, 0), window=f, anchor="nw")
        f.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(fid, width=e.width))
        self.win.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        f.columnconfigure(1, weight=1)

        r = 0
        if first_run:
            tk.Label(f, text="Вітаю! Заповніть ім'я та вставте безкоштовний ключ Groq — і все готово.",
                     bg=BG, fg=WARN, font=("Segoe UI", 10, "bold"), anchor="w", wraplength=600,
                     justify="left").grid(row=r, column=0, columnspan=3, sticky="w")
            r += 1

        self._section(f, "Про вас", r); r += 1
        self.name_var = tk.StringVar(value=raw.get("user_name", ""))
        self._label(f, "Ваше ім'я", r, "Підказки пишуться від вашого імені. Можна з англійським написанням: "
                                         "«Іван Петренко (Ivan Petrenko)».")
        self._entry(f, self.name_var, r); r += 2
        self._label(f, "Профіль і матеріали", r,
                    "Профіль — що асистент знає про вас (навчання, плани, хобі). «Знання» — тексти, за якими "
                    "відповідати: .docx, .pdf, .txt, .md.")
        row = tk.Frame(f, bg=BG)
        row.grid(row=r, column=1, columnspan=2, sticky="w", pady=(8, 0))
        self._button(row, "Відкрити профіль", self.open_profile).pack(side="left")
        self._button(row, "Відкрити папку «знання»", self.open_knowledge).pack(side="left", padx=6)
        r += 2

        self._section(f, "Ключі", r); r += 1
        keys = raw.get("keys") or {}
        for prov, title, page, hint in KEY_SERVICES:
            var = tk.StringVar(value=keys.get(prov) or (raw.get("api_key", "") if raw.get("provider") == prov else ""))
            self.key_vars[prov] = var
            self._label(f, title, r, hint)
            self.key_entries.append(self._entry(f, var, r, show="•"))
            btns = tk.Frame(f, bg=BG)
            btns.grid(row=r, column=2, sticky="w", padx=(6, 0), pady=(8, 0))
            self._button(btns, "Отримати", lambda u=page: webbrowser.open(u)).pack(side="left")
            self._button(btns, "Перевірити", lambda p=prov: self.check(p)).pack(side="left", padx=(4, 0))
            st = tk.Label(f, text="", bg=BG, fg=MUTED, font=("Segoe UI", 8), anchor="w")
            st.grid(row=r + 2, column=1, columnspan=2, sticky="w")
            self.key_status[prov] = st
            r += 3
        self.show_keys = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="Показати ключі", variable=self.show_keys, bg=BG, fg=MUTED, selectcolor=FIELD,
                       activebackground=BG, activeforeground=FG, command=self._toggle_keys).grid(
            row=r, column=1, sticky="w"); r += 1

        self._section(f, "Відповіді", r); r += 1
        labels = [c[0] for c in MODEL_CHOICES]
        cur = next((c[0] for c in MODEL_CHOICES if (c[1], c[2]) == (raw.get("provider"), raw.get("model"))), None)
        if cur is None:
            cur = f'{raw.get("provider", "groq")} · {raw.get("model", "")}'
            labels.append(cur)
        self.model_var = tk.StringVar(value=cur)
        self._label(f, "Модель", r, "Запасна модель ставиться сама — інша безкоштовна, для якої є ключ.")
        self._menu(f, self.model_var, labels, r); r += 2
        self.mode_var = tk.StringVar(value=next((l for l, v in MODES if v == raw.get("answer_mode")), MODES[0][0]))
        self._label(f, "Коли відповідати", r)
        self._menu(f, self.mode_var, [m[0] for m in MODES], r); r += 1
        self.speed_var = tk.StringVar(value=SPEEDS[speed_index(raw)][0])
        self._label(f, "Скільки чекати кінця фрази", r,
                    "Якщо асистент відповідає на півпитання — виберіть «Чекати довше».")
        self._menu(f, self.speed_var, [s[0] for s in SPEEDS], r); r += 2
        self._label(f, "Як відповідати", r, "Напр.: «Англійською, рівень B1, коротко, під відповіддю переклад "
                                              "українською в дужках».")
        self.instr = tk.Text(f, height=3, wrap="word", bg=FIELD, fg=FG, insertbackground=FG, relief="flat",
                             font=("Segoe UI", 10))
        self.instr.insert("1.0", raw.get("answer_instructions", ""))
        self.instr.grid(row=r, column=1, columnspan=2, sticky="ew", pady=(8, 0)); r += 2

        self._section(f, "Розпізнавання мови", r); r += 1
        self.lang_var = tk.StringVar(value=next((l for l, v in LANGS if v == raw.get("language")), LANGS[0][0]))
        self._label(f, "Мова розмови", r)
        self._menu(f, self.lang_var, [l[0] for l in LANGS], r); r += 1
        self.terms_var = tk.StringVar(value=raw.get("whisper_prompt", ""))
        self._label(f, "Терміни теми", r, "Слова, які Whisper має писати правильно: назви, прізвища, терміни "
                                           "через кому.")
        self._entry(f, self.terms_var, r).grid(columnspan=2); r += 2
        self.backend_var = tk.StringVar(
            value=next((l for l, v in BACKENDS if v == raw.get("whisper_backend", "auto")), BACKENDS[0][0]))
        self._label(f, "Де розпізнавати", r, "Автоматично: відеокарта NVIDIA, якщо є; інакше хмара Groq "
                                             "(потрібен ключ Groq); інакше процесор.")
        self._menu(f, self.backend_var, [b[0] for b in BACKENDS], r); r += 2

        self._section(f, "Звук", r); r += 1
        devices = [DEFAULT_DEVICE] + speaker_names()
        cur_dev = raw.get("output_device") or ""
        match = next((d for d in devices[1:] if cur_dev and cur_dev.lower() in d.lower()), None)
        if cur_dev and not match:
            devices.append(cur_dev)
            match = cur_dev
        self.dev_var = tk.StringVar(value=match or DEFAULT_DEVICE)
        self._label(f, "Що слухати", r, "Навушники або динаміки, в яких грає розмова.")
        self._menu(f, self.dev_var, devices, r); r += 2

    # ---- дії
    def _toggle_keys(self):
        for e in self.key_entries:
            e.configure(show="" if self.show_keys.get() else "•")

    def check(self, prov):
        st = self.key_status[prov]
        st.configure(text="Перевіряю…", fg=MUTED)
        key = self.key_vars[prov].get()

        def work():
            ok, msg = check_key(prov, key)
            self.win.after(0, lambda: st.configure(text=msg, fg=ACCENT if ok else ERROR))
        threading.Thread(target=work, daemon=True).start()

    def open_profile(self):
        prof = HERE / (self.raw.get("profile_file") or "про_мене.md")
        ex = HERE / "про_мене.example.md"
        if not prof.exists() and ex.exists():
            shutil.copyfile(ex, prof)
        open_path(prof)

    def open_knowledge(self):
        d = HERE / (self.raw.get("knowledge_dir") or "знання")
        d.mkdir(exist_ok=True)
        open_path(d)

    def values(self):
        model = next((c for c in MODEL_CHOICES if c[0] == self.model_var.get()), None)
        prov, mod = (model[1], model[2]) if model else (self.raw.get("provider"), self.raw.get("model"))
        speed = next(s for s in SPEEDS if s[0] == self.speed_var.get())
        dev = self.dev_var.get()
        return {
            "user_name": self.name_var.get(),
            "keys": {p: v.get() for p, v in self.key_vars.items()},
            "provider": prov, "model": mod,
            "language": next(v for l, v in LANGS if l == self.lang_var.get()),
            "answer_mode": next(v for l, v in MODES if l == self.mode_var.get()),
            "answer_instructions": self.instr.get("1.0", "end"),
            "whisper_prompt": self.terms_var.get(),
            "whisper_backend": next(v for l, v in BACKENDS if l == self.backend_var.get()),
            "output_device": "" if dev == DEFAULT_DEVICE else dev,
            "turn_gaps": (speed[1], speed[2]),
        }

    def save(self):
        v = self.values()
        raw = apply_values(self.raw, v)
        try:
            save_raw(raw)
        except OSError as e:
            self.status.configure(text=f"Не вдалося зберегти: {e}", fg=ERROR)
            return
        self.raw = raw
        warn = ""
        if not any(raw["keys"].values()) and raw["provider"] != "ollama":
            warn = "Без жодного ключа відповіді не працюватимуть. "
        elif not raw["keys"].get(raw["provider"]):
            warn = f"Для вибраної моделі ({raw['provider']}) немає ключа — відповідатиме запасна. "
        if self.on_save:
            self.on_save(raw)
            self.status.configure(text=warn + "Збережено. Пристрій звуку й спосіб розпізнавання зміняться після "
                                              "перезапуску програми.", fg=WARN if warn else ACCENT)
            if not warn:
                self.win.after(1200, self.close)
        else:
            self.status.configure(text=warn + "Збережено. Можна запускати 2_запустити.bat.",
                                  fg=WARN if warn else ACCENT)

    def close(self):
        self.win.destroy()

    def run(self):
        self.win.mainloop()
