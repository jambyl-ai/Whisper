"""Відповіді: Gemini (рідний API), OpenAI-сумісні сервіси, Anthropic; запасна модель."""
import json
import socket
import time

from .config import OPENAI_COMPAT_BASES, key_for
from .knowledge import load_profile_and_knowledge


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
