"""Відповіді: Gemini (рідний API), OpenAI-сумісні сервіси, Anthropic; запасна модель."""
import json
import math
import re
import socket
import time

from .config import OPENAI_COMPAT_BASES, key_for
from .knowledge import load_profile_and_knowledge

# Скільки токенів можна надіслати одним запитом. Безкоштовний Groq рахує ліміт за хвилину
# (напр. 8000 TPM для gpt-oss-120b, сюди входить і max_tokens відповіді), тож туди йде лише
# найпотрібніше. None — без обмеження (Gemini, Claude: великі ліміти).
DEFAULT_PROMPT_BUDGET = {"groq": 5000, "ollama": 6000}


def est_tokens(text):
    """Груба оцінка з запасом: ~3 символи на токен (кирилиця «дорожча» за латиницю)."""
    return len(text) // 3 + 1


_WORD = re.compile(r"[^\W_]{2,}", re.U)
_STOP = set("""
the and what does did do is are was were be been you your yours it its of to in on for with this that these those
how why when where who which can could would will shall should about me my mine we our they them their he him his
she her an or at by as from have has had not no yes ok so if then than there here also just very please tell say
що як це та ти ви він вона воно вони мене мені мій моя твій ваш чи по за до від для про але або так ні теж там тут
що також коли де хто який яка яке які чому навіщо
что как это ты вы он она оно они меня мне мой моя твой ваш ли по за до от для про но или да нет тоже там тут
также когда где кто какой какая какое какие почему зачем
""".split())


def _words(text):
    return {w.lower() for w in _WORD.findall(text)} - _STOP


def chunk_text(name, text, size=900):
    """Розбити документ на уривки ~size символів по абзацах."""
    parts = [x.strip() for x in re.split(r"\n+", text) if x.strip()]
    chunks, cur = [], ""
    for para in parts:
        while len(para) > size:                      # дуже довгий абзац
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(para[:size])
            para = para[size:]
        if cur and len(cur) + len(para) + 1 > size:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n{para}" if cur else para
    if cur:
        chunks.append(cur)
    return [(name, i, c) for i, c in enumerate(chunks)]


def select_chunks(docs, query, budget, last=""):
    """Уривки з матеріалів, найближчі до питання, що вміщаються в budget токенів (у порядку тексту).

    Оцінка — збіг слів з питанням, рідкісні слова важать більше (IDF), слова останньої
    фрази — утричі. Повертає (уривки, чи довелось щось відкинути).
    """
    chunks = [c for name, text in docs for c in chunk_text(name, text)]
    if not chunks or budget <= 0:
        return [], bool(chunks)
    sizes = [est_tokens(c[2]) + 8 for c in chunks]
    if sum(sizes) <= budget:
        return chunks, False
    cw = [_words(c[2]) for c in chunks]
    df = {}
    for ws in cw:
        for w in ws:
            df[w] = df.get(w, 0) + 1
    n = len(chunks)
    qlast, qall = _words(last), _words(query)
    q = qlast | qall

    def score(i):
        return sum(math.log((n + 1) / (df[w] + 0.5)) * (3 if w in qlast else 1) for w in q & cw[i])

    ranked = sorted(range(n), key=lambda i: (-score(i), i))
    picked, used = [], 0
    for i in ranked:
        if used + sizes[i] <= budget:
            picked.append(i)
            used += sizes[i]
    return [chunks[i] for i in sorted(picked)], True


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

    def system_prompt(self, query="", budget=None, last=""):
        name = (self.cfg.get("user_name") or "").strip()
        who = f"користувача, якого звати {name}" if name else "користувача"
        s = (f"Ти — підказувач для {who}. "
             "Ти чуєш розмову, що звучить на його комп'ютері (пара в Zoom, дзвінок, відео): тобі дають "
             "розшифровку останніх фраз (Whisper, можливі помилки розпізнавання — враховуй схожі за "
             "звучанням слова). Склади відповідь на ОСТАННЄ питання так, щоб користувач міг одразу сказати "
             "її вголос.\n"
             "ГОЛОВНЕ: ти пишеш репліку, яку скаже ЛЮДИНА — користувач. Питання ставлять йому, а не тобі. "
             "Ніколи не згадуй, що ти ШІ, модель, асистент чи програма («As an AI…», «I'm a language model…», "
             "«Як штучний інтелект…»). Не відмовляйся і не читай моралі — просто дай природну людську відповідь.\n"
             "Правила:\n"
             "0. Спершу визнач, ПРО КОГО питання.\n"
             "   • Про самого користувача (you / your / ти / твій / вы: «Where are you from?», «What do you "
             "think about…?», «Чим ти займаєшся?») — відповідай від його імені, від першої особи: «I…», «Я…».\n"
             "   • Про річ, поняття, факт, термін, людину чи подію («What is X?», «How does X work?», «Що таке X?», "
             "«Хто такий X?») — відповідай звичайним реченням про цю річ, у третій особі: «X is…», «X — це…». "
             "НІКОЛИ не пиши так, ніби користувач сам є цією річчю (не «I am a condition…», а «It is a "
             "condition…»).\n"
             "   • Якщо просять думку про річ («What do you think about X?») — «I think X is…».\n"
             "1. Спершу шукай відповідь у МАТЕРІАЛАХ нижче (його тексти, роботи, конспекти). Якщо питання "
             "про текст чи тему роботи — відповідай за текстом, його словами й термінами.\n"
             "2. Питання про нього самого (навчання, спеціальність, досвід, плани) — бери з ПРОФІЛЮ. "
             "Не вигадуй фактів про нього: якщо в профілі немає потрібного, дай нейтральну правдоподібну "
             "відповідь і познач її в кінці «(уточни)». На особисті, провокаційні чи жартівливі питання "
             "(стосунки, орієнтація, релігія, політика, «ти дурний?») — спокійна, ввічлива людська відповідь "
             "або м'яко ухилитись («That's a bit personal — I'd rather keep it private.»), без вигаданих фактів.\n"
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
        if budget is None:
            if self.profile:
                s += "\n\n=== ПРОФІЛЬ (про користувача) ===\n" + self.profile
            for name, text in self.docs:
                s += f"\n\n=== МАТЕРІАЛ: {name} ===\n{text}"
            return s
        left = budget - est_tokens(s)
        if self.profile and left > 0:
            prof = self.profile
            cap = max(300, left // 3)
            if est_tokens(prof) > cap:
                prof = prof[:cap * 3].rsplit("\n", 1)[0] + "\n…"
            s += "\n\n=== ПРОФІЛЬ (про користувача) ===\n" + prof
            left = budget - est_tokens(s)
        picked, cut = select_chunks(self.docs, query, left - 60, last)
        if picked:
            s += "\n\n=== МАТЕРІАЛИ ===" + ("\n(через ліміт сервера — лише уривки, найближчі до питання)" if cut else "")
            last = None
            for name, _i, text in picked:
                if name != last:
                    s += f"\n\n--- {name} ---"
                    last = name
                s += "\n" + text
        return s

    @staticmethod
    def budget_for(cfg):
        b = cfg.get("prompt_budget_tokens")
        if isinstance(b, dict):
            b = b.get(cfg.get("provider"))
        if b in (None, "", 0):
            b = DEFAULT_PROMPT_BUDGET.get(cfg.get("provider"))
        return int(b) if b else None

    @staticmethod
    def user_part(history, force=False, budget=None):
        """Історія для моделі; з budget — найновіші фрази, що вміщаються."""
        last, prev = history[-1], list(history[:-1])
        if budget is not None:
            last = last[-1500:]
            used, keep = est_tokens(last) + 80, []
            for h in reversed(prev):
                h = h if len(h) <= 400 else h[:400] + "…"
                t = est_tokens(h) + 2
                if used + t > budget:
                    break
                keep.append(h)
                used += t
            prev = keep[::-1]
        convo = "\n".join(f"- {h}" for h in prev) or "(немає)"
        user = f"Попередні фрази:\n{convo}\n\nОстання фраза:\n{last}"
        if force:
            user += "\n\nКористувач сам попросив відповідь: дай найкращу відповідь або коментар, не NONE."
        return user

    def prompt_for(self, pair, history, force=False, shrink=1):
        budget = self.budget_for(pair[1])
        if budget is not None:
            budget = max(1500, int(budget / shrink))
        user = self.user_part(history, force, None if budget is None else max(500, budget // 4))
        query = " ".join(history[-4:])
        system = self.system_prompt(query=query, budget=None if budget is None else budget - est_tokens(user),
                                    last=history[-1])
        return system, user

    def stream(self, history, force=False):
        """Генератор шматків відповіді. Кожен сервер отримує запит під свій ліміт."""
        first = self.main or self.backup
        got = False
        shrink = 1
        while True:
            system, user = self.prompt_for(first, history, force, shrink)
            try:
                for t in self._stream(first, system, user):
                    got = True
                    yield t
                return
            except Exception as e:  # noqa: BLE001
                too_big = "413" in str(e) or "too large" in str(e).lower()
                if not got and too_big and shrink < 4 and (not self.backup or first is self.backup):
                    shrink *= 2            # запасного немає — пробуємо коротший запит
                    continue
                if got or not self.backup or first is self.backup:
                    raise
                why = str(e).replace("\n", " ")[:220]
                break
        self.notify(f"Основний сервер ({first[1]['model']}) не відповів — відповідаю через запасний "
                    f"({self.backup[1]['model']}). Причина: {why}")
        system, user = self.prompt_for(self.backup, history, force)
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
