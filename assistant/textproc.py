"""Аналіз тексту: схоже на питання, обірване речення, галюцинації Whisper."""
import re


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

def looks_like_question(text):
    t = text.strip().lower()
    if "?" in t:
        return True
    words = re.findall(r"[\w'’-]+", t)
    return bool(words) and words[0] in QUESTION_WORDS


CONTINUATION_WORDS = {
    "and", "or", "but", "the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "about", "from",
    "because", "so", "that", "which", "who", "is", "are", "was", "were", "my", "your", "his", "her",
    "their", "our", "like", "if", "when", "than", "then", "um", "uh", "erm",
    "і", "й", "та", "а", "але", "що", "як", "в", "у", "на", "з", "із", "до", "про", "для", "бо", "тому",
    "щоб", "який", "яка", "які", "коли", "якщо", "ну", "е", "еее",
    "и", "но", "что", "с", "для", "потому", "чтобы", "который", "когда", "если"}


def looks_unfinished(text):
    """Речення, схоже, обірване: немає кінцевого розділового знака, «…», кома чи службове слово в кінці."""
    t = text.rstrip(' "»”)\'')
    if t.endswith(("...", "…", ",", "-", "—", ":")) or not t.endswith((".", "!", "?", "。")):
        return True
    if t.endswith("?"):       # «Where are you from?» — питання з прийменником у кінці закінчене
        return False
    words = re.findall(r"[\w'’-]+", t.lower())
    return bool(words) and words[-1] in CONTINUATION_WORDS


def sentence_done(text):
    """Схоже на завершене речення чи питання."""
    t = text.rstrip(' "»”)\'')
    return t.endswith((".", "!", "?", "…", "。")) or looks_like_question(text)


def is_hallucination(text):
    t = text.strip().lower()
    if len(t) < 2 or t.strip(" .!") in ("you", "thank you", "дякую", "bye", "спасибо"):
        return True
    return any(h in t for h in HALLUCINATIONS)
