"""Профіль користувача і папка «знання»: читання .docx, .pdf, .txt, .md."""
import html
import re
import zipfile

from .config import HERE


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
