"""Windows-специфічне: IPv4 першим, DLL бібліотек NVIDIA, soundcard для Python 3.12+, COM."""
import ctypes
import os
import socket
import sys
import types
from pathlib import Path


# Спершу IPv4: якщо IPv6 у мережі «зламаний», кожна спроба з'єднання інакше
# висить до тайм-ауту, і відповідь приходить через хвилини.
_orig_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_ipv4_first(*args, **kwargs):
    res = _orig_getaddrinfo(*args, **kwargs)
    return sorted(res, key=lambda r: 0 if r[0] == socket.AF_INET else 1)


socket.getaddrinfo = _getaddrinfo_ipv4_first


def setup_cuda_dlls():
    """Додати теки з DLL бібліотек NVIDIA (pip-пакети nvidia-*) у пошук Windows."""
    found = []
    if sys.platform != "win32":
        return found
    for sp in {p for p in sys.path if p and "site-packages" in p}:
        base = Path(sp) / "nvidia"
        if not base.is_dir():
            continue
        for b in sorted(base.glob("*/bin")):
            try:
                os.add_dll_directory(str(b))
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
            found.append(b.parent.name)
    return found


CUDA_LIBS = setup_cuda_dlls()


def _load_soundcard():
    """Імпорт soundcard з виправленням для Python 3.12+ на Windows.

    soundcard при імпорті викликає platform.win32_ver(), який у нових Python сам
    ініціалізує COM, після чого CoInitializeEx повертає S_FALSE (1) — «вже
    ініціалізовано». Це код успіху, але soundcard вважає його помилкою
    («Error 0x100000001»). Тут модуль завантажується з одним виправленням:
    S_FALSE приймається як успіх.
    """
    if sys.platform != "win32":
        import soundcard
        return soundcard
    import importlib.util
    spec = importlib.util.find_spec("soundcard")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("бібліотеку soundcard не встановлено (запустіть 1_встановити.bat)")
    pkg_dir = Path(list(spec.submodule_search_locations)[0])
    src = (pkg_dir / "mediafoundation.py").read_text(encoding="utf-8")
    marker = "self.check_error(hr)"          # перша поява — у _COMLibrary.__init__
    if marker not in src:                     # інша версія soundcard — звичайний імпорт
        import soundcard
        return soundcard
    src = src.replace(marker, "(hr == 1) or self.check_error(hr)", 1)

    pkg = types.ModuleType("soundcard")
    pkg.__path__ = [str(pkg_dir)]
    pkg.__file__ = str(pkg_dir / "__init__.py")
    pkg.__package__ = "soundcard"
    mf = types.ModuleType("soundcard.mediafoundation")
    mf.__file__ = str(pkg_dir / "mediafoundation.py")
    mf.__package__ = "soundcard"
    sys.modules["soundcard"] = pkg
    sys.modules["soundcard.mediafoundation"] = mf
    try:
        exec(compile(src, mf.__file__, "exec"), mf.__dict__)
    except Exception:
        sys.modules.pop("soundcard", None)
        sys.modules.pop("soundcard.mediafoundation", None)
        raise
    for name in dir(mf):
        if not name.startswith("_"):
            setattr(pkg, name, getattr(mf, name))
    pkg.mediafoundation = mf
    return pkg


try:
    sc = _load_soundcard()
    SC_ERROR = None
except Exception as _e:  # noqa: BLE001
    sc = None
    SC_ERROR = _e


def com_init():
    """COM у фоновому потоці (той самий багатопотоковий режим, що й у soundcard)."""
    if sys.platform == "win32":
        try:
            ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        except Exception:
            pass
