"""One-click installer for ArbuzDiffusion.

It runs before the WebUI environment exists, so it is restricted to the
standard library, and it does as little as possible itself: everything the
project already knows how to install (clip, open_clip, gradio,
requirements.txt, extension installers) is delegated to ``launch.py --exit``.
What it adds on top is the part that usually goes wrong:

  * a Python 3.11 of its own, downloaded into the installation folder, so the
    user's own Python is never touched;
  * every cache redirected into the installation folder, so nothing grows on
    the system drive while a 3 GB torch wheel is downloaded;
  * a reachability check of the torch index before pip is started, so an
    unreachable host is reported in a second instead of five times fifteen;
  * an offline path: drop the wheels into installer_files/wheels and no
    network is needed at all;
  * a generated webui.settings.bat, so the next start is a double-click too.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALLER_FILES = ROOT / "installer_files"
PYTHON_DIR = INSTALLER_FILES / "python"
ENV_DIR = INSTALLER_FILES / "env"
CACHE_DIR = INSTALLER_FILES / "cache"
WHEELS_DIR = INSTALLER_FILES / "wheels"
SETTINGS_FILE = ROOT / "webui.settings.bat"

# Bumped by hand; install.bat carries the same number, so a pasted log says
# immediately whether the machine ran the build we think it did.
INSTALLER_REV = "7"

PORTABLE_PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260825/"
    "cpython-3.11.16+20260825-x86_64-pc-windows-msvc-install_only.tar.gz"
)

# Where a CUDA torch can come from. The official index is always tried first;
# the rest are mirrors kept for networks that cannot reach it -- a machine
# behind a blocking provider has no other way to get a CUDA build, because
# PyPI only carries CPU wheels for Windows.
TORCH_MIRRORS = [
    "https://download.pytorch.org/whl/cu128",
    "https://mirrors.aliyun.com/pytorch-wheels/cu128",
    "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128",
    "https://mirrors.cloud.tencent.com/pytorch-wheels/cu128",
    "https://mirrors.ustc.edu.cn/pytorch-wheels/cu128",
]

FALLBACK_TORCH_INDEX = TORCH_MIRRORS[0]
FALLBACK_TORCH = "2.8.0"
FALLBACK_TORCHVISION = "0.23.0"
FALLBACK_CUDA = "128"

RECOMMENDED_FREE_BYTES = 15 * 1024 ** 3
MINIMUM_FREE_BYTES = 6 * 1024 ** 3

TEXT = {
    "title": ("Установка ArbuzDiffusion", "ArbuzDiffusion installer"),
    "folder": ("Папка установки", "Installation folder"),
    "free": ("Свободно на диске", "Free space"),
    "gpu": ("Видеокарта", "GPU"),
    "gpu.none": ("не найдена nvidia-smi", "no nvidia-smi"),
    "git.missing": (
        "git не найден -- обновление расширений будет недоступно, сама установка это не ломает",
        "git not found -- extension updates will be unavailable, the install itself is fine",
    ),
    "step": ("Шаг", "Step"),
    "python.reuse": ("Нашёл свой Python:", "Reusing the bundled Python:"),
    "python.using": ("Использую найденный Python:", "Using the Python found:"),
    "python.downloading": (
        "Скачиваю portable Python 3.11 -- это разово, и он останется внутри папки",
        "Downloading a portable Python 3.11 -- one time only, and it stays inside the folder",
    ),
    "python.failed": (
        "Не смог скачать Python. Положи распакованный Python 3.11 в installer_files\\python "
        "и запусти install.bat снова.",
        "Could not download Python. Drop an unpacked Python 3.11 into installer_files\\python "
        "and run install.bat again.",
    ),
    "venv.creating": ("Создаю виртуальное окружение", "Creating the virtual environment"),
    "venv.reuse_torch": (
        "окружение видит пакеты твоего Python, поэтому torch качать не придётся",
        "the environment can see your Python's packages, so torch will not be downloaded",
    ),
    "venv.ready": ("Окружение готово", "Environment ready"),
    "pip.repair": (
        "pip в окружении нет -- восстанавливаю из колёс, встроенных в Python (сеть не нужна)",
        "pip is missing from the environment -- restoring it from the wheels bundled with "
        "Python, which needs no network",
    ),
    "venv.failed": (
        "Не смог создать виртуальное окружение. Проверь, что Python запускается: "
        "python -m venv installer_files\\env",
        "Could not create the virtual environment. Check that Python runs at all: "
        "python -m venv installer_files\\env",
    ),
    "python.not311": (
        "Этот Python {}.{}, а проект рассчитан на 3.11 -- установка пройдёт, но предупреждение "
        "при запуске будет. Лучше дай скачать portable 3.11.",
        "This is Python {}.{}, while the project is built for 3.11 -- the install will finish, "
        "but the WebUI will warn on start. Letting it download a portable 3.11 is better.",
    ),
    "pip.upgrade": ("Обновляю pip", "Upgrading pip"),
    "torch.present": ("torch уже установлен:", "torch is already installed:"),
    "torch.installing": ("Ставлю torch и torchvision", "Installing torch and torchvision"),
    "torch.unreachable": (
        "Внимание: {} не отвечает ({}). Если сейчас начнётся долгая пауза -- это pip ждёт сеть.",
        "Warning: {} does not answer ({}). If a long pause follows, that is pip waiting for the network.",
    ),
    "torch.failed": (
        "Установка torch не удалась. Что делать:\n"
        "  1) если у тебя уже есть torch -- запусти install.bat и выбери вариант с существующим torch;\n"
        "  2) если сеть режет download.pytorch.org -- положи колёса в installer_files\\wheels "
        "и запусти install.bat снова;\n"
        "  3) либо укажи зеркало в переменной TORCH_INDEX_URL и запусти снова.",
        "The torch install failed. What to do:\n"
        "  1) if torch is already installed, run install.bat and pick the existing-torch option;\n"
        "  2) if download.pytorch.org is blocked, drop the wheels into installer_files\\wheels "
        "and run install.bat again;\n"
        "  3) or point TORCH_INDEX_URL at a mirror and run it again.",
    ),
    "wheels.offline": ("Ставлю из локальных колёс (без сети)", "Installing from local wheels (offline)"),
    "wheels.missing": (
        "Колёса не нашлись, поэтому качаю как обычно. Положи torch-*.whl и torchvision-*.whl "
        "в installer_files\\wheels, чтобы установить вообще без сети.",
        "No wheels found, so downloading as usual. Drop torch-*.whl and torchvision-*.whl into "
        "installer_files\\wheels to install without any network at all.",
    ),
    "torch.not_reused": (
        "В новом окружении torch не виден -- качаю, как при обычной установке.",
        "The new environment cannot see torch after all -- downloading it like in a normal install.",
    ),
    "wheels.found": ("Нашёл колёса в installer_files\\wheels", "Found wheels in installer_files\\wheels"),
    "project.installing": (
        "Доустанавливаю остальное силами самого проекта (launch.py --exit)",
        "Installing the rest through the project itself (launch.py --exit)",
    ),
    "project.failed": (
        "launch.py завершился с ошибкой -- текст выше подскажет причину. Частые причины:\n"
        "  * torch не видит CUDA -- обнови драйвер NVIDIA;\n"
        "  * недоступен github.com -- оттуда берутся CLIP и nunchaku. Их можно положить\n"
        "    в installer_files\\wheels или подменить через переменные CLIP_PACKAGE и\n"
        "    NUNCHAKU_PACKAGE;\n"
        "  * оборвался pip -- просто запусти install.bat ещё раз, он продолжит с того же места.",
        "launch.py failed -- the output above should say why. Usual causes:\n"
        "  * torch cannot see CUDA -- update the NVIDIA driver;\n"
        "  * github.com is unreachable -- CLIP and nunchaku come from there. Drop them into\n"
        "    installer_files\\wheels, or override CLIP_PACKAGE and NUNCHAKU_PACKAGE;\n"
        "  * pip dropped out -- running install.bat again picks up where it stopped.",
    ),
    "settings.written": ("Написал webui.settings.bat", "Wrote webui.settings.bat"),
    "settings.backup": (
        "Старый webui.settings.bat сохранил рядом как {} -- вдруг там были твои флаги",
        "The old webui.settings.bat is saved next to it as {} -- in case it had your flags in it",
    ),
    "done": ("Готово", "Done"),
    "done.hint": (
        "Дальше запускай run.bat -- либо нажми Enter, и я запущу сразу.",
        "From now on start run.bat -- or press Enter and I will start it right away.",
    ),
    "space.low": (
        "Мало места: нужно примерно 8-10 ГБ, свободно {:.1f} ГБ.",
        "Low disk space: about 8-10 GB is needed, {:.1f} GB is free.",
    ),
    "choose": ("Как устанавливаем?", "How should we install?"),
    "choose.full": (
        "Всё своё: скачаю Python 3.11 и torch в папку форка",
        "Everything of our own: download Python 3.11 and torch into the fork's folder",
    ),
    "choose.reuse": (
        "Использовать уже установленный torch {} -- без скачивания",
        "Reuse the torch {} that is already installed -- no download",
    ),
    "choose.offline": (
        "Офлайн: поставить из installer_files\\wheels",
        "Offline: install from installer_files\\wheels",
    ),
    "prompt": ("Номер", "Number"),
    "choose.default": ("Enter -- вариант {}, или набери другой номер",
                       "Enter for option {}, or type another number"),
    "yes": ("да", "y"),
    "no": ("нет", "n"),
    "aborted": ("Прервано", "Aborted"),
    "doctor.title": ("Проверка сети", "Network check"),
    "doctor.hosts": ("Что доступно:", "What can be reached:"),
    "doctor.python": ("Твой Python:", "Your Python:"),
    "pip.present": ("pip на месте", "pip is there"),
    "pip.absent": ("pip отсутствует", "pip is missing"),
    "pip.fixable": (
        "но его можно восстановить без сети: python -m ensurepip --upgrade",
        "but it can be restored without any network: python -m ensurepip --upgrade",
    ),
    "pip.unfixable": (
        "и восстановить его нечем -- переустанови Python с галочкой «Install pip»",
        "and there is nothing to restore it from -- reinstall Python with «Install pip» ticked",
    ),
    "torch.present": (
        "torch установлен: {} (CUDA {})",
        "torch is installed: {} (CUDA {})",
    ),
    "torch.cuda.yes": ("видит видеокарту -- качать torch не нужно", "sees the GPU -- torch does not have to be downloaded"),
    "torch.cuda.no": ("не видит CUDA -- проверь драйвер NVIDIA", "cannot see CUDA -- check the NVIDIA driver"),
    "torch.absent": ("torch не установлен", "torch is not installed"),
    "doctor.mirrors": ("Откуда можно взять torch с CUDA:", "Where a CUDA torch can come from:"),
    "torch.mirror": (
        "Официальный индекс недоступен, беру зеркало: {}",
        "The official index is unreachable, using this mirror instead: {}",
    ),
    "advice.pip.broken": (
        "  * PyPI доступен, а вот pip в этом Python не работает. Установщик восстановит pip "
        "внутри своего окружения сам, сеть для этого не нужна.",
        "  * PyPI is reachable, it is pip in this Python that is broken. The installer rebuilds "
        "pip inside the environment it creates, and that needs no network.",
    ),
    "advice.mirror": (
        "  * torch с CUDA возьмём с зеркала: установщик выберет его сам, ничего делать не нужно.",
        "  * a CUDA torch will come from a mirror: the installer picks it automatically, "
        "there is nothing to do.",
    ),
    "advice.no_torch_source": (
        "  * Ни один источник torch не отвечает. Остаётся одно из трёх: включить VPN "
        "(тогда install.bat сделает всё сам), скачать колесо на другом компьютере и положить "
        "в installer_files\\wheels, либо попросить колесо у кого-то, у кого сеть работает.",
        "  * Not a single torch source answers. One of three is left: turn on a VPN (then "
        "install.bat does everything itself), download the wheel on another machine and drop it "
        "into installer_files\\wheels, or get one from somebody whose network works.",
    ),
    "doctor.pip": ("Пробую скачать крошечный пакет через pip:", "Trying to download a tiny package with pip:"),
    "doctor.pip.ok": ("pip работает -- установка пакетов пойдёт.", "pip works -- package installation will go through."),
    "doctor.pip.fail": ("pip не смог скачать даже 10 КБ. Значит, установка упрётся именно здесь.",
                        "pip could not download even 10 KB. That is what the install will hit."),
    "doctor.proxies": ("Прокси в переменных окружения:", "Proxies in the environment:"),
    "doctor.none": ("нет", "none"),
    "doctor.advice": ("Что это значит:", "What this means:"),
    "advice.pypi.bad": (
        "  * PyPI недоступен. Задай зеркало (INDEX_URL=<зеркало>) или прокси (HTTPS_PROXY=...), "
        "либо собери колёса на другой машине и положи их в installer_files\\wheels.",
        "  * PyPI is unreachable. Point INDEX_URL at a mirror or set HTTPS_PROXY, or collect the "
        "wheels on another machine and drop them into installer_files\\wheels.",
    ),
    "advice.torch.bad": (
        "  * Индекс torch недоступен. Скачивай torch сам в installer_files\\wheels или выбери "
        "вариант 2 -- использовать torch, который уже стоит.",
        "  * The torch index is unreachable. Fetch torch yourself into installer_files\\wheels, "
        "or pick option 2 -- reuse the torch that is already installed.",
    ),
    "advice.assets.bad": (
        "  * Файлы GitHub Releases недоступны. Это не страшно: portable Python нам не нужен, "
        "раз у тебя уже есть Python, а nunchaku необязателен и будет пропущен.",
        "  * GitHub release assets are unreachable. That is survivable: the portable Python is not "
        "needed when a Python is already installed, and nunchaku is optional and will be skipped.",
    ),
    "advice.pip.absent": (
        "  * В этом Python нет pip. Установщик восстановит его сам внутри окружения (это не "
        "требует сети), но если не выйдет -- переустанови Python с галочкой «Install pip».",
        "  * This Python has no pip. The installer restores it inside the environment it builds "
        "(no network needed), but if that fails, reinstall Python with «Install pip» ticked.",
    ),
    "advice.torch.absent": (
        "  * torch нет, а индекс PyTorch недоступен. На PyPI для Windows только CPU-версия, "
        "поэтому варианты такие: взять wheel с Hugging Face (если он доступен), попросить "
        "кого-то скачать его и положить в installer_files\\wheels, либо поставить через VPN.",
        "  * torch is missing and the PyTorch index is unreachable. PyPI only carries the CPU "
        "build for Windows, so: take a wheel from Hugging Face if it is reachable, have someone "
        "download one into installer_files\\wheels, or install it through a VPN.",
    ),
    "advice.ok": (
        "  * Всё нужное доступно. Запускай install.bat и, если torch у тебя уже стоит, "
        "соглашайся на вариант 2 -- тогда качать придётся только пакеты с PyPI.",
        "  * Everything needed is reachable. Run install.bat and, if torch is already installed, "
        "take option 2 -- then only the packages from PyPI have to be downloaded.",
    ),
    "enter.continue": (
        "Enter -- продолжить, n -- отмена.",
        "Enter to continue, n to cancel.",
    ),
}


def t(key, language="en"):
    entry = TEXT[key]
    return entry[0] if language == "ru" else entry[1]


def pick_language(requested="auto"):
    if requested in ("ru", "en"):
        return requested
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "")
    return "ru" if encoding in ("utf8", "utf8sig", "cp65001") else "en"


def venv_python(env_dir=None):
    # Resolved at call time so tests can point it elsewhere, and it never
    # re-wraps a Path: Path() itself dispatches on os.name, which would build a
    # WindowsPath on a Linux test runner.
    if env_dir is None:
        target = ENV_DIR
    elif isinstance(env_dir, Path):
        target = env_dir
    else:
        target = Path(env_dir)
    return target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def cache_env(root=None):
    """Environment variables that keep every cache on the installation drive."""
    root = ROOT if root is None else Path(root)
    cache = root / "installer_files" / "cache"
    tmp = root / "installer_files" / "tmp"
    return {
        "PIP_CACHE_DIR": str(cache / "pip"),
        "HF_HOME": str(cache / "huggingface"),
        "XDG_CACHE_HOME": str(cache),
        "TORCH_HOME": str(cache / "torch"),
        "TEMP": str(tmp),
        "TMP": str(tmp),
        "GRADIO_TEMP_DIR": str(root / "tmp"),
    }


def settings_text(root=None):
    """The webui.settings.bat the installer writes.

    webui.bat sources this file before doing anything else, so this is where
    the choices live: which environment to use and where the caches go.
    """
    root = ROOT if root is None else Path(root)
    lines = [
        "@echo off",
        "rem Generated by install.bat. This file is ignored by git, edit it freely.",
        "rem Point the WebUI at the environment the installer built:",
        'set "VENV_DIR=%~dp0installer_files\\env"',
        "rem Keep every download and cache on this drive instead of the system one:",
    ]
    for name, value in cache_env(root).items():
        try:
            relative = "\\".join(Path(value).relative_to(root).parts)
        except ValueError:
            continue  # a cache outside the folder cannot be expressed relative to the bat
        lines.append(f'set "{name}=%~dp0{relative}"')
    lines += [
        "rem Open the UI in the browser and keep the rest of the defaults:",
        "set COMMANDLINE_ARGS=--autolaunch",
        "",
    ]
    return "\r\n".join(lines)


def pinned_versions(root=None):
    """Read the torch pins out of modules/launch_utils.py instead of copying them."""
    root = ROOT if root is None else Path(root)
    try:
        source = (root / "modules" / "launch_utils.py").read_text(encoding="utf8", errors="ignore")
    except OSError:
        source = ""  # no checkout to read the pins from, fall back to the constants
    index = re.search(r'TORCH_INDEX_URL",\s*"([^"]+)"', source)
    torch = re.search(r"torch==([\d.]+)\+cu(\d+)", source)
    torchvision = re.search(r"torchvision==([\d.]+)\+cu(\d+)", source)
    return {
        "index": index.group(1) if index else FALLBACK_TORCH_INDEX,
        "torch": torch.group(1) if torch else FALLBACK_TORCH,
        "cuda": torch.group(2) if torch else FALLBACK_CUDA,
        "torchvision": torchvision.group(1) if torchvision else FALLBACK_TORCHVISION,
    }


def torch_install_command(root=None, index=None):
    pins = pinned_versions(root)
    return (
        f'install torch=={pins["torch"]}+cu{pins["cuda"]} '
        f'torchvision=={pins["torchvision"]}+cu{pins["cuda"]} '
        f'--extra-index-url {index or pins["index"]}'
    )


def quote(executable):
    """Quote real paths, leave launcher commands such as 'py -3.11' alone."""
    if Path(str(executable)).exists():
        return f'"{executable}"'
    return str(executable)


def first_reachable_index(preferred=None, timeout=8):
    """The first torch index that answers, official first, mirrors after."""
    candidates = [preferred] if preferred else []
    candidates += [url for url in TORCH_MIRRORS if url not in candidates]
    for url in candidates:
        if reachable(url, timeout=timeout):
            return url
    return None


def reachable(url, timeout=8):
    """Cheap liveness probe, so a dead host is reported before pip waits for it."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "arbuz-installer"})
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except Exception:
        return False


def free_bytes(path=ROOT):
    return shutil.disk_usage(str(path)).free


def gpu_name():
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    name = output.stdout.strip().splitlines()
    return name[0].strip() if name else None


def run(command, env=None, check=False):
    merged = os.environ.copy()
    merged.update(env or {})
    merged.update(cache_env())
    return subprocess.run(command, shell=True, env=merged, check=check)


def ask(question, default="1"):
    try:
        answer = input(question).strip()
    except EOFError:
        return default
    return answer or default


def candidate_pythons():
    """Interpreters worth asking, the one running this installer first.

    install.bat may have picked a Python that is not on PATH at all -- py -3.11
    resolves to a full path -- so sys.executable is both the most reliable
    candidate and the fastest to check. Probing the PATH first is what made the
    installer reach for a download on machines that already had Python.
    """
    ordered = [sys.executable, "python", "python3", "py -3.11", "py -3.10", "py -3"]
    seen, result = set(), []
    for candidate in ordered:
        if candidate and candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def probe_python(candidate, code):
    return subprocess.run(
        f'{quote(candidate)} -c "{code}"', shell=True, capture_output=True, text=True,
    )


def find_torch_python():
    """A Python that already has a working torch, if there is one on this machine."""
    for candidate in candidate_pythons():
        probe = probe_python(candidate, "import torch; print(torch.__version__)")
        if probe.returncode == 0 and probe.stdout.strip():
            return candidate, probe.stdout.strip()
    return None, None


def system_python(minimum=(3, 9), prefer=(3, 11)):
    """Best usable interpreter on this machine, preferring the version the project wants."""
    found = []
    for candidate in candidate_pythons():
        probe = probe_python(candidate, "import sys; print(sys.version_info[:2])")
        if probe.returncode != 0:
            continue
        try:
            version = tuple(int(part) for part in probe.stdout.strip("() ").split(","))
        except ValueError:
            continue
        if version >= minimum:
            found.append((candidate, version))
    if not found:
        return None
    for candidate, version in found:
        if version == prefer:
            return candidate
    return found[0][0]


def menu(options, language="en", default="1"):
    print()
    print(t("choose", language))
    for number, description in options:
        print(f"  {number}) {description}")
    print(f"  {t('choose.default', language).format(default)}")
    print(f"  [{t('prompt', language)}] ", end="", flush=True)


def bundled_python():
    """The portable Python installed by a previous run, if there is one."""
    for candidate in (PYTHON_DIR / "python.exe", PYTHON_DIR / "python", PYTHON_DIR / "bin" / "python3"):
        if candidate.exists():
            return candidate
    return None


def warn_if_not_311(executable, language="en"):
    probe = subprocess.run(
        f'{quote(executable)} -c "import sys; print(sys.version_info[:2])"',
        shell=True, capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return
    try:
        major, minor = (int(part) for part in probe.stdout.strip("() ").split(","))
    except ValueError:
        return
    if (major, minor) != (3, 11):
        print(f"  ! {t('python.not311', language).format(major, minor)}")


def fetch_python(language="en"):
    """Download a portable Python into installer_files, or fall back to the system one."""
    url = os.environ.get("ARBUZ_PYTHON_URL", PORTABLE_PYTHON_URL)
    print(f"  {t('python.downloading', language)}")
    INSTALLER_FILES.mkdir(parents=True, exist_ok=True)
    archive = INSTALLER_FILES / "python.tar.gz"
    if download(url, archive):
        PYTHON_DIR.mkdir(parents=True, exist_ok=True)
        extract = run(f'tar -xzf "{archive}" -C "{PYTHON_DIR}" --strip-components=1')
        archive.unlink(missing_ok=True)
        for candidate in (PYTHON_DIR / "python.exe", PYTHON_DIR / "python", PYTHON_DIR / "bin" / "python3"):
            if candidate.exists():
                return str(candidate)
    else:
        fallback = system_python()
        if fallback:
            warn_if_not_311(fallback, language)
            return fallback
    print(t("python.failed", language))
    return None


def doctor(language="en"):
    """Say what the network allows, so the install can be planned around it."""
    import socket

    print("=" * 62)
    print(f"{t('title', language)}  [rev {INSTALLER_REV}] -- {t('doctor.title', language)}")
    print("=" * 62)
    print(f"{t('folder', language)}: {ROOT}")
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    print(f"{t('free', language)}: {free_bytes() / 1024 ** 3:.1f} GB")

    torch_index = os.environ.get("TORCH_INDEX_URL") or pinned_versions()["index"]
    hosts = [
        ("PyPI", "https://pypi.org/simple/"),
        ("PyPI files", "https://files.pythonhosted.org/"),
        ("GitHub archive (CLIP)", "https://codeload.github.com/"),
        ("GitHub release assets (portable Python, nunchaku)", "https://objects.githubusercontent.com/"),
        ("Hugging Face (wheels published by the community)", "https://huggingface.co/"),
        ("Hugging Face file CDN", "https://cdn-lfs.huggingface.co/"),
        (f"torch index ({torch_index})", torch_index),
    ]

    print()
    print(f"{t('doctor.python', language)} {sys.executable} ({sys.version.split()[0]})")

    pip_here = subprocess.run(f'{quote(sys.executable)} -m pip --version',
                              shell=True, capture_output=True, text=True)
    if pip_here.returncode == 0:
        print(f"  pip: {t('pip.present', language)} ({pip_here.stdout.strip().split()[1]})")
    else:
        ensurepip = subprocess.run(f'{quote(sys.executable)} -c "import ensurepip; print(1)"',
                                   shell=True, capture_output=True, text=True)
        fixable = ensurepip.returncode == 0
        print(f"  pip: {t('pip.absent', language)} -- "
              f"{t('pip.fixable' if fixable else 'pip.unfixable', language)}")

    torch_here = subprocess.run(
        f'{quote(sys.executable)} -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"',
        shell=True, capture_output=True, text=True,
    )
    if torch_here.returncode == 0:
        parts = torch_here.stdout.split()
        version, cuda = parts[0], (parts[1] if len(parts) > 1 else "?")
        available = len(parts) > 2 and parts[2] == "True"
        print(f"  {t('torch.present', language).format(version, cuda)} -- "
              f"{t('torch.cuda.yes' if available else 'torch.cuda.no', language)}")
    else:
        print(f"  {t('torch.absent', language)}")

    print()
    print(t("doctor.hosts", language))
    results = {}
    for name, url in hosts:
        try:
            started = time.time()
            results[name] = reachable(url, timeout=8)
            elapsed = (time.time() - started) * 1000
            status = "OK  " if results[name] else "FAIL"
        except socket.gaierror as error:
            results[name], status, elapsed = False, "FAIL", 0.0
            print(f"       (DNS: {error})")
        print(f"  {status} {elapsed:6.0f} ms  {name}")
        print(f"         {url}")

    print()
    print(t("doctor.pip", language))
    try:
        probe = subprocess.run(
            f'"{sys.executable}" -m pip download --no-deps --no-cache-dir '
            f'--dest "{CACHE_DIR / "probe"}" six',
            shell=True, capture_output=True, text=True, timeout=180,
            env={**os.environ, **cache_env()},
        )
        pip_ok = probe.returncode == 0
        if not pip_ok:
            print((probe.stdout or "") + (probe.stderr or ""))
    except subprocess.TimeoutExpired:
        pip_ok = False
        print("timeout")
    print(f"  {t('doctor.pip.ok' if pip_ok else 'doctor.pip.fail', language)}")

    proxies = {name: os.environ[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "INDEX_URL")
               if os.environ.get(name)}
    print()
    print(f"{t('doctor.proxies', language)} {proxies or t('doctor.none', language)}")

    print()
    print(t("doctor.mirrors", language))
    working_mirror = None
    for url in TORCH_MIRRORS:
        ok = reachable(url, timeout=8)
        working_mirror = working_mirror or (url if ok else None)
        print(f"  {'OK  ' if ok else 'FAIL'}  {url}")

    print()
    print(t("doctor.advice", language))
    if pip_here.returncode != 0:
        print(t("advice.pip.absent", language))
    if not results.get("PyPI"):
        print(t("advice.pypi.bad", language))
    elif not pip_ok:
        print(t("advice.pip.broken", language))
    if working_mirror and working_mirror != TORCH_MIRRORS[0]:
        print(t("advice.mirror", language))
    if torch_here.returncode != 0 and not working_mirror:
        print(t("advice.no_torch_source", language))
    if torch_here.returncode != 0 and not results.get(f"torch index ({torch_index})") and working_mirror:
        print(t("advice.torch.absent", language))
    if not results.get("GitHub release assets (portable Python, nunchaku)"):
        print(t("advice.assets.bad", language))
    if pip_ok and results.get("PyPI") and results.get(f"torch index ({torch_index})"):
        print(t("advice.ok", language))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Install ArbuzDiffusion in one click.")
    parser.add_argument("--yes", action="store_true", help="take the default option everywhere")
    parser.add_argument("--lang", default="auto", help="ru, en or auto (default)")
    parser.add_argument("--reuse-torch", action="store_true", help="build the venv on top of an existing torch")
    parser.add_argument("--offline", action="store_true", help="install from installer_files/wheels only")
    parser.add_argument("--index-url", default=os.environ.get("TORCH_INDEX_URL"), help="torch wheel index")
    parser.add_argument("--no-launch", action="store_true", help="do not offer to start the UI")
    parser.add_argument("--doctor", action="store_true", help="check what the network allows and exit")
    args = parser.parse_args(argv)

    if args.doctor:
        return doctor(pick_language(args.lang))

    language = pick_language(args.lang)
    if language == "ru":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            language = "en"

    print("=" * 62)
    print(f"{t('title', language)}  [rev {INSTALLER_REV}]")
    print("=" * 62)
    print(f"{t('folder', language)}: {ROOT}")
    print(f"{t('free', language)}: {free_bytes() / 1024 ** 3:.1f} GB")
    print(f"{t('gpu', language)}: {gpu_name() or t('gpu.none', language)}")
    if not shutil.which("git"):
        print(f"! {t('git.missing', language)}")

    if free_bytes() < MINIMUM_FREE_BYTES:
        print(f"! {t('space.low', language).format(free_bytes() / 1024 ** 3)}")
        if not args.yes:
            print(f"  {t('enter.continue', language)}")
            if ask(f"  [{t('yes', language)}] ", "y").lower() == "n":
                print(t("aborted", language))
                return 1
    elif free_bytes() < RECOMMENDED_FREE_BYTES:
        print(f"! {t('space.low', language).format(free_bytes() / 1024 ** 3)}")

    steps = 6
    def step(number, message):
        print(f"\n[{number}/{steps}] {message}")

    # 1 -- a Python to build the environment from.
    step(1, "Python")
    wheels = sorted(WHEELS_DIR.glob("*.whl")) if WHEELS_DIR.is_dir() else []
    torch_python, torch_version = find_torch_python()
    options = [(1, t("choose.full", language))]
    if torch_python:
        options.append((2, t("choose.reuse", language).format(torch_version)))
    if wheels:
        options.append((3, t("choose.offline", language)))

    # Nothing of ours has been installed yet and torch is already on the
    # machine: reusing it saves three gigabytes, so that is the default.
    choice = "2" if torch_python and not bundled_python() else "1"
    if args.reuse_torch and torch_python:
        choice = "2"
    elif args.offline and wheels:
        choice = "3"
    elif not args.yes and len(options) > 1:
        menu(options, language, default=choice)
        choice = ask("", choice)

    base_python = None
    reuse_torch = choice == "2"
    if reuse_torch:
        base_python = torch_python
    else:
        bundled = bundled_python()
        if bundled:
            base_python = str(bundled)
            print(f"  {t('python.reuse', language)} {base_python}")
        else:
            found = system_python()
            if found:
                base_python = found
                print(f"  {t('python.using', language)} {base_python}")
                warn_if_not_311(found, language)
            else:
                base_python = fetch_python(language)
                if not base_python:
                    return 1

    # 2 -- the virtual environment.
    step(2, t("venv.creating", language))
    if reuse_torch:
        print(f"  {t('venv.reuse_torch', language)}")
    # every directory the redirected environment points at, derived from the
    # same mapping that generates webui.settings.bat so they cannot drift apart
    for directory in cache_env().values():
        Path(directory).mkdir(parents=True, exist_ok=True)

    if not venv_python().exists():
        flags = "--system-site-packages" if reuse_torch else ""
        created = run(" ".join(part for part in
                               (quote(base_python), "-m venv", flags, f'"{ENV_DIR}"') if part))
        if created.returncode != 0:
            print(t("venv.failed", language))
            return 1
    target = venv_python()
    if not target.exists():
        print(t("venv.failed", language))
        return 1
    print(f"  {t('venv.ready', language)}: {target}")

    step(3, t("pip.upgrade", language))
    if run(f'"{target}" -m pip --version').returncode != 0:
        # Some Python installations simply have no pip -- the machine this was
        # written for had none -- and ensurepip rebuilds it from the wheels
        # shipped with the interpreter, so it works with no network at all.
        print(f"  {t('pip.repair', language)}")
        run(f'"{target}" -m ensurepip --upgrade')
    run(f'"{target}" -m pip install --upgrade pip wheel', env=cache_env())

    # 4 -- torch.
    step(4, "torch")
    probe = subprocess.run(
        f'"{target}" -c "import torch; print(torch.__version__)"',
        shell=True, capture_output=True, text=True, env={**os.environ, **cache_env()},
    )
    if probe.returncode == 0:
        print(f"  {t('torch.present', language)} {probe.stdout.strip()}")
    elif wheels:
        print(f"  {t('wheels.offline', language)}")
        run(f'"{target}" -m pip install --no-index --find-links "{WHEELS_DIR}" '
            + " ".join(f'"{wheel}"' for wheel in wheels))
    else:
        if choice == "3" or args.offline:
            print(f"  ! {t('wheels.missing', language)}")
        elif reuse_torch:
            print(f"  ! {t('torch.not_reused', language)}")
        preferred = args.index_url or os.environ.get("TORCH_INDEX_URL")
        index = first_reachable_index(preferred)
        if not index:
            index = preferred or pinned_versions()["index"]
            print(f"  ! {t('torch.unreachable', language).format(index, 'timeout')}")
        elif index != TORCH_MIRRORS[0]:
            print(f"  {t('torch.mirror', language).format(index)}")
        print(f"  {t('torch.installing', language)}")
        installed = run(f'"{target}" -m pip {torch_install_command(index=index)}', env=cache_env())
        if installed.returncode != 0:
            print(t("torch.failed", language))
            return 1

    # 5 -- everything the project installs itself.
    step(5, t("project.installing", language))
    project = run(f'"{target}" launch.py --exit', env=cache_env())
    if project.returncode != 0:
        print(t("project.failed", language))
        return 1

    # 6 -- settings for the next start.
    step(6, t("settings.written", language))
    backup = write_settings()
    if backup:
        print(f"  {t('settings.backup', language).format(backup.name)}")
    print(f"  {SETTINGS_FILE}")

    print()
    print(t("done", language))
    if not args.no_launch:
        print(f"  {t('done.hint', language)}")
        answer = ask("  > ", "") if not args.yes else ""
        if answer.lower() in ("", "y", "д", "да"):
            launch = "call webui.bat" if os.name == "nt" else f'"{target}" launch.py'
            return run(launch).returncode
    return 0


def write_settings():
    """Write webui.settings.bat, keeping whatever was there under .bak."""
    backup = None
    if SETTINGS_FILE.exists():
        # people keep their launch flags in this file; never throw it away
        backup = SETTINGS_FILE.with_suffix(SETTINGS_FILE.suffix + ".bak")
        backup.write_bytes(SETTINGS_FILE.read_bytes())
    SETTINGS_FILE.write_text(settings_text(), encoding="utf8", newline="")
    return backup


def download(url, destination):
    """Stream a file with a progress line; returns True on success."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response, open(destination, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1024 ** 2:.0f} / {total / 1024 ** 2:.0f} MB", end="", flush=True)
        print()
        return True
    except Exception as error:
        print(f"\n  {error}")
        return False


if __name__ == "__main__":
    sys.exit(main())
