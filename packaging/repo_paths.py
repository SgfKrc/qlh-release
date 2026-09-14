"""Resolve the repositories that make up a local QLH development checkout."""

from __future__ import annotations

import os
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = RELEASE_ROOT.parent


def _configured(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


def core_root() -> Path:
    return _configured("QLH_CORE_ROOT", WORKSPACE_ROOT / "qlh")


def shell_root() -> Path:
    return _configured("QLH_SHELL_ROOT", WORKSPACE_ROOT / "qlh-shell")


def android_root() -> Path:
    return _configured("QLH_ANDROID_ROOT", WORKSPACE_ROOT / "qlh-android")
