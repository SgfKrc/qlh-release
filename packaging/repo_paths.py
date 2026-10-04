"""Resolve the repositories that make up a local QLH development checkout."""

from __future__ import annotations

import os
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = RELEASE_ROOT.parent


def _configured(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


def _core_root_default() -> Path:
    """主仓根的默认值，**同时支持两种布局**。

    - **子模块布局（现行）**：`packaging` 挂载在 `<core>/packaging`，此时
      `WORKSPACE_ROOT` 已经是主仓根本身，而外部兄弟检出 `qlh-release` /
      `qlh-shell` / `qlh-android` 已不再保留。
    - **外部兄弟布局（旧）**：`<workspace>/qlh-release` 与 `<workspace>/qlh`
      并列，`qlh-shell` / `qlh-android` 也在 `<workspace>` 下。

    判据用「主仓特征文件」而不是名字猜测：`<core>/src` 存在才认子模块布局。
    """
    embedded = RELEASE_ROOT.parent
    if (embedded / "src").is_dir():
        return embedded
    return WORKSPACE_ROOT / "qlh"


def core_root() -> Path:
    return _configured("QLH_CORE_ROOT", _core_root_default())


def shell_root() -> Path:
    # 子模块布局下产品壳是 `<core>/frontend_cybergothic`；旧布局才是兄弟目录。
    core = _core_root_default()
    embedded = core / "frontend_cybergothic"
    default = embedded if embedded.is_dir() else WORKSPACE_ROOT / "qlh-shell"
    return _configured("QLH_SHELL_ROOT", default)


def android_root() -> Path:
    core = _core_root_default()
    embedded = core / "android"
    default = embedded if embedded.is_dir() else WORKSPACE_ROOT / "qlh-android"
    return _configured("QLH_ANDROID_ROOT", default)
