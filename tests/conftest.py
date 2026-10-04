"""Test defaults for the standalone release repository."""

import os
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _core_root_default() -> Path:
    """主仓根默认值，**同时支持子模块布局（现行）与外部兄弟目录布局（旧）**。

    `packaging` 现在是主仓子模块（挂载在 `<core>/packaging`），外部兄弟检出
    `qlh-release` / `qlh-shell` / `qlh-android` 已不再保留。判据用主仓特征目录
    `<core>/src`，而不是名字猜测。
    """
    embedded = RELEASE_ROOT.parent
    if (embedded / "src").is_dir():
        return embedded
    return RELEASE_ROOT.parent / "qlh"


def _sibling_or_embedded(name_embedded: str, name_sibling: str) -> Path:
    core = _core_root_default()
    embedded = core / name_embedded
    return embedded if embedded.is_dir() else RELEASE_ROOT.parent / name_sibling


CORE_ROOT = Path(os.environ.get("QLH_CORE_ROOT", _core_root_default())).resolve()
SHELL_ROOT = Path(os.environ.get(
    "QLH_SHELL_ROOT", _sibling_or_embedded("frontend_cybergothic", "qlh-shell"),
)).resolve()
ANDROID_ROOT = Path(os.environ.get(
    "QLH_ANDROID_ROOT", _sibling_or_embedded("android", "qlh-android"),
)).resolve()
os.environ.setdefault("QLH_CORE_ROOT", str(CORE_ROOT))
os.environ.setdefault("QLH_SHELL_ROOT", str(SHELL_ROOT))
os.environ.setdefault("QLH_ANDROID_ROOT", str(ANDROID_ROOT))

for path in (RELEASE_ROOT / "packaging", CORE_ROOT, CORE_ROOT / "scripts", CORE_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
