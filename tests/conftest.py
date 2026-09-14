"""Test defaults for the standalone release repository."""

import os
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = Path(os.environ.get("QLH_CORE_ROOT", RELEASE_ROOT.parent / "qlh")).resolve()
SHELL_ROOT = Path(os.environ.get("QLH_SHELL_ROOT", RELEASE_ROOT.parent / "qlh-shell")).resolve()
ANDROID_ROOT = Path(os.environ.get("QLH_ANDROID_ROOT", RELEASE_ROOT.parent / "qlh-android")).resolve()
os.environ.setdefault("QLH_CORE_ROOT", str(CORE_ROOT))
os.environ.setdefault("QLH_SHELL_ROOT", str(SHELL_ROOT))
os.environ.setdefault("QLH_ANDROID_ROOT", str(ANDROID_ROOT))

for path in (RELEASE_ROOT / "packaging", CORE_ROOT, CORE_ROOT / "scripts", CORE_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
