# -*- mode: python ; coding: utf-8 -*-
"""QLH 瘦身安装包（SLIM）：不再把 PyTorch 系打进包体。

★ 2026-09-30 产品基线整改：**产品以 TUI 为主**，包体**不再内置 ts 前端**
（`frontend_cybergothic`）。此前本 spec 硬要求 `frontend_cybergothic/dist` 存在、
否则直接 `SystemExit` —— 整改后该依赖连同 `QLH_SHELL_ROOT` 一并去掉，
`qlh-shell` 仓不再是本包的构建前置。

产物只含：
  - QLH-Edge-Inference.exe：轻量引导器（qlh_launcher.py，不 import 推理运行时）
  - _internal/src/            ：主程序源码（运行时由外部 venv python 以源码方式运行）
  - _internal/packaging/      ：外部运行时依赖清单（llama_cpp_only/cpu/cuda）
  - _internal/pubkeys         ：验签公钥
PyTorch / Transformers / llama.cpp / FastAPI / uvicorn 等全部从包体**排除**，
由 qlh_launcher 每次启动用 runtime_guard 检查外部 runtime venv，缺失则
pip 引导（CPU --index-url .../whl/cpu；CUDA 官方默认），再用该 venv python
以 src 源码方式跑 uvicorn。体积从 ~734MB(CPU)/~1.7GB-13GB(CUDA) 降到几乎
仅引导器 + 源码（几十 MB），安装包更新不重装大 PyTorch。
"""

import os
import sys

#: ★ 构建期**显式选** runtime profile（`QLH_BUILD_PROFILE`）。默认 `torch_cpu`
#: 保留旧行为。选定后包内**只带这一份**运行时依赖清单 —— "这个包需要什么运行时"
#: 由构建期决定并随 manifest 签名发布，而不是靠目标机器探测反推（探测会把
#: "这台机器碰巧装了 torch"误当成"这个包支持 torch"）。
_PROFILE = os.environ.get("QLH_BUILD_PROFILE", "").strip().lower() or "torch_cpu"

sys.path.insert(0, SPECPATH)
from runtime_guard import RUNTIME_PROFILES, profile_requirements_filename  # noqa: E402

if _PROFILE not in RUNTIME_PROFILES:
    raise SystemExit(
        f"[qlh-slim.spec] 未知的 QLH_BUILD_PROFILE={_PROFILE!r}；"
        f"可选：{sorted(RUNTIME_PROFILES)}"
    )

_RELEASE_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
_ROOT = os.environ.get(
    "QLH_CORE_ROOT", os.path.abspath(os.path.join(_RELEASE_ROOT, "..", "qlh"))
)
_SRC_DIR = os.path.join(_ROOT, "src")
_PUBKEYS = os.path.join(SPECPATH, "pubkeys")
_ICO = os.path.join(SPECPATH, "leds.ico")

_RUNTIME_REQS = [
    # 只带**选定 profile** 的清单（见文件头 `_PROFILE`）；包内不存在其它 profile
    # 的清单 ⇒ 运行时无法"顺手"按另一个 profile 装依赖。
    (os.path.join(SPECPATH, profile_requirements_filename(_PROFILE)), "packaging"),
]

if not os.path.isdir(_SRC_DIR):
    raise SystemExit(
        f"[qlh-slim.spec] 主程序源码目录不存在：{_SRC_DIR}"
        "（可用 QLH_CORE_ROOT 覆盖主仓根目录）"
    )

a = Analysis(
    [os.path.join(SPECPATH, "qlh_launcher.py")],
    pathex=[SPECPATH, _ROOT],
    datas=[
        (_ICO, "."),
        (_PUBKEYS, "pubkeys"),
        (_SRC_DIR, "src"),
    ] + _RUNTIME_REQS,
    hiddenimports=[
        "tkinter",
        "tkinter.ttk",
        "cryptography",
        "cryptography.hazmat.primitives.asymmetric.ed25519",
        "launcher_slots",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # 重量级推理运行时：全部移到外部 runtime venv，由 qlh_launcher 引导安装
        "torch", "torchvision", "torchaudio",
        "transformers", "accelerate", "bitsandbytes",
        "llama_cpp", "fastapi", "uvicorn", "pywebview",
    ],
)

pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    [],
    [],
    [],
    name="QLH-Edge-Inference",
    icon=_ICO,
    console=False,
    debug=False,
    strip=True,
    upx=False,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="QLH-Edge-Inference",
)
