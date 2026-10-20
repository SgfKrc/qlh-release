# -*- mode: python ; coding: utf-8 -*-
"""QLH TUI 发行包：默认边缘 CPU，QLH_EDGE_FREEZE_TORCH=1 冻结 CUDA PyTorch。

## 与相邻两个 spec 的分工

| spec | 入口 | 主引擎 | PyTorch | 前端 | 体积量级 |
|---|---|---|---|---|---|
| `qlh-cpu.spec`  | `launcher.py`    | ✅ 冻结 | ✅ 打进 | 必需 | ~722MB |
| `qlh-cuda.spec` | `launcher.py`    | ✅ 冻结 | ✅ CUDA | 必需 | ~12.5GB |
| `qlh-slim.spec` | `app_maintenance.py` | ❌ 只带引导器（源码以 data 形式随包，运行时靠外部 venv 跑） | ❌ | ❌ | ~42MB |
| **本 spec**     | `launcher.py`    | ✅ 冻结 | 默认排除；显式CUDA档冻结 | ❌ | 按CPU／CUDA依赖闭包 |

★ 2026-10-03：按产品基线惯例，发行物应是「**免安装的 PyInstaller 目录 + Inno Setup 6
安装包**」，且打的是**主引擎**（`launcher.py` 直接 `from api_server import run_api_servers`），
而不是一个单纯启动器。本 spec 面向**边缘 CPU 设备**：主引擎冻结在内，但不装 torch ——
边缘档的推理走 llama.cpp/GGUF，不需要 PyTorch。

**可行性前提（已实测）**：`api_server` 的 torch 依赖是**惰性**的
（`model_host._LazyModelManager` 到首次用模型才导入 `model_module`）。在无 torch 的
`.venv-edge` 里 `import api_server` **0.80s 成功**，且 `torch` 未进 `sys.modules`
⇒ 排除 torch 不会让主引擎在导入期崩，只会让"需要 PyTorch 引擎"的路径在**用到时**明确
失败（那是正确行为：边缘档本就不提供该引擎）。

显式`QLH_EDGE_FREEZE_TORCH=1`时，复用同一TUI及keep-head原生闭包，冻结CUDA PyTorch与转换依赖，并按`torch_cuda`扫描；默认0按`llama_cpp_only`扫描。

**前端**：`frontend_cybergothic` 已停止维护，产品以 TUI 为准
（`qlh-slim.spec` 的文件头 2026-09-30 起即已声明），本 spec 不依赖它，也不需要
`QLH_SHELL_ROOT`。
"""

import glob
import os
import sys
from pathlib import Path

# 项目根目录（SPECPATH = 当前 spec 文件所在目录）
_RELEASE_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
_PROJECT_ROOT = os.environ.get(
    "QLH_CORE_ROOT", os.path.abspath(os.path.join(_RELEASE_ROOT, "..", "qlh"))
)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
_RELEASE_CONTRACT = os.path.join(_PROJECT_ROOT, "release-contract.json")

# Freeze the same TUI product with CUDA PyTorch when explicitly requested.
# The source-only/CPU package keeps its existing dependency closure.
_torch_choice = os.environ.get("QLH_EDGE_FREEZE_TORCH", "0")
if _torch_choice not in {"0", "1"}:
    raise ValueError("QLH_EDGE_FREEZE_TORCH must be 0 or 1")
_FREEZE_TORCH = _torch_choice == "1"
if _FREEZE_TORCH:
    import torch as _torch
    if not getattr(_torch.version, "cuda", None):
        raise RuntimeError("CUDA TUI packaging requires a CUDA-enabled PyTorch build")
_BUNDLE_NAME = "QLH-Edge-Inference-CUDA" if _FREEZE_TORCH else "QLH-Edge-Inference"
_TORCH_IMPORTS = ["model_module", "torch", "transformers", "transformers.models.auto",
    "transformers.models.qwen2.modeling_qwen2", "transformers.models.qwen3_5.modeling_qwen3_5",
    "accelerate", "safetensors", "tokenizers", "sentencepiece", "huggingface_hub", "einops"] if _FREEZE_TORCH else []


# MODEL-SUPPLY-01: explicit data + dependency closure for offline GGUF cutting.
sys.path.insert(0, SPECPATH)
from keep_head_resources import collect_keep_head_binaries
from model_supply_resources import collect_model_supply_resources

# Required patched ABI: fail before Analysis if the native closure is incomplete.
# Keep this TOC out of Analysis: its dependency scan relocates build-tree DLLs
# into the shared bundle root, where they collide with pip's same-name ggml DLLs.
_KEEP_HEAD_BINARIES = collect_keep_head_binaries(_PROJECT_ROOT)

_SUPPLY_DATAS, _SUPPLY_IMPORTS = collect_model_supply_resources(_PROJECT_ROOT, freeze_conversion=_FREEZE_TORCH)

# 受管 llama-quantize（模型工具链）—— 可选：缺失时只跳过该工具，不阻断主引擎打包。
sys.path.insert(0, _PROJECT_ROOT)
try:
    from scripts.model_tools.llama_quantize_toolchain import verify_managed_package

    _LLAMA_QUANTIZE_PACKAGE = os.path.join(
        _PROJECT_ROOT, "build", "model-tools", "llama-quantize", "packages",
        "windows-x86_64",
    )
    _lq_ok = bool(
        verify_managed_package(
            Path(_LLAMA_QUANTIZE_PACKAGE), expected_target="windows-x86_64",
        )["valid"]
    )
except Exception as _exc:  # noqa: BLE001 - 打包期诊断
    print(f"[spec] llama-quantize 校验不可用，跳过该工具：{_exc}")
    _lq_ok = False

if not os.path.isdir(_SRC_DIR):
    raise SystemExit(
        f"[spec] 主程序源码目录不存在：{_SRC_DIR}（可用 QLH_CORE_ROOT 覆盖主仓根目录）"
    )

# ============================================================
# llama.cpp 原生库（DLL）—— 边缘档的推理引擎就是它
# ============================================================
_llama_cpp_native = []
try:
    import llama_cpp as _lc

    _lib_dir = os.path.join(os.path.dirname(os.path.abspath(_lc.__file__)), "lib")
    if os.path.isdir(_lib_dir):
        for _pattern in ("*.dll", "*.lib"):
            for _native in glob.glob(os.path.join(_lib_dir, _pattern)):
                _llama_cpp_native.append((_native, "llama_cpp/lib"))
    print(f"[spec] Collected {len(_llama_cpp_native)} llama.cpp native files")
except Exception as _e:  # noqa: BLE001 - 打包期诊断
    print(f"[spec] WARNING: Failed to collect llama.cpp DLLs: {_e}")

_datas = [(_RELEASE_CONTRACT, "."), *_SUPPLY_DATAS]
if _lq_ok:
    _datas.append((_LLAMA_QUANTIZE_PACKAGE, "model-tools/llama-quantize/windows-x86_64"))
else:
    print("[spec] 跳过 llama-quantize（未通过受管包校验）")

# Cutting scripts, GGUF sources and worker data are required by the shared collector.

a = Analysis(
    ["launcher.py"],
    pathex=[_SRC_DIR, SPECPATH],
    binaries=_llama_cpp_native,
    datas=_datas,
    hiddenimports=["app_bootstrap", "app_maintenance",
        # ---- llama.cpp 引擎（边缘档唯一推理引擎）----
        "llama_cpp",
        "llama_cpp._internals",
        "llama_cpp.llama_cpp",
        "llama_engine",

        # ---- uvicorn 子模块（动态导入）----
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",

        # ---- FastAPI / Starlette ----
        "fastapi",
        "starlette",
        "starlette.middleware",
        "starlette.middleware.cors",
        "httpx",
        "pydantic",
        "python_multipart",

        # ---- SSL / OpenSSL（uvicorn 依赖，显式收集避免 DLL 冲突）----
        "ssl",
        "_ssl",

        # ---- 本地存储（DB 不可用时自动降级）----
        "local_store",
        "recovery_journal",
        "execution_provider",
        "task_worker_protocol",
        "task_worker_adapter",
        "sqlite3",
        "_sqlite3",

        # ---- 智能编排 ----

        # ---- 管理界面（TUI，产品界面以此为准）----
        # `launcher.py` 的 `_run_tui()` 在服务就绪后 `from tui_textual import main`；
        # Textual 按平台动态导入驱动，故显式列出运行时可能走到的几个。
        "tui_textual",
        "tui_api",
        "tui_commands",
        "tui_shared",
        "tui_sse",
        "textual",
        "textual.drivers.windows_driver",
        "textual.drivers.linux_driver",
        "textual.drivers.headless_driver",
        "textual.drivers.linux_inline_driver",
        "textual.drivers._input_reader_windows",
        "textual.drivers._input_reader_linux",

        # ---- 工具 ----
        "psutil",

        # ---- 标准库 ----
        "asyncio",
    ] + _SUPPLY_IMPORTS + _TORCH_IMPORTS,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Supply modules ship as whitelisted sources with a lightweight package
        # initializer; do not archive the developer probes/adapters initializer.
        "scripts.model_tools",
        "tkinter",
        "test",
        "pydoc",
        # ★ 边缘档的核心差别：不装 PyTorch 系。它们只能由**外部 runtime venv** 提供
        #   （`app_maintenance` + `runtime_guard` 那条路），或者干脆不提供 —— 边缘档的
        #   推理走 llama.cpp。`api_server` 对它们的依赖是惰性的（见文件头）。同时
        #   `pywebview` / 前端相关也排除（前端已停止维护）。
        "torchvision",
        "torchaudio",
        "webview",
        "pandas",
        "matplotlib",
        "IPython",
        "jedi",
    ] + ([] if _FREEZE_TORCH else ["torch", "transformers", "transformers_stream_generator",
        "accelerate", "bitsandbytes", "einops", "tiktoken"]),
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],                    # ★ onedir: 二进制 DLL 不嵌入 EXE（由 COLLECT 放入 _internal/）
    [],                    # ★ onedir: 数据文件不嵌入 EXE
    [],
    name=_BUNDLE_NAME,
    icon="leds.ico",
    # ★ 2026-10-03：**必须 console=True**。产品界面是 **TUI**（Textual），它需要真实的
    #   控制台 —— 此前沿用 GUI 应用的 `console=False`，从这种 exe 里起 TUI 会以
    #   `AttributeError: 'NoneType' object has no attribute 'fileno'`（textual 取
    #   stdout 失败）→ `AssertionError: Driver must be in application mode` 崩掉（实测）。
    #   `console=True` 让 exe 自带控制台窗口，TUI 就在其中运行。
    console=True,
    debug=False,
    strip=True,
    upx=False,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    _KEEP_HEAD_BINARIES,
    strip=False,
    upx=False,
    name=_BUNDLE_NAME,
)
