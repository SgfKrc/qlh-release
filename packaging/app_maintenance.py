#!/usr/bin/env python3
"""QLH application maintenance and external runtime commands.

It intentionally knows how to launch the application but never imports the
inference runtime.  This keeps startup, update checks, and repair available
when torch/model/database dependencies are broken.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

_PACKAGING_DIR = Path(__file__).resolve().parent
if str(_PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGING_DIR))

from update_core import (
    DownloadProgress,
    DownloadProgressCallback,
    UpdateError,
    default_state_dir,
    load_json_state,
)
from diagnose import diagnose_install, format_diagnosis, write_diagnosis_report
from data_retention import (
    DataRetentionError,
    auto_reassociate_user_data,
    reassociate_user_data,
    retain_user_data,
    retention_status,
)
from install_manifest import main as install_manifest_main
from install_manifest import verify_install_tree
from repair import RepairError, format_repair, repair_install
from updater import (
    check_updates, configured_sources, detect_current_version, detect_profile,
    main as updater_main,
)
from version_store import VersionStore
from packaging_release_contract import (
    ReleaseContractError,
    build_release_child_environment,
)



def install_root() -> Path:
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
        if root.name.lower() == "bin":
            return root.parent
        return root
    return _PACKAGING_DIR.parent


def core_root() -> Path:
    """Return the sibling core checkout used by the development launcher."""
    configured = os.environ.get("QLH_CORE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return _PACKAGING_DIR.parent.parent


def _candidate_app_roots(preferred_variant: str | None = None) -> list[Path]:
    state = load_json_state(default_state_dir() / "launcher.json")
    configured = [os.environ.get("QLH_APP_HOME", ""), state.get("app_path", "")]
    roots = [Path(value).expanduser() for value in configured if isinstance(value, str) and value]
    try:
        active = VersionStore().active_path()
    except Exception:
        active = None
    if active is not None:
        roots.append(active)
    roots.append(install_root())
    roots.append(core_root())
    if os.name == "nt":
        names = (
            ("QLH-Edge-Inference-CUDA", "QLH-Edge-Inference")
            if preferred_variant == "cuda"
            else ("QLH-Edge-Inference", "QLH-Edge-Inference-CUDA")
        )
        for variable in ("ProgramFiles", "ProgramFiles(x86)"):
            base = os.environ.get(variable)
            if base:
                roots.extend(Path(base) / name for name in names)
        local = os.environ.get("LOCALAPPDATA")
        if local:
            roots.extend(Path(local) / "Programs" / name for name in names)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def installed_app_root(preferred_variant: str | None = None) -> Path | None:
    for root in _candidate_app_roots(preferred_variant):
        if os.name == "nt" and (root / "QLH-Edge-Inference.exe").is_file():
            return root
        if os.name != "nt" and (root / "bin" / "qlh-app").is_file():
            return root
        if (root / "src" / "api_server.py").is_file():
            return root
        # 瘦身安装包（PyInstaller datas 把 src 收进 _internal/）
        if (root / "_internal" / "src" / "api_server.py").is_file():
            return root
    return None


def quick_verify_after_launch_failure(
    preferred_variant: str | None = None,
) -> tuple[Path, dict] | None:
    """Run the bounded UP-N6.1 check only after a startup call fails."""
    candidates: list[Path] = []
    detected = installed_app_root(preferred_variant)
    if detected is not None:
        candidates.append(detected)
    candidates.extend(_candidate_app_roots(preferred_variant))
    seen: set[str] = set()
    for root in candidates:
        try:
            candidate = root.expanduser().absolute()
        except OSError:
            continue
        key = str(candidate).casefold()
        if key in seen:
            continue
        seen.add(key)
        if not (candidate / "manifest" / "install-manifest.json").is_file():
            continue
        return candidate, verify_install_tree(candidate, level="quick")
    return None


def diagnosis_app_root(preferred_variant: str | None = None) -> Path | None:
    """Find an application root for read-only diagnosis without following links."""
    detected = installed_app_root(preferred_variant)
    if detected is not None:
        return detected.expanduser().absolute()
    for root in _candidate_app_roots(preferred_variant):
        candidate = root.expanduser().absolute()
        if (candidate / "manifest" / "install-manifest.json").is_file():
            return candidate
    return None


def _slim_src_root(root: Path) -> Path | None:
    """瘦身安装包识别：``_internal/src/api_server.py`` 存在、且**包内未捆绑服务依赖**
    （``fastapi`` 被 spec 的 ``excludes`` 排除）→ 返回 ``_internal`` 作为 uvicorn 的 cwd。

    判据用「**包内有没有捆绑 fastapi**」，而不是「根目录有没有 ``QLH-Edge-Inference.exe``」：
    PyInstaller onedir 产物的根目录**本来就带着这个引导器 exe**，拿它当判据会把
    dist 形态的瘦身包误判成完整包 ⇒ 走包内同进程 import ⇒ 撞上被排除的 fastapi
    ⇒ `ModuleNotFoundError: No module named 'fastapi'`（2026-10-06 设备侧实测踩到）。

    瘦身包形态仅 **Windows Inno 安装**；Linux deb 已用包内 venv 运行时形态，
    不适用本分支。非 nt 直接返回 None —— 也避免 Python 3.12 下 ``Path``
    按被 mock 的 ``os.name`` 实例化 PosixPath 而崩（Linux 模拟单测）。"""
    if os.name != "nt":
        return None
    internal = Path(root).expanduser() / "_internal"
    if not (internal / "src" / "api_server.py").is_file():
        return None
    # 包内自带 fastapi ⇒ 完整包，可以在包内同进程起服务
    if (internal / "fastapi" / "__init__.py").is_file():
        return None
    return internal


def _requirements_path(root: Path, engine: str, profile: str | None = None) -> Path:
    """外部运行时清单位置：优先安装树内 _internal/packaging/，其次打包树根。"""
    # Keep the shipped CPU/CUDA filenames stable; the new Edge profile gets
    # its own name so an existing installation can be upgraded in place.
    filename = (
        "requirements-runtime-llama_cpp_only.txt"
        if profile == "llama_cpp_only"
        else f"requirements-runtime-{engine}.txt"
    )
    candidates = [
        Path(root).expanduser() / "_internal" / "packaging"
        / filename,
        Path(root).expanduser() / "packaging" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def bundled_runtime_profile(root: Path) -> str | None:
    """Return the runtime profile this package was built with, or None.

    The slim package ships exactly one `requirements-runtime-*.txt` — the one
    for the profile `qlh-slim.spec` was told to build (`_RUNTIME_REQS`) — so
    "which manifest is inside" *is* "which profile this package is".  None when
    no manifest is present (source checkout, legacy package) or more than one is
    (a broken package): the caller then falls back to the engine default.
    """
    from runtime_guard import PROFILE_REQUIREMENTS_FILENAME

    for directory in (
        Path(root).expanduser() / "_internal" / "packaging",
        Path(root).expanduser() / "packaging",
    ):
        if not directory.is_dir():
            continue
        present = {
            profile
            for profile, filename in PROFILE_REQUIREMENTS_FILENAME.items()
            if (directory / filename).is_file()
        }
        if present:
            return present.pop() if len(present) == 1 else None
    return None


def _runtime_proxy() -> str:
    """运行时引导代理：QLH_RUNTIME_PROXY > QLH_HTTP_PROXY > 空。"""
    return (
        os.environ.get("QLH_RUNTIME_PROXY", "").strip()
        or os.environ.get("QLH_HTTP_PROXY", "").strip()
        or ""
    )


def runtime_app_command(
    root: Path, *, engine: str | None = None, profile: str | None = None,
) -> list[str] | None:
    """瘦身包：确保外部运行时就绪（缺失则 pip 引导），返回以外部 runtime venv 的
    python 启动 uvicorn 的命令。引擎缺失/引导失败 → None。"""
    from runtime_guard import RuntimeContext, runtime_dir, venv_python, ensure_runtime
    engine = (engine or os.environ.get("QLH_RUNTIME_ENGINE", "cpu")).strip().lower()
    if engine not in {"cpu", "cuda"}:
        engine = "cpu"
    root = Path(root).expanduser()
    profile = (profile or os.environ.get("QLH_RUNTIME_PROFILE", "")).strip().lower() or None
    if profile is None:
        profile = bundled_runtime_profile(root)
    if profile is None:
        profile = "torch_cuda" if engine == "cuda" else "torch_cpu"
    requirements = _requirements_path(root, engine, profile)
    ctx = RuntimeContext(
        root=root, engine=engine, profile=profile,
        requirements=requirements, proxy=_runtime_proxy(),
    )
    report = ensure_runtime(ctx)
    if report["state"] not in {"ok", "installed"}:
        return None
    python = venv_python(runtime_dir())
    return [
        str(python), "src/app_bootstrap.py",
    ]


def _runtime_check_main(args: Any) -> int:
    """只读诊断：检查/引导外部运行时并输出 JSON 状态（不启动主程序）。"""
    from runtime_guard import RuntimeContext, runtime_dir, venv_python, ensure_runtime
    root = Path(args.root).expanduser() if args.root else (
        installed_app_root(args.variant) or Path.cwd()
    )
    engine = (getattr(args, "runtime_engine", None)
              or os.environ.get("QLH_RUNTIME_ENGINE", "cpu")).strip().lower()
    if engine not in {"cpu", "cuda"}:
        engine = "cpu"
    profile = (getattr(args, "runtime_profile", None)
               or os.environ.get("QLH_RUNTIME_PROFILE", "")).strip().lower() or None
    if profile is None:
        profile = bundled_runtime_profile(root)
    if profile is None:
        profile = "torch_cuda" if engine == "cuda" else "torch_cpu"
    requirements = _requirements_path(root, engine, profile)
    ctx = RuntimeContext(
        root=root, engine=engine, profile=profile,
        requirements=requirements, proxy=_runtime_proxy(),
    )
    report = ensure_runtime(ctx, create_venv=False, install_missing=False)
    keys = ("state", "created", "exit", "missing_before", "missing_after", "error")
    summary = {
        "state": report["state"],
        "profile": report.get("profile", profile),
        "runtime_python": str(venv_python(runtime_dir())),
    }
    for key in keys:
        if key in report:
            summary[key] = report[key]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["state"] in {"ok", "installed"} else 1


def app_command(mode: str, variant_override: str | None = None) -> list[str]:
    root = installed_app_root(variant_override)
    if root is None:
        raise FileNotFoundError("未检测到 QLH 主程序；请先检查更新并安装应用包")
    try:
        # A previous explicit reinstall leaves a marker outside the install
        # tree.  Consume it before launching the freshly installed app.
        auto_reassociate_user_data(root)
    except DataRetentionError as exc:
        raise OSError(f"保留数据重新关联失败: {exc}") from exc
    # ---- 瘦身包：无打包 exe，外部运行时由 runtime_guard 引导，以源码跑 uvicorn ----
    slim = _slim_src_root(root)
    if slim is not None:
        command = runtime_app_command(root, engine=variant_override)
        if command is not None:
            return [*command, f"--{mode}"]
        raise FileNotFoundError(
            "外部运行时依赖未就绪或引导失败（CPU/CUDA）；请运行 --runtime-check 查看",
        )
    if os.name == "nt":
        app = root / "QLH-Edge-Inference.exe"
        if app.is_file():
            return [str(app), f"--{mode}"]
    else:
        installed_app = root / "bin" / "qlh-app"
        if installed_app.is_file():
            # Linux 包内 venv 是主应用的唯一依赖边界；仅旧包缺失时才回退 shebang。
            venv_python = root / "venv" / "bin" / "python"
            if venv_python.is_file():
                return [str(venv_python), str(installed_app), f"--{mode}"]
            return [str(installed_app), f"--{mode}"]
    legacy = root / "src" / "app_bootstrap.py"
    if legacy.is_file():
        return [sys.executable, str(legacy), f"--{mode}"]
    raise FileNotFoundError("QLH 安装目录存在，但主程序入口缺失；请执行覆盖安装")


def launch_app(mode: str, variant_override: str | None = None) -> subprocess.Popen:
    command = app_command(mode, variant_override)
    root = installed_app_root(variant_override) or install_root()
    slim = _slim_src_root(root)
    # 瘦身包以 _internal 作为 cwd（uvicorn 需可 import src.api_server）
    cwd = str(slim) if slim is not None else str(root)
    try:
        environment = build_release_child_environment(root)
    except ReleaseContractError as exc:
        raise OSError(f"发行合同无效，拒绝启动主程序: {exc}") from exc
    return subprocess.Popen(command, cwd=cwd, env=environment)


















def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qlh-maintenance",
        description="Signed application verification, maintenance, runtime and data commands.",
    )
    parser.add_argument(
        "command", nargs="?", default="verify",
        choices=(
            "check", "download", "install", "start",
            "version-status", "version-stage", "version-activate",
            "version-rollback", "version-recover",
            "diagnostics", "verify", "diagnose", "repair",
            "data-status", "retain-data", "reassociate-data", "reinstall",
        ),
    )
    parser.add_argument("--check-update", action="store_true", help="alias for check")
    parser.add_argument("--source", action="append", help="manifest URL; may be repeated")
    parser.add_argument("--variant", choices=("cpu", "cuda"), help="application package variant")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--download-dir")
    parser.add_argument("--version-store")
    parser.add_argument("--bundle")
    parser.add_argument("--version")
    parser.add_argument("--diagnostics-output")
    parser.add_argument("--diagnose-report", help="bundle-safe diagnosis JSON for diagnostics")
    parser.add_argument("--trusted-keys-dir")
    parser.add_argument("--root", help="install root for verify, diagnose, or repair")
    parser.add_argument("--data-root", help="external user-data root for UP-N6.4")
    parser.add_argument("--level", choices=("quick", "full", "deep"), default="quick")
    parser.add_argument("--runtime-check", action="store_true",
                        help="检查/引导外部运行时依赖（瘦身包），不启动主程序")
    parser.add_argument("--runtime-engine", choices=("cpu", "cuda"),
                        help="外部运行时引擎（瘦身包）；默认 QLH_RUNTIME_ENGINE 或 cpu")
    parser.add_argument("--runtime-profile", choices=("llama_cpp_only", "torch_cpu", "torch_cuda"),
                        help="外部运行时依赖 profile；默认按引擎选择 torch_cpu/torch_cuda，"
                             "Edge 无 torch 使用 llama_cpp_only")
    parser.add_argument("--error", help="optional local startup error text for diagnose")
    parser.add_argument("--diagnosis-output", help="local JSON output path for diagnose")
    parser.add_argument("--repair-output", help="local JSON output path for repair")
    parser.add_argument("--no-gpu-probe", action="store_true", help="skip the bounded CUDA driver probe")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--allow-unsigned", action="store_true")
    parser.add_argument("--headless", action="store_true", help="run the application backend")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Forward the runtime selection to the slim app path: the child process — and
    # the cluster presence payload it later reports — inherit it.  An explicit
    # flag wins; otherwise the profile the package was built with is used, so a
    # shipped Windows SLIM advertises `llama_cpp_only` without the installer
    # having to pass anything.  The launcher itself stays dependency-free.
    runtime_profile = args.runtime_profile
    if not runtime_profile and not os.environ.get("QLH_RUNTIME_PROFILE", "").strip():
        runtime_profile = bundled_runtime_profile(
            installed_app_root(args.variant) or Path.cwd()
        )
    if runtime_profile:
        os.environ["QLH_RUNTIME_PROFILE"] = runtime_profile
    if args.runtime_engine:
        os.environ["QLH_RUNTIME_ENGINE"] = args.runtime_engine
    if args.runtime_check:
        return _runtime_check_main(args)
    if args.command == "start":
        try:
            return launch_app("headless" if args.headless else "tui", args.variant).wait()
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.command == "verify":
        root = Path(args.root).expanduser() if args.root else installed_app_root(args.variant)
        if root is None:
            print("未检测到带签名安装清单的 QLH 主程序；请使用 --root 指定安装目录。", file=sys.stderr)
            return 2
        forwarded = ["verify", "--root", str(root), "--level", args.level]
        if args.trusted_keys_dir:
            forwarded.extend(("--trusted-keys-dir", args.trusted_keys_dir))
        if args.as_json:
            forwarded.append("--json")
        return install_manifest_main(forwarded)
    if args.command == "diagnose":
        root = Path(args.root).expanduser() if args.root else diagnosis_app_root(args.variant)
        if root is None:
            print("未检测到 QLH 主程序；请使用 --root 指定安装目录。", file=sys.stderr)
            return 2
        report = diagnose_install(
            root,
            error=args.error,
            trusted_keys_dir=args.trusted_keys_dir,
            probe_gpu=not args.no_gpu_probe,
        )
        if args.diagnosis_output:
            write_diagnosis_report(report, args.diagnosis_output)
        if args.as_json:
            import json

            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        else:
            print(format_diagnosis(report))
        return 0 if report["ok"] else 3
    if args.command == "repair":
        root = Path(args.root).expanduser() if args.root else diagnosis_app_root(args.variant)
        if root is None:
            print("未检测到 QLH 主程序；请使用 --root 指定安装目录。", file=sys.stderr)
            return 2
        try:
            report = repair_install(
                root,
                sources=args.source or configured_sources(),
                profile=detect_profile(variant_override=args.variant),
                trusted_keys_dir=args.trusted_keys_dir,
                timeout=args.timeout,
                download_dir=args.download_dir,
            )
        except RepairError as exc:
            report = {"ok": False, "action": "failed", "error": str(exc)}
        if args.repair_output:
            output = Path(args.repair_output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                __import__("json").dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if args.as_json:
            import json

            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        else:
            print(format_repair(report))
        return 0 if report.get("ok") else 3
    if args.command in {"data-status", "retain-data", "reassociate-data"}:
        root = Path(args.root).expanduser() if args.root else diagnosis_app_root(args.variant)
        if root is None:
            print("未检测到 QLH 主程序；请使用 --root 指定安装目录。", file=sys.stderr)
            return 2
        try:
            if args.command == "data-status":
                report = retention_status(root, args.data_root)
            elif args.command == "retain-data":
                report = retain_user_data(root, args.data_root, confirm=args.yes)
            else:
                report = reassociate_user_data(root, args.data_root, confirm=args.yes)
        except DataRetentionError as exc:
            report = {"ok": False, "action": "failed", "error": str(exc)}
        if args.as_json:
            print(__import__("json").dumps(report, ensure_ascii=False, sort_keys=True))
        else:
            print(__import__("json").dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.get("ok") else 3
    if args.command == "reinstall":
        if args.data_root:
            print("自动重装仅支持默认保留数据目录；自定义目录请分别执行 retain-data 与 reassociate-data。", file=sys.stderr)
            return 2
        root = Path(args.root).expanduser() if args.root else diagnosis_app_root(args.variant)
        if root is None:
            print("未检测到 QLH 主程序；请使用 --root 指定安装目录。", file=sys.stderr)
            return 2
        try:
            retained = retain_user_data(root, confirm=args.yes)
        except DataRetentionError as exc:
            retained = {"ok": False, "action": "failed", "error": str(exc)}
        if not retained.get("ok"):
            if args.as_json:
                print(__import__("json").dumps(retained, ensure_ascii=False, sort_keys=True))
            else:
                print(__import__("json").dumps(retained, ensure_ascii=False, indent=2, sort_keys=True))
            return 3
        forwarded = ["install", "--yes"]
        for source in args.source or []:
            forwarded.extend(("--source", source))
        if args.variant:
            forwarded.extend(("--variant", args.variant))
        if args.download_dir:
            forwarded.extend(("--download-dir", args.download_dir))
        if args.timeout != 8.0:
            forwarded.extend(("--timeout", str(args.timeout)))
        if args.as_json:
            forwarded.append("--json")
        return updater_main(forwarded)
    if args.command and args.command.startswith("version-"):
        forwarded = [args.command]
        for name in ("version_store", "bundle", "version", "variant"):
            value = getattr(args, name, None)
            if value:
                forwarded.extend((f"--{name.replace('_', '-')}", value))
        if args.as_json:
            forwarded.append("--json")
        return updater_main(forwarded)
    if args.command == "diagnostics":
        forwarded = [args.command]
        for source in args.source or []:
            forwarded.extend(("--source", source))
        for name in ("bundle", "version", "download_dir", "variant", "diagnostics_output", "diagnose_report", "trusted_keys_dir"):
            value = getattr(args, name, None)
            if value:
                forwarded.extend((f"--{name.replace('_', '-')}", str(value)))
        if args.timeout != 8.0:
            forwarded.extend(("--timeout", str(args.timeout)))
        for flag in ("as_json", "yes", "allow_unsigned"):
            if getattr(args, flag):
                forwarded.append("--" + ("json" if flag == "as_json" else flag.replace("_", "-")))
        return updater_main(forwarded)
    if args.check_update or args.command == "check":
        forwarded = ["check"]
        for source in args.source or []:
            forwarded.extend(("--source", source))
        if args.as_json:
            forwarded.append("--json")
        if args.variant:
            forwarded.extend(("--variant", args.variant))
        if args.timeout != 8.0:
            forwarded.extend(("--timeout", str(args.timeout)))
        return updater_main(forwarded)
    if args.command in {"download", "install"}:
        forwarded = [args.command]
        for source in args.source or []:
            forwarded.extend(("--source", source))
        if args.variant:
            forwarded.extend(("--variant", args.variant))
        if args.download_dir:
            forwarded.extend(("--download-dir", args.download_dir))
        if args.timeout != 8.0:
            forwarded.extend(("--timeout", str(args.timeout)))
        for flag in ("as_json", "yes", "allow_unsigned"):
            if getattr(args, flag):
                forwarded.append("--" + ("json" if flag == "as_json" else flag.replace("_", "-")))
        return updater_main(forwarded)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
