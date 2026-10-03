"""瘦身包自描述 profile：包内清单 → 构建期选定的 runtime profile。

`qlh-slim.spec` 只把选定 profile 的那一份 `requirements-runtime-*.txt` 打进包，
因此"包内是哪份清单"就是"这个包是什么 profile"。引导器据此在没有任何命令行
参数的情况下也能按正确 profile 检查/引导外部运行时，并把同一个值交给子进程
（子进程再随 device_info 上报给主节点）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packaging"))

import qlh_launcher as launcher  # noqa: E402
import runtime_guard as guard  # noqa: E402


def _package(tmp_path: Path, *filenames: str, inner: bool = False) -> Path:
    root = tmp_path / "QLH-Edge-Inference"
    target = root / "_internal" / "packaging" if inner else root / "packaging"
    target.mkdir(parents=True)
    for name in filenames:
        (target / name).write_text("", encoding="utf-8")
    return root


def test_bundled_profile_is_the_single_manifest_inside(tmp_path):
    root = _package(
        tmp_path, guard.PROFILE_REQUIREMENTS_FILENAME["llama_cpp_only"], inner=True
    )

    assert launcher.bundled_runtime_profile(root) == "llama_cpp_only"


@pytest.mark.parametrize("profile", ["torch_cpu", "torch_cuda"])
def test_each_profile_is_recovered_from_its_manifest(tmp_path, profile):
    root = _package(tmp_path, guard.PROFILE_REQUIREMENTS_FILENAME[profile], inner=True)

    assert launcher.bundled_runtime_profile(root) == profile


def test_missing_manifest_returns_none(tmp_path):
    root = _package(tmp_path, "README.txt", inner=True)

    assert launcher.bundled_runtime_profile(root) is None


def test_multiple_manifests_return_none(tmp_path):
    """一个包只该带一份清单；多于一份说明包被改过，不做猜测。"""
    root = _package(
        tmp_path,
        guard.PROFILE_REQUIREMENTS_FILENAME["torch_cpu"],
        guard.PROFILE_REQUIREMENTS_FILENAME["torch_cuda"],
        inner=True,
    )

    assert launcher.bundled_runtime_profile(root) is None


def test_install_tree_wins_over_build_tree(tmp_path):
    root = _package(
        tmp_path, guard.PROFILE_REQUIREMENTS_FILENAME["llama_cpp_only"], inner=True
    )
    build_tree = root / "packaging"
    build_tree.mkdir(parents=True, exist_ok=True)
    (build_tree / guard.PROFILE_REQUIREMENTS_FILENAME["torch_cpu"]).write_text(
        "", encoding="utf-8"
    )

    assert launcher.bundled_runtime_profile(root) == "llama_cpp_only"


def test_runtime_app_command_uses_bundled_profile(tmp_path, monkeypatch):
    root = _package(
        tmp_path, guard.PROFILE_REQUIREMENTS_FILENAME["llama_cpp_only"], inner=True
    )
    monkeypatch.delenv("QLH_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("QLH_RUNTIME_ENGINE", raising=False)
    seen: dict[str, object] = {}

    def fake_ensure(ctx):
        seen["profile"] = ctx.profile
        return {"state": "ok"}

    monkeypatch.setattr(guard, "ensure_runtime", fake_ensure)
    monkeypatch.setattr(guard, "runtime_dir", lambda: tmp_path / "runtime")
    monkeypatch.setattr(
        guard, "venv_python", lambda directory: directory / "Scripts" / "python.exe"
    )

    command = launcher.runtime_app_command(root)

    assert seen["profile"] == "llama_cpp_only"
    assert command is not None and "src.api_server:app" in command
