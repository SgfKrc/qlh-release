"""Standard-library preflight and fresh-process entry for packaged HF conversion."""
from __future__ import annotations

import importlib.util
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

CONVERT_HF_FLAG = "--qlh-convert-hf-to-gguf"
CONVERTER_RELATIVE = Path("android/app/src/main/cpp/llama.cpp/convert_hf_to_gguf.py")
CONVERSION_DEPENDENCIES = (
    "numpy", "safetensors", "torch", "transformers", "tokenizers",
    "sentencepiece", "google.protobuf",
)


def bundled_converter(root: Path) -> Path:
    return root / CONVERTER_RELATIVE


def _available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def dependency_probe(converter: Path | None) -> dict:
    """Discover dependencies without importing Torch or inspecting tensor data.

    Missing requirements must never disable preflight (the canonical tool's
    not_probed fallback would otherwise make an incomplete Edge bundle ready).
    """
    if converter is None:
        return {"status": "unavailable", "checked": [], "missing": []}
    missing = [name for name in CONVERSION_DEPENDENCIES if not _available(name)]
    local_gguf = converter.parent / "gguf-py" / "gguf" / "__init__.py"
    if not local_gguf.is_file() and not _available("gguf"):
        missing.append("gguf")
    return {
        "status": "missing" if missing else "ready",
        "checked": [*CONVERSION_DEPENDENCIES, "gguf"], "missing": missing,
    }


def tool_command(root: Path, tool: Path, *args: str) -> list[str]:
    if tool.suffix.lower() != ".py":
        return [str(tool), *args]
    if getattr(sys, "frozen", False):
        if tool.resolve() != bundled_converter(root).resolve():
            raise ValueError("model_supply_converter_untrusted: expected bundled converter")
        return [sys.executable, CONVERT_HF_FLAG, *args]
    return [sys.executable, str(tool), *args]


def run_stage(command, *, root, timeout_seconds, stage, env):
    """Match canonical stage results while hiding native child console windows."""
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, cwd=str(root), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout_seconds, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired:
        code = "stage_timeout"
    except (OSError, ValueError):
        code = "stage_start_failed"
    else:
        elapsed = int((time.perf_counter() - started) * 1000)
        if completed.returncode:
            return {"stage": stage, "status": "failed", "code": "stage_failed",
                    "exit_code": int(completed.returncode), "elapsed_ms": elapsed}
        return {"stage": stage, "status": "passed", "elapsed_ms": elapsed}
    return {"stage": stage, "status": "failed", "code": code,
            "elapsed_ms": int((time.perf_counter() - started) * 1000)}


def run_converter(root: Path, argv) -> int:
    script = bundled_converter(root)
    required = [
        script, script.parent / "conversion" / "__init__.py",
        script.parent / "gguf-py" / "gguf" / "__init__.py",
        script.parent / "requirements" / "requirements-convert_hf_to_gguf.txt",
    ]
    for resource in required:
        if not resource.is_file():
            print(f"model_supply_resource_missing: {resource}", file=sys.stderr)
            return 2
    dependencies = dependency_probe(script)
    if dependencies["missing"]:
        print("converter_dependencies_missing: " + ", ".join(dependencies["missing"]),
              file=sys.stderr)
        return 2
    previous_argv, previous_path = sys.argv, sys.path[:]
    try:
        sys.argv = [str(script), *argv]
        # Use the converter revision's matching GGUF sources; other dependencies
        # (including NumPy/Torch native libraries) come from the frozen archive.
        sys.path[:0] = [str(script.parent / "gguf-py"), str(script.parent), str(root)]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path
    return 0
