"""Offline GGUF cutter dispatch for a fresh packaged application process.

Frozen callers use [sys.executable, "--qlh-cut-layers", *cut_args].  They must
never treat the frozen executable as a Python interpreter for a script path.
This module uses only the standard library and never starts an inference host.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from collections.abc import Sequence

CUT_LAYERS_FLAG = "--qlh-cut-layers"


def _open_inherited_output(standard_handle: int):
    """Own a duplicate of a Windows stdout/stderr handle, never the original.

    A windowed PyInstaller bootloader leaves sys.stdout/sys.stderr as None even
    when CreateProcess inherited working pipe handles from the caller.  Opening
    those handles directly would transfer their ownership to the CRT; duplicate
    first so closing Python's stream cannot close the process's original handle.
    """
    import ctypes
    import msvcrt
    import os
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.DuplicateHandle.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD, wintypes.BOOL, wintypes.DWORD,
    ]
    kernel32.DuplicateHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    inherited = kernel32.GetStdHandle(wintypes.DWORD(standard_handle))
    if inherited in (None, 0, ctypes.c_void_p(-1).value):
        raise OSError(f"GetStdHandle({standard_handle}) is unavailable")
    process = kernel32.GetCurrentProcess()
    duplicate = wintypes.HANDLE()
    if not kernel32.DuplicateHandle(
        process, inherited, process, ctypes.byref(duplicate), 0, False, 2,
    ):  # DUPLICATE_SAME_ACCESS; never make the new handle inheritable.
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(
            duplicate.value, os.O_WRONLY | os.O_BINARY | os.O_NOINHERIT,
        )
    except BaseException:
        kernel32.CloseHandle(duplicate)
        raise
    try:
        return os.fdopen(
            descriptor, "w", buffering=1, encoding="utf-8", errors="replace",
        )
    except BaseException:
        os.close(descriptor)
        raise


def _restore_helper_stdio() -> bool:
    """Reconnect missing output streams only inside the offline helper branch.

    No console is allocated or attached.  stdout/stderr must be supplied by the
    caller when using a windowed executable.  The cutter never consumes stdin.
    """
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return True
    failures = []
    for name, handle in (("stdout", -11), ("stderr", -12)):
        stream = getattr(sys, name)
        if stream is not None and not stream.closed:
            continue
        try:
            setattr(sys, name, _open_inherited_output(handle))
        except (OSError, ValueError) as exc:
            failures.append(f"{name}: {exc}")
    if not failures:
        return True
    message = "model_supply_stdio_unavailable: " + "; ".join(failures)
    for stream in (sys.stderr, sys.stdout, sys.__stderr__, sys.__stdout__):
        if stream is not None and not stream.closed:
            try:
                print(message, file=stream, flush=True)
                return False
            except (OSError, ValueError):
                continue
    # If both inherited pipes are absent there is physically no parent output
    # channel.  Preserve the named diagnostic for Windows debug listeners and
    # return exit 2; do not create a console or launch the application instead.
    import ctypes

    ctypes.windll.kernel32.OutputDebugStringW(message)
    return False


def run_cut_layers(argv: Sequence[str]) -> int:
    """Run the shipped generator, preserving its CLI, exit status and manifest.

    GGUFReader maps source tensors; this path does not load an inference model.
    Safetensors conversion is a separate capability even in a full/Torch build.
    """
    if not _restore_helper_stdio():
        return 2
    if getattr(sys, "frozen", False):
        root = Path(sys._MEIPASS)
    else:
        from repo_paths import core_root

        root = core_root()
    script = root / "scripts" / "cut_layers.py"
    if not script.is_file():
        print(f"model_supply_resource_missing: {script}", file=sys.stderr)
        return 2
    if getattr(sys, "frozen", False) and not (root / "gguf" / "__init__.py").is_file():
        print("model_supply_dependency_missing: bundled gguf sources", file=sys.stderr)
        return 2

    # Leave full validation/help to the canonical cutter.  Reject other source
    # formats before importing GGUF or any optional runtime dependencies.
    if not any(arg in ("--help", "-h") for arg in argv):
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument("--src")
        known, _ = parser.parse_known_args(argv)
        if known.src and Path(known.src).suffix.lower() != ".gguf":
            print(
                "model_supply_unsupported_source_format: --qlh-cut-layers supports "
                "GGUF only; convert Safetensors on a capable node or receive a "
                "segment from master.", file=sys.stderr,
            )
            return 2

    previous_argv, previous_path = sys.argv, sys.path[:]
    try:
        sys.argv = [str(script), *argv]
        # Explicit data sources are available to imports in the runpy script;
        # native dependencies are still provided by the frozen import archive.
        sys.path.insert(0, str(root))
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path
    return 0


def run_hf_conversion(argv: Sequence[str]) -> int:
    """Dispatch conversion before application startup, with named Edge refusal.

    This is a separate capability/process; the GGUF cutter never reaches Torch.
    """
    if not _restore_helper_stdio():
        return 2
    if not getattr(sys, "frozen", False):
        print("model_supply_conversion_entry_requires_frozen: use venv Python with the converter script",
              file=sys.stderr)
        return 2
    root = Path(sys._MEIPASS)
    resource = root / "scripts/model_tools/conversion_runtime.py"
    if not resource.is_file():
        print(f"model_supply_resource_missing: {resource}", file=sys.stderr)
        return 2
    import importlib.util

    spec = importlib.util.spec_from_file_location("_qlh_supply_conversion_runtime", resource)
    if spec is None or spec.loader is None:
        print("model_supply_resource_missing: conversion runtime loader", file=sys.stderr)
        return 2
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime.run_converter(root, argv)


def dispatch_helper(flag: str, argv: Sequence[str]) -> int:
    """Catch offline helper failures before a windowed bootloader can show UI."""
    try:
        if not _restore_helper_stdio():
            return 2
        if flag == CUT_LAYERS_FLAG:
            return run_cut_layers(argv)
        if flag == "--qlh-convert-hf-to-gguf":
            return run_hf_conversion(argv)
        raise ValueError("model_supply_helper_flag_invalid")
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(str(exc.code), file=sys.stderr)
        return 1
    except BaseException:
        import traceback

        diagnostic = "model_supply_helper_failed: " + flag + "\n" + traceback.format_exc()
        for stream in (sys.stderr, sys.stdout, sys.__stderr__, sys.__stdout__):
            if stream is not None and not stream.closed:
                try:
                    print(diagnostic, file=stream, flush=True)
                    return 1
                except (OSError, ValueError):
                    continue
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.OutputDebugStringW(diagnostic)
        return 1
