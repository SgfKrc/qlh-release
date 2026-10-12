"""Release adapter around the unchanged canonical conversion transaction.

Keep this module's __file__ at scripts/model_tools/gguf_convert.py so canonical
root/quantizer discovery resolves the installation rather than the worktree.
The canonical source remains a separate, byte-identical packaged resource.
"""
from pathlib import Path as _SupplyPath
import sys as _supply_sys

_supply_canonical = _SupplyPath(__file__).parent / "_canonical" / "gguf_convert.py"
exec(compile(_supply_canonical.read_bytes(), str(_supply_canonical), "exec"), globals())

from . import conversion_runtime as _supply_runtime

_dependency_probe = _supply_runtime.dependency_probe
_supply_root = _SupplyPath(__file__).resolve().parents[2]


def _find_converter(explicit):
    bundled = _supply_runtime.bundled_converter(_supply_root)
    if explicit is not None and explicit.resolve() != bundled.resolve():
        return None, "missing"
    return (bundled, "available") if bundled.is_file() else (None, "missing")


if getattr(_supply_sys, "frozen", False):
    def _tool_command(tool, *args):
        return _supply_runtime.tool_command(_supply_root, tool, *args)

    def _run_stage(command, *, timeout_seconds, stage, env):
        return _supply_runtime.run_stage(
            command, root=_supply_root, timeout_seconds=timeout_seconds,
            stage=stage, env=env,
        )
