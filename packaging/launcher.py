"""Frozen application entry: helper stdout is dispatched before bootstrap imports."""
import os
import sys
from pathlib import Path
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    if sys.argv[1:2] in (["--qlh-cut-layers"], ["--qlh-convert-hf-to-gguf"]):
        from model_supply_runtime import dispatch_helper
        return dispatch_helper(sys.argv[1], sys.argv[2:])
    if sys.argv[1:2] == ["--maintenance"]:
        from app_maintenance import main as maintenance
        return maintenance(sys.argv[2:])
    from app_bootstrap import main as bootstrap
    return bootstrap()

if __name__ == "__main__":
    raise SystemExit(main())
