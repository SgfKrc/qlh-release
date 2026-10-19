#!/opt/qlh-edge-inference/venv/bin/python3
"""Installed application entry; package venv owns dependencies."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src"), str(root / "bin")]
if __name__ == "__main__":
    from app_bootstrap import main
    raise SystemExit(main())
