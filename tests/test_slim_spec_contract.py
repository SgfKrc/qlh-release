"""Static contract for the source-backed slim PyInstaller spec."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_slim_source_tree_excludes_build_residue():
    spec = (ROOT / "packaging" / "qlh-slim.spec").read_text(encoding="utf-8")

    assert "from PyInstaller.building.datastruct import Tree" in spec
    assert "prefix=\"src\"" in spec
    for excluded in ("__pycache__", "*.pyc", "*.pyo", "*.log"):
        assert excluded in spec
    assert "a.datas + _SRC_TREE" in spec
