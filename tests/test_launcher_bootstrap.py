import os
import sys
from pathlib import Path


PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

import app_maintenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = Path(os.environ.get("QLH_CORE_ROOT", PROJECT_ROOT.parent / "qlh"))


def test_development_launcher_finds_legacy_application_entry():
    command = app_maintenance.app_command("ui")
    assert command[-2].endswith("src/app_bootstrap.py") or command[-2].endswith("src\\app_bootstrap.py")
    assert command[-1] == "--ui"






def test_headless_launch_returns_success_code(monkeypatch):
    launched = []
    monkeypatch.setattr(app_maintenance, "launch_app", lambda mode, _variant=None: launched.append(mode) or type("Process", (), {"wait": lambda self: 0})())
    assert app_maintenance.main(["start", "--headless"]) == 0
    assert launched == ["headless"]






def test_verify_command_forwards_selected_root_and_json(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(app_maintenance, "installed_app_root", lambda _variant: tmp_path)
    monkeypatch.setattr(app_maintenance, "install_manifest_main", lambda argv: captured.append(argv) or 7)

    assert app_maintenance.main(["verify", "--level", "deep", "--json"]) == 7
    assert captured == [["verify", "--root", str(tmp_path), "--level", "deep", "--json"]]


def test_diagnose_command_uses_detected_root_and_structured_output(monkeypatch, tmp_path, capsys):
    report = {"ok": True, "issues": [], "integrity": {"ok": True}}
    monkeypatch.setattr(app_maintenance, "diagnosis_app_root", lambda _variant: tmp_path)
    monkeypatch.setattr(app_maintenance, "diagnose_install", lambda root, **kwargs: report)

    assert app_maintenance.main(["diagnose", "--json"]) == 0
    assert __import__("json").loads(capsys.readouterr().out) == report


def test_repair_command_uses_detected_root_and_structured_output(monkeypatch, tmp_path, capsys):
    report = {"ok": True, "action": "repaired", "repaired": ["app.exe"]}
    monkeypatch.setattr(app_maintenance, "diagnosis_app_root", lambda _variant: tmp_path)
    monkeypatch.setattr(app_maintenance, "repair_install", lambda root, **kwargs: report)

    assert app_maintenance.main(["repair", "--source", "https://updates.example/latest.json", "--json"]) == 0
    assert __import__("json").loads(capsys.readouterr().out) == report


def test_retain_data_command_requires_explicit_confirmation_and_forwards_root(monkeypatch, tmp_path, capsys):
    captured = {}
    monkeypatch.setattr(app_maintenance, "diagnosis_app_root", lambda _variant: tmp_path)
    monkeypatch.setattr(
        app_maintenance,
        "retain_user_data",
        lambda root, data_root, *, confirm: captured.update(
            {"root": root, "data_root": data_root, "confirm": confirm}
        ) or {"ok": True, "action": "retained"},
    )

    assert app_maintenance.main(["retain-data", "--yes", "--data-root", str(tmp_path / "data"), "--json"]) == 0
    assert captured == {"root": tmp_path, "data_root": str(tmp_path / "data"), "confirm": True}
    assert __import__("json").loads(capsys.readouterr().out)["action"] == "retained"


def test_reinstall_retains_data_before_starting_signed_installer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(app_maintenance, "diagnosis_app_root", lambda _variant: tmp_path)
    monkeypatch.setattr(
        app_maintenance,
        "retain_user_data",
        lambda root, *, confirm: calls.append(("retain", root, confirm)) or {"ok": True, "action": "retained"},
    )
    monkeypatch.setattr(app_maintenance, "updater_main", lambda argv: calls.append(("install", argv)) or 0)

    assert app_maintenance.main(["reinstall", "--yes", "--source", "https://updates.example/latest.json"]) == 0
    assert calls[0] == ("retain", tmp_path, True)
    assert calls[1] == ("install", ["install", "--yes", "--source", "https://updates.example/latest.json"])






def test_windows_application_discovery_prefers_requested_variant(tmp_path, monkeypatch):
    cpu = tmp_path / "QLH-Edge-Inference"
    cuda = tmp_path / "QLH-Edge-Inference-CUDA"
    cpu.mkdir()
    cuda.mkdir()
    (cpu / "QLH-Edge-Inference.exe").write_bytes(b"cpu")
    (cuda / "QLH-Edge-Inference.exe").write_bytes(b"cuda")

    monkeypatch.setattr(
        app_maintenance, "_candidate_app_roots",
        lambda preferred: [cuda, cpu] if preferred == "cuda" else [cpu, cuda],
    )

    assert app_maintenance.installed_app_root("cpu") == cpu
    assert app_maintenance.installed_app_root("cuda") == cuda
    assert app_maintenance.app_command("ui", "cuda") == [
        str(cuda / "QLH-Edge-Inference.exe"), "--ui",
    ]




def test_linux_deb_build_keeps_bootstrap_and_venv_fallback_contract():
    script = (PROJECT_ROOT / "packaging" / "linux" / "build-deb.sh").read_text(
        encoding="utf-8"
    )

    assert '"$PACKAGING_DIR/app_maintenance.py"' in script
    assert '"$PACKAGING_DIR/diagnose.py"' in script
    assert '"$PACKAGING_DIR/repair.py"' in script
    assert '"$PACKAGING_DIR/data_retention.py"' in script
    assert '"$PACKAGING_DIR/update_core.py"' in script
    assert '"$PACKAGING_DIR/updater.py"' in script
    assert '"$PACKAGING_DIR/version_store.py"' in script
    assert '"$PACKAGING_DIR/signing.py"' in script
    assert '"$PACKAGING_DIR/pubkeys"' in script
    assert "python3 -c 'import venv'" in script
    assert "curl -fsSL --retry 3 https://bootstrap.pypa.io/get-pip.py" in script






def test_linux_application_command_prefers_packaged_venv(tmp_path, monkeypatch):
    app_root = tmp_path / "qlh-edge-inference"
    app = app_root / "bin" / "qlh-app"
    venv_python = app_root / "venv" / "bin" / "python"
    app.parent.mkdir(parents=True)
    venv_python.parent.mkdir(parents=True)
    app.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(app_maintenance, "installed_app_root", lambda *_args: app_root)
    monkeypatch.setattr(app_maintenance.os, "name", "posix")

    assert app_maintenance.app_command("tui") == [
        str(venv_python), str(app), "--tui",
    ]


def test_launcher_discovers_active_atomic_version_before_installed_root(tmp_path, monkeypatch):
    active = tmp_path / "versions" / "0.1.9-cpu"
    active.mkdir(parents=True)
    (active / "QLH-Edge-Inference.exe").write_bytes(b"active")
    monkeypatch.setenv("QLH_LAUNCHER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(app_maintenance, "install_root", lambda: tmp_path / "legacy")
    monkeypatch.setattr(app_maintenance.VersionStore, "active_path", lambda _self: active)

    assert app_maintenance.installed_app_root("cpu") == active
