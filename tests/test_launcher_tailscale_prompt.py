"""Tailscale first-start choices and dialog ownership without desktop/network I/O."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


LAUNCHER_PATH = Path(__file__).resolve().parents[2] / "src" / "app_bootstrap.py"
NO_TAILSCALE = {
    "installed": False,
    "running": False,
    "logged_in": False,
    "tailscale_ip": None,
}
CONNECTED = {
    "installed": True,
    "running": True,
    "logged_in": True,
    "tailscale_ip": "100.64.0.1",
}


@pytest.fixture()
def launcher(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("qlh_tailscale_prompt_test", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_TAILSCALE_FLAG_DIR", str(tmp_path))
    monkeypatch.setattr(module, "_TAILSCALE_FLAG_FILE", str(tmp_path / ".tailscale_joined"))
    monkeypatch.setattr(module, "_has_interactive_stdin", Mock(return_value=False))
    monkeypatch.setattr(module, "_check_tailscale_status", Mock(return_value=NO_TAILSCALE.copy()))
    monkeypatch.setattr(module, "_show_dialog", Mock(return_value=module._IDNO))
    monkeypatch.setattr(module, "_open_url", Mock())
    monkeypatch.setattr(module, "_safe_input", Mock(side_effect=AssertionError("unexpected console input")))
    for name in ("_show_windows_messagebox", "_show_linux_dialog"):
        monkeypatch.setattr(module, name, Mock(side_effect=AssertionError("real dialog forbidden")))
    return module


def test_no_console_without_tailscale_uses_default_dialog_owner(launcher):
    assert launcher._prompt_tailscale_setup(NO_TAILSCALE) is True
    launcher._show_dialog.assert_called_once()
    assert launcher._show_dialog.call_args.kwargs == {"owner_hwnd": None}
    launcher._open_url.assert_not_called()
    launcher._safe_input.assert_not_called()


@pytest.mark.parametrize("has_marker", [False, True], ids=["first-start", "reconnect"])
@pytest.mark.parametrize(
    ("button", "expected", "open_links"),
    [("_IDYES", False, True), ("_IDNO", True, False), ("_IDCANCEL", False, False)],
    ids=["yes", "no", "cancel"],
)
def test_gui_choices_forward_owner_through_requirement_check(
    launcher, has_marker, button, expected, open_links,
):
    if has_marker:
        Path(launcher._TAILSCALE_FLAG_FILE).write_text("joined=1\n", encoding="utf-8")
    launcher._show_dialog.return_value = getattr(launcher, button)

    assert launcher._check_tailscale_requirement(owner_hwnd=0x1234) is expected

    launcher._check_tailscale_status.assert_called_once_with()
    launcher._show_dialog.assert_called_once()
    args, kwargs = launcher._show_dialog.call_args
    assert args[0] == "Tailscale 组网检查"
    assert "未检测到 Tailscale" in args[1]
    assert args[2] == "yesnocancel"
    assert kwargs == {"owner_hwnd": 0x1234}
    assert [call.args[0] for call in launcher._open_url.call_args_list] == (
        [launcher.TAILSCALE_DOWNLOAD_URL, launcher.TAILSCALE_INVITE_URL] if open_links else []
    )
    launcher._safe_input.assert_not_called()


def test_gui_yes_when_tailscale_installed_opens_only_invitation(launcher):
    launcher._check_tailscale_status.return_value = {**NO_TAILSCALE, "installed": True}
    launcher._show_dialog.return_value = launcher._IDYES

    assert launcher._check_tailscale_requirement(owner_hwnd=0x1234) is False

    launcher._open_url.assert_called_once_with(launcher.TAILSCALE_INVITE_URL)
    assert launcher._show_dialog.call_args.kwargs == {"owner_hwnd": 0x1234}


@pytest.mark.parametrize("has_marker", [False, True], ids=["first-start", "marked-online"])
def test_connected_tailscale_bypasses_prompt(launcher, has_marker):
    if has_marker:
        Path(launcher._TAILSCALE_FLAG_FILE).write_text("joined=1\n", encoding="utf-8")
    launcher._check_tailscale_status.return_value = CONNECTED.copy()

    assert launcher._check_tailscale_requirement(owner_hwnd=0x1234) is True

    launcher._show_dialog.assert_not_called()
    launcher._open_url.assert_not_called()
    launcher._safe_input.assert_not_called()


@pytest.mark.parametrize(("choice", "expected"), [("skip", True), ("no", False), ("yes", True)])
def test_console_choices_remain_interactive_without_gui(launcher, choice, expected):
    launcher._has_interactive_stdin.return_value = True
    launcher._safe_input.side_effect = None
    launcher._safe_input.return_value = choice
    launcher._check_tailscale_status.side_effect = [NO_TAILSCALE.copy(), CONNECTED.copy()]

    assert launcher._check_tailscale_requirement(owner_hwnd=0x1234) is expected

    launcher._safe_input.assert_called_once()
    launcher._show_dialog.assert_not_called()
    launcher._open_url.assert_not_called()
    marker = Path(launcher._TAILSCALE_FLAG_FILE)
    assert marker.exists() is (choice == "yes")
    if choice == "yes":
        assert "tailscale_ip=100.64.0.1" in marker.read_text(encoding="utf-8")
        assert launcher._check_tailscale_status.call_count == 2
    else:
        launcher._check_tailscale_status.assert_called_once_with()
