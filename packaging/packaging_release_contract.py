"""Read the canonical release contract without importing the app runtime."""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


CONTRACT_FILENAME = "release-contract.json"
_ROLE_ALIASES = {
    "master": "master",
    "auto": "auto",
    "client": "client",
    "worker": "client",
    "slave": "client",
}


class ReleaseContractError(RuntimeError):
    pass


def _candidate_paths(root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("QLH_RELEASE_CONTRACT_PATH", "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    if root is not None:
        candidates.extend(
            (root / CONTRACT_FILENAME, root / "_internal" / CONTRACT_FILENAME)
        )
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / CONTRACT_FILENAME)
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / CONTRACT_FILENAME)
    core_override = os.environ.get("QLH_CORE_ROOT", "").strip()
    if core_override:
        candidates.append(Path(core_override).expanduser() / CONTRACT_FILENAME)
    candidates.append(Path(__file__).resolve().parents[2] / CONTRACT_FILENAME)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def find_release_contract(root: Path | None = None) -> Path:
    for path in _candidate_paths(root):
        if path.is_file():
            return path
    raise ReleaseContractError("release-contract.json is unavailable")


@lru_cache(maxsize=8)
def _load_path(path: str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseContractError(f"invalid release contract {path}: {exc}") from exc
    if data.get("schema_version") != 1:
        raise ReleaseContractError(f"unsupported release contract schema: {path}")
    return data


def load_release_contract(root: Path | None = None) -> dict[str, Any]:
    return _load_path(str(find_release_contract(root)))


_SOURCE_CONTRACT = load_release_contract()
PRODUCT_VERSION = str(_SOURCE_CONTRACT["version"]["product"])
LAUNCHER_VERSION = str(_SOURCE_CONTRACT["version"]["launcher"])


def _node_config_path(environment: Mapping[str, str]) -> Path:
    override = str(environment.get("QLH_NODE_CONFIG_PATH", "")).strip()
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(environment.get("LOCALAPPDATA") or Path.home())
        return base / "QLH-Edge-Inference" / "node_config.json"
    base = Path(environment.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "qlh" / "node_config.json"


def _confirmed_role(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    node = data.get("node") if isinstance(data, dict) else None
    if not isinstance(node, dict):
        return None
    if not bool(node.get("role_confirmed", False) or data.get("bootstrapped", False)):
        return None
    raw = str(node.get("role", "")).strip().lower()
    if raw not in _ROLE_ALIASES:
        raise ReleaseContractError(f"invalid confirmed node role in {path}: {raw!r}")
    return _ROLE_ALIASES[raw]


def build_release_child_environment(
    root: Path, environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construct the explicit environment for an installed app child process."""
    child = dict(os.environ if environment is None else environment)
    contract_path = find_release_contract(Path(root).expanduser())
    contract = _load_path(str(contract_path))
    profile = contract["profile"]
    child["QLH_RELEASE_CONTRACT_PATH"] = str(contract_path)
    child["QLH_RELEASE_PROFILE_ENFORCE"] = "1"
    child["QLH_RELEASE_PROFILE"] = str(profile["name"])
    for name, value in profile["fixed_environment"].items():
        child[str(name)] = str(value)

    config_path = _node_config_path(child)
    child["QLH_NODE_CONFIG_PATH"] = str(config_path)
    confirmed = _confirmed_role(config_path)
    explicit = str(child.get("QLH_NODE_ROLE", "")).strip().lower()
    if confirmed is not None:
        role = confirmed
    elif explicit:
        if explicit not in _ROLE_ALIASES:
            raise ReleaseContractError(f"invalid QLH_NODE_ROLE: {explicit!r}")
        role = _ROLE_ALIASES[explicit]
    else:
        role = str(profile["default_node_role"])
    child["QLH_NODE_ROLE"] = role
    return child
