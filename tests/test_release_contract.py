from __future__ import annotations

import json
from pathlib import Path

import pytest

import packaging_release_contract as release_contract


CORE_ROOT = Path(__file__).resolve().parents[2]


def test_packaging_versions_match_canonical_contract() -> None:
    contract = json.loads(
        (CORE_ROOT / "release-contract.json").read_text(encoding="utf-8")
    )
    assert release_contract.PRODUCT_VERSION == contract["version"]["product"]
    assert release_contract.LAUNCHER_VERSION == contract["version"]["launcher"]


def test_child_environment_uses_confirmed_role_and_fixed_product_profile(
    tmp_path: Path,
) -> None:
    root = tmp_path / "app"
    root.mkdir()
    (root / "release-contract.json").write_text(
        (CORE_ROOT / "release-contract.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config_path = tmp_path / "node.json"
    config_path.write_text(
        json.dumps({"node": {"role": "client", "role_confirmed": True}}),
        encoding="utf-8",
    )
    env = release_contract.build_release_child_environment(
        root,
        {
            "QLH_NODE_CONFIG_PATH": str(config_path),
            "QLH_ROUTE_A_STAGE_OFFER": "0",
            "QLH_TASK_GRAPH_ENABLED": "1",
        },
    )
    assert env["QLH_NODE_ROLE"] == "client"
    assert env["QLH_RELEASE_PROFILE_ENFORCE"] == "1"
    assert env["QLH_ROUTE_A_STAGE_OFFER"] == "1"
    assert env["QLH_TASK_WORKER_EXPERIMENTAL_ENABLED"] == "1"
    assert env["QLH_TASK_GRAPH_ENABLED"] == "0"


def test_child_environment_defaults_to_packaged_master(tmp_path: Path) -> None:
    root = tmp_path / "app"
    root.mkdir()
    (root / "release-contract.json").write_text(
        (CORE_ROOT / "release-contract.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    env = release_contract.build_release_child_environment(
        root, {"QLH_NODE_CONFIG_PATH": str(tmp_path / "missing.json")},
    )
    assert env["QLH_NODE_ROLE"] == "master"


def test_child_environment_rejects_invalid_explicit_role(tmp_path: Path) -> None:
    root = tmp_path / "app"
    root.mkdir()
    (root / "release-contract.json").write_text(
        (CORE_ROOT / "release-contract.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(release_contract.ReleaseContractError, match="invalid QLH_NODE_ROLE"):
        release_contract.build_release_child_environment(
            root,
            {
                "QLH_NODE_CONFIG_PATH": str(tmp_path / "missing.json"),
                "QLH_NODE_ROLE": "typo",
            },
        )
