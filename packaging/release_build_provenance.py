"""Bind release build entrypoints to a fresh, unchanged main Git checkout.

The snapshot is generated before compilation, then checked and copied verbatim
into the new application tree immediately before install-manifest signing.
Dirty developer builds retain their true identity; release acceptance decides
whether that identity is admissible. Git-free exports require a future verified
export contract and are deliberately rejected, regardless of environment SHA.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


class ReleaseBuildSourceError(ValueError):
    """A named failure which must stop packaging before signing."""


def _provider(root: Path, packaging_root: Path):
    root, packaging_root = root.resolve(), packaging_root.resolve()
    if not (root / ".git").exists():
        raise ReleaseBuildSourceError(
            "SOURCE_CHECKOUT_REQUIRED: mount the real main Git checkout and fixed "
            "submodules; a Git-free export or environment revision is not build provenance"
        )
    if (packaging_root != (root / "packaging").resolve()
            or Path(__file__).resolve().parent.parent != packaging_root):
        raise ReleaseBuildSourceError(
            "PACKAGING_SOURCE_MISMATCH: the executing packaging scripts must belong "
            "to the main checkout's packaging submodule"
        )
    path = root / "scripts/package_contract/build_provenance.py"
    if not path.is_file():
        raise ReleaseBuildSourceError("SOURCE_PROVIDER_MISSING: main build_provenance.py is required")
    spec = importlib.util.spec_from_file_location("_qlh_release_build_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot_path(root: Path, output: Path) -> Path:
    if output.is_symlink():
        raise ReleaseBuildSourceError("BUILD_INFO_PATH_INVALID: snapshot must not be a symbolic link")
    output = output.resolve()
    try:
        relative = output.relative_to(root.resolve())
    except ValueError as exc:
        raise ReleaseBuildSourceError("BUILD_INFO_PATH_INVALID: snapshot must be under main build/") from exc
    if (len(relative.parts) < 3 or relative.parts[0] != "build"
            or "dist" in relative.parts or output.name != "build-info.json"):
        raise ReleaseBuildSourceError("BUILD_INFO_PATH_INVALID: use a fresh build/.../build-info.json")
    return output


def begin_snapshot(root: Path, packaging_root: Path, output: Path) -> dict[str, Any]:
    """Create one snapshot in a new run directory; never reuse an old build id."""
    provider = _provider(root, packaging_root)
    output = _snapshot_path(root, output)
    try:
        output.parent.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ReleaseBuildSourceError("BUILD_RUN_ALREADY_EXISTS: choose a new provenance run directory") from exc
    try:
        return provider.write_build_info(root, output)
    except provider.ProvenanceError as exc:
        raise ReleaseBuildSourceError(f"SOURCE_IDENTITY_UNPROVEN: {exc}") from exc


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ReleaseBuildSourceError("BUILD_INFO_INVALID: duplicate JSON key")
        result[name] = value
    return result


def _invalid_constant(value):
    raise ReleaseBuildSourceError("BUILD_INFO_INVALID: non-finite JSON value")


def stage_snapshot(root: Path, packaging_root: Path, snapshot: Path, app_root: Path) -> dict[str, Any]:
    """Check sources and stage the exact pre-build bytes, without signing anything."""
    provider = _provider(root, packaging_root)
    snapshot = _snapshot_path(root, snapshot)
    if not snapshot.is_file() or snapshot.stat().st_size > 65536:
        raise ReleaseBuildSourceError("BUILD_INFO_INVALID: bounded pre-build snapshot required")
    data = snapshot.read_bytes()
    try:
        expected = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError) as exc:
        raise ReleaseBuildSourceError("BUILD_INFO_INVALID: pre-build snapshot is not strict JSON") from exc
    if not isinstance(expected, dict) or type(expected.get("source_dirty")) is not bool:
        raise ReleaseBuildSourceError("BUILD_INFO_INVALID: actual source_dirty fact is required")
    app_root = app_root.resolve()
    if not app_root.is_dir():
        raise ReleaseBuildSourceError("APPLICATION_TREE_MISSING: build the application before staging metadata")
    manifest = app_root / "manifest/install-manifest.json"
    if manifest.exists() or manifest.is_symlink():
        raise ReleaseBuildSourceError("APPLICATION_ALREADY_SIGNED: existing dist cannot be relabelled")
    destination = app_root / "build-info.json"
    if destination.exists() or destination.is_symlink():
        raise ReleaseBuildSourceError("APPLICATION_METADATA_EXISTS: refusing to overwrite build identity")
    try:
        current = provider.collect_build_info(root)
    except provider.ProvenanceError as exc:
        raise ReleaseBuildSourceError(f"SOURCE_IDENTITY_UNPROVEN: {exc}") from exc
    # Compare JSON types as well as values (True must not equal schema version 1).
    if json.dumps(current, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ReleaseBuildSourceError("BUILD_SOURCES_CHANGED: main/submodule sources changed during the build")
    if snapshot.read_bytes() != data:
        raise ReleaseBuildSourceError("BUILD_INFO_CHANGED: pre-build metadata changed during verification")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=app_root, prefix=".build-info-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        # .deb installs these public source facts for ordinary application users.
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("begin", "stage"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--packaging-root", required=True, type=Path)
        command.add_argument("--snapshot", required=True, type=Path)
        if name == "stage":
            command.add_argument("--app-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "begin":
            info = begin_snapshot(args.root, args.packaging_root, args.snapshot)
        else:
            info = stage_snapshot(args.root, args.packaging_root, args.snapshot, args.app_root)
    except ReleaseBuildSourceError as exc:
        print(f"release_build_provenance: {exc}", file=sys.stderr)
        return 2
    except OSError:
        # File errors can contain user paths; do not echo source/credential paths.
        print("release_build_provenance: BUILD_SOURCE_IO_FAILED: cannot read/write build evidence", file=sys.stderr)
        return 2
    print(json.dumps({
        "command": args.command, "source_revision": info["source_revision"],
        "source_dirty": info["source_dirty"], "source_tree_sha256": info["source_tree_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
