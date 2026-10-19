"""Build source wiring tests use synthetic Git checkouts, never production artifacts.

No build entrypoint, compiler, device, service or production signing key is run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import install_manifest

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
CORE = Path(__file__).resolve().parents[2]
PROVIDER = CORE / "scripts/package_contract/build_provenance.py"
HELPER = PACKAGING / "release_build_provenance.py"
GRAPH = {"": ("android", "frontend_cybergothic", "harness_workbench", "packaging"),
         "android": ("app/src/main/cpp/llama.cpp",)}


def _git(root, *arguments):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update(GIT_AUTHOR_NAME="Build Wiring Fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
               GIT_COMMITTER_NAME="Build Wiring Fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid",
               GIT_AUTHOR_DATE="2026-10-01T00:00:00Z", GIT_COMMITTER_DATE="2026-10-01T00:00:00Z")
    result = subprocess.run(["git", "-C", str(root), *arguments], env=env, capture_output=True,
                            check=True, timeout=20)
    return result.stdout.decode("utf-8").strip()


def _put(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))


@pytest.fixture(scope="session")
def build_source_seed(tmp_path_factory):
    root = tmp_path_factory.mktemp("release-source-seed")

    def create(component):
        checkout = root / component
        checkout.mkdir(parents=True, exist_ok=True)
        _git(checkout, "init", "-q")
        _git(checkout, "config", "core.autocrlf", "false")
        _git(checkout, "config", "core.hooksPath", "NUL" if os.name == "nt" else "/dev/null")
        _put(checkout, ".gitignore", "build/\ndist/\n__pycache__/\n.signing-keys/\n.env\n*.pyc\n")
        _put(checkout, "source.txt", "fixture source\n")
        names = [".gitignore", "source.txt"]
        if component == "":
            _put(checkout, "release-contract.json", '{"schema_version":1}\n')
            _put(checkout, "src/core.py", "VALUE = 1\n")
            _put(checkout, "scripts/package_contract/build_provenance.py", PROVIDER.read_bytes())
            names += ["release-contract.json", "src/core.py", "scripts/package_contract/build_provenance.py"]
        if component == "packaging":
            _put(checkout, "packaging/release_build_provenance.py", HELPER.read_bytes())
            names += ["packaging/release_build_provenance.py"]
        modules = []
        for child in GRAPH.get(component, ()):
            revision = create(f"{component}/{child}" if component else child)
            _git(checkout, "update-index", "--add", "--cacheinfo", f"160000,{revision},{child}")
            modules.append(f'[submodule "{child}"]\n\tpath = {child}\n\turl = https://example.invalid/{child}.git\n')
        if modules:
            _put(checkout, ".gitmodules", "".join(modules))
            names.append(".gitmodules")
        _git(checkout, "add", "--", *names)
        _git(checkout, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "synthetic build wiring fixture")
        return _git(checkout, "rev-parse", "HEAD")

    create("")
    return root


@pytest.fixture
def build_sources(build_source_seed, tmp_path):
    root = tmp_path / "main checkout with spaces"
    shutil.copytree(build_source_seed, root)
    path = root / "packaging/packaging/release_build_provenance.py"
    spec = importlib.util.spec_from_file_location("_release_wiring_fixture", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return root, helper


def _begin(build_sources):
    root, helper = build_sources
    snapshot = root / "build/release-provenance/test-run/build-info.json"
    info = helper.begin_snapshot(root, root / "packaging", snapshot)
    app = root / "packaging/dist/new application"
    app.mkdir(parents=True)
    _put(app, "application.exe", b"new build fixture")
    return root, helper, snapshot, app, info


def _stage(root, helper, snapshot, app):
    return helper.stage_snapshot(root, root / "packaging", snapshot, app)


def test_prebuild_metadata_is_verbatim_and_part_of_install_baseline(build_sources, monkeypatch):
    monkeypatch.setenv("QLH_SOURCE_REVISION", "f" * 40)
    root, helper, snapshot, app, before = _begin(build_sources)
    data = snapshot.read_bytes()
    assert before["source_revision"] == _git(root, "rev-parse", "HEAD")
    assert before["source_dirty"] is False
    assert _stage(root, helper, snapshot, app) == before
    assert (app / "build-info.json").read_bytes() == data
    if os.name != "nt":
        assert (app / "build-info.json").stat().st_mode & 0o777 == 0o644
    baseline = install_manifest.build_install_manifest(
        app, app_id="qlh-edge-inference", version="0.1.8.1", platform="windows",
        variant="cpu", package_kind="application", runtime_profile="llama_cpp_only",
    )
    entry = next(item for item in baseline["files"] if item["path"] == "build-info.json")
    assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    assert entry["size"] == len(data)


def test_dirty_snapshot_remains_dirty_and_can_only_stage_unchanged(build_sources):
    root, helper = build_sources
    _put(root, "src/core.py", "VALUE = 2\n")
    root, helper, snapshot, app, before = _begin(build_sources)
    assert before["source_dirty"] is True
    assert "src/core.py" in before["components"]["."]["tracked_changes"]
    assert _stage(root, helper, snapshot, app)["source_dirty"] is True
    assert json.loads((app / "build-info.json").read_bytes())["source_dirty"] is True


@pytest.mark.parametrize("component,relative", [
    ("", "src/core.py"), ("", "src/untracked_new.py"),
    ("packaging", "source.txt"), ("android", "source.txt"),
    ("android/app/src/main/cpp/llama.cpp", "source.txt"),
])
def test_source_change_during_build_stops_before_metadata_or_signing(build_sources, component, relative):
    root, helper, snapshot, app, _ = _begin(build_sources)
    before = snapshot.read_bytes()
    _put(root / component, relative, "changed build inputs\n")
    with pytest.raises(helper.ReleaseBuildSourceError, match="BUILD_SOURCES_CHANGED"):
        _stage(root, helper, snapshot, app)
    assert snapshot.read_bytes() == before
    assert not (app / "build-info.json").exists()
    assert not (app / "manifest").exists()


def test_same_size_mtime_and_assume_unchanged_cannot_hide_build_drift(build_sources):
    root, helper, snapshot, app, _ = _begin(build_sources)
    source = root / "src/core.py"
    state = source.stat()
    _git(root, "update-index", "--assume-unchanged", "src/core.py")
    _put(root, "src/core.py", "VALUE = 2\n")
    os.utime(source, ns=(state.st_atime_ns, state.st_mtime_ns))
    with pytest.raises(helper.ReleaseBuildSourceError, match="BUILD_SOURCES_CHANGED"):
        _stage(root, helper, snapshot, app)
    assert not (app / "build-info.json").exists()


def test_missing_submodule_identity_during_build_fails_closed(build_sources):
    root, helper, snapshot, app, _ = _begin(build_sources)
    (root / "android/.git").rename(root / "android/build-hidden-git")
    with pytest.raises(helper.ReleaseBuildSourceError, match="SOURCE_IDENTITY_UNPROVEN"):
        _stage(root, helper, snapshot, app)
    assert not (app / "build-info.json").exists()


def test_same_run_directory_cannot_silently_refresh_baseline(build_sources):
    root, helper, snapshot, app, _ = _begin(build_sources)
    before = snapshot.read_bytes()
    _put(root, "src/core.py", "VALUE = 2\n")
    with pytest.raises(helper.ReleaseBuildSourceError, match="BUILD_RUN_ALREADY_EXISTS"):
        helper.begin_snapshot(root, root / "packaging", snapshot)
    assert snapshot.read_bytes() == before


@pytest.mark.parametrize("name", ["dist/new/build-info.json", "outside/build-info.json", "build/run/other.json"])
def test_generation_never_relabels_existing_dist(build_sources, name):
    root, helper = build_sources
    with pytest.raises(helper.ReleaseBuildSourceError, match="BUILD_INFO_PATH_INVALID"):
        helper.begin_snapshot(root, root / "packaging", root / name)
    assert not (root / name).exists()


def test_snapshot_cannot_be_outside_main_checkout(build_sources, tmp_path):
    root, helper = build_sources
    with pytest.raises(helper.ReleaseBuildSourceError, match="BUILD_INFO_PATH_INVALID"):
        helper.begin_snapshot(root, root / "packaging", tmp_path / "build/run/build-info.json")


@pytest.mark.parametrize("relative", ["manifest/install-manifest.json", "build-info.json"])
def test_existing_signed_or_labelled_tree_is_never_relabelled(build_sources, relative):
    root, helper, snapshot, app, _ = _begin(build_sources)
    _put(app, relative, "original identity")
    with pytest.raises(helper.ReleaseBuildSourceError, match="APPLICATION_ALREADY_SIGNED|APPLICATION_METADATA_EXISTS"):
        _stage(root, helper, snapshot, app)
    assert (app / relative).read_text() == "original identity"


def test_packaging_checkout_must_be_the_one_in_observed_source_graph(build_sources, tmp_path):
    root, helper = build_sources
    with pytest.raises(helper.ReleaseBuildSourceError, match="PACKAGING_SOURCE_MISMATCH"):
        helper.begin_snapshot(root, tmp_path, root / "build/run/build-info.json")


@pytest.mark.parametrize("data", [b"not json", b'{"source_dirty":true,"source_dirty":false}',
                                  b'{"source_dirty":false,"value":NaN}', b'{"source_revision":"legacy"}',
                                  b'{"source_dirty":0}', b"x" * 65537],
                         ids=["syntax", "duplicate", "nonfinite", "legacy", "dirty-not-bool", "too-large"])
def test_invalid_or_legacy_unproven_snapshot_cannot_stage(build_sources, data):
    root, helper, snapshot, app, _ = _begin(build_sources)
    snapshot.write_bytes(data)
    with pytest.raises(helper.ReleaseBuildSourceError, match="BUILD_INFO_INVALID"):
        _stage(root, helper, snapshot, app)
    assert not (app / "build-info.json").exists()


def test_missing_application_tree_cannot_be_created_by_metadata_helper(build_sources):
    root, helper, snapshot, app, _ = _begin(build_sources)
    missing = app / "missing"
    with pytest.raises(helper.ReleaseBuildSourceError, match="APPLICATION_TREE_MISSING"):
        _stage(root, helper, snapshot, missing)
    assert not missing.exists()


def test_cli_uses_real_checkout_and_spaces_then_rejects_mutation(build_sources):
    root, _ = build_sources
    snapshot = root / "build/cli run/build-info.json"
    app = root / "packaging/dist/cli app"
    app.mkdir(parents=True)
    command = [sys.executable, "-B", str(root / "packaging/packaging/release_build_provenance.py")]
    common = ["--root", str(root), "--packaging-root", str(root / "packaging"), "--snapshot", str(snapshot)]
    begun = subprocess.run(command + ["begin", *common], capture_output=True, text=True, timeout=45)
    assert begun.returncode == 0, begun.stderr
    assert json.loads(begun.stdout)["source_dirty"] is False
    _put(root, "src/core.py", "VALUE = 2\n")
    staged = subprocess.run(command + ["stage", *common, "--app-root", str(app)],
                            capture_output=True, text=True, timeout=45)
    assert staged.returncode == 2
    assert "BUILD_SOURCES_CHANGED" in staged.stderr
    assert not (app / "build-info.json").exists()


def test_git_free_export_has_named_failure_even_with_environment_revision(tmp_path, monkeypatch):
    exported = tmp_path / "source export"
    exported.mkdir()
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("QLH_SOURCE_REVISION", "a" * 40)
    result = subprocess.run([sys.executable, "-B", str(HELPER), "begin", "--root", str(exported),
                             "--packaging-root", str(PACKAGING.parent), "--snapshot",
                             str(exported / "build/run/build-info.json")],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert "SOURCE_CHECKOUT_REQUIRED" in result.stderr
    assert not (exported / "build").exists()


@pytest.mark.parametrize("name", ["build-cpu.bat"])
def test_windows_entrypoints_snapshot_before_build_and_stage_before_signing(name):
    raw = (PACKAGING / name).read_bytes()
    assert raw.count(b"\r\n") > 0 and b"\n" not in raw.replace(b"\r\n", b"")
    script = raw.decode("utf-8").replace("\r\n", "\n")
    begin = script.index("release_build_provenance.py begin")
    compile_step = script.index("-m PyInstaller")
    stage = script.index("release_build_provenance.py stage")
    signing = script.index("packaging\\install_manifest.py build")
    assert begin < script.index("-m pip install") < compile_step < stage < signing
    assert "\\build\\release-provenance\\" in script
    assert script.count('--snapshot "%QLH_BUILD_INFO%"') == 2
    for command in ("begin", "stage"):
        start = script.index(f"release_build_provenance.py {command}")
        assert "if errorlevel 1 exit /b 1" in script[start:script.index("\n\n", start)]
    assert '--packaging-root "%CD%"' in script
    assert 'for %%I in ("%QLH_CORE_ROOT%")' in script


def test_linux_entrypoint_snapshot_precedes_compilation_and_metadata_precedes_signing():
    script = (PACKAGING / "linux/build-deb.sh").read_text(encoding="utf-8")
    guard = script.index("SOURCE_CHECKOUT_REQUIRED")
    begin = script.index('release_build_provenance.py" begin')
    compile_step = script.index('scripts/build_llama_quantize.py"')
    stage = script.index('release_build_provenance.py" stage')
    signing = script.index('install_manifest.py" build')
    assert guard < begin < compile_step < script.index("npm ci") < stage < signing < script.index("dpkg-deb --build")
    assert 'BUILD_INFO="$PROVENANCE_DIR/source/build-info.json"' in script
    assert script.count('--snapshot "$BUILD_INFO"') == 2
    assert '--app-root "$BUILD_DIR/opt/qlh-edge-inference"' in script
    assert "set -euo pipefail" in script


def _native_bash():
    if os.name == "nt":
        git_exe = Path(shutil.which("git"))
        for candidate in (git_exe.parent.parent / "bin/bash.exe", git_exe.parent.parent / "usr/bin/bash.exe"):
            if candidate.is_file():
                return str(candidate)
    candidate = shutil.which("bash")
    if candidate:
        return candidate
    pytest.skip("Bash is unavailable for the non-building shell syntax/export check")


def test_linux_shell_syntax_without_running_the_build():
    result = subprocess.run([_native_bash(), "--noprofile", "--norc", "-n",
                             str(PACKAGING / "linux/build-deb.sh")],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_linux_git_free_export_stops_before_toolchain_and_writes_nothing(tmp_path):
    exported = tmp_path / "git free source"
    exported.mkdir()
    env = dict(os.environ)
    env.pop("BASH_ENV", None)
    env.pop("ENV", None)
    env.update(QLH_CORE_ROOT=exported.as_posix(), QLH_SOURCE_REVISION="a" * 40,
               SOURCE_REVISION="a" * 40)
    result = subprocess.run([_native_bash(), "--noprofile", "--norc",
                             str(PACKAGING / "linux/build-deb.sh"), "cpu"],
                            capture_output=True, text=True, env=env, timeout=15)
    assert result.returncode == 2
    assert "SOURCE_CHECKOUT_REQUIRED" in result.stderr
    assert "[toolchain]" not in result.stdout
    assert list(exported.iterdir()) == []


def test_json_type_coercion_cannot_disguise_metadata_change(build_sources):
    root, helper, snapshot, app, _ = _begin(build_sources)
    content = json.loads(snapshot.read_bytes())
    content["schema_version"] = True
    snapshot.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(helper.ReleaseBuildSourceError, match="BUILD_SOURCES_CHANGED"):
        _stage(root, helper, snapshot, app)
    assert not (app / "build-info.json").exists()
