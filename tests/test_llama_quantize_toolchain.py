"""Contracts for the pinned llama-quantize build and managed package."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from scripts import build_llama_quantize as builder
from scripts.build_llama_quantize import cmake_definitions
from scripts.model_tools.llama_quantize_toolchain import (
    ROOT,
    file_sha256,
    expected_source_revision,
    LlamaQuantizeToolchainError,
    host_target_id,
    load_lock,
    normalize_architecture,
    normalize_platform,
    resolve_quantizer,
    verify_managed_package,
)


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _write_package(root: Path, target_id: str, *, revision: str | None = None,
                   source_revision: str | None = None, lock=None, legacy=False) -> Path:
    lock = lock or load_lock()
    target = lock["targets"][target_id]
    package = root / "llama-quantize" / target_id
    package.mkdir(parents=True)
    executable = package / target["executable"]
    executable.write_bytes(b"pinned-quantizer-fixture")
    if target["platform"] == "linux":
        executable.chmod(0o755)
    license_file = package / "LICENSE.llama.cpp"
    license_file.write_text("fixture license\n", encoding="ascii")
    files = [
        {"path": item.name, "size_bytes": item.stat().st_size, "sha256": file_sha256(item)}
        for item in sorted((executable, license_file), key=lambda path: path.name)
    ]
    manifest = {
        "schema_version": 1,
        "tool": "llama-quantize",
        "target_id": target_id,
        "upstream": {
            "repository": lock["upstream"]["repository"],
            "revision": revision or lock["upstream"]["revision"],
        },
        "executable": target["executable"],
        "build": {},
        "smoke": {
            "help": True,
            "q4_k_m_listed": True,
            "runtime_dependencies_verified": True,
            "runtime_dependencies": [],
        },
        "files": files,
    }
    if not legacy:
        manifest.update({
            "source_revision": source_revision or expected_source_revision(lock),
            "source_dirty": False,
            "source_patches": [{"id": item.get("id") or item["path"], "patch_sha256": item["sha256"].lower(),
                                "state": "applied_committed"} for item in lock.get("patches", [])],
        })
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return package


def test_lock_matches_checked_out_llama_cpp_revision():
    lock = load_lock()
    source = ROOT / lock["source"]

    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    assert result.stdout.strip() == expected_source_revision(lock)
    observed = builder._source_contract(source, ROOT, lock)
    assert observed["source_revision"] == result.stdout.strip()
    assert all(item["state"] == "applied_committed" for item in observed["source_patches"])
    assert observed["source_dirty"] is False


def test_host_target_normalization_contract():
    assert normalize_platform("win32") == "windows"
    assert normalize_platform("Linux") == "linux"
    assert normalize_architecture("AMD64") == "x86_64"
    assert normalize_architecture("x64") == "x86_64"
    assert host_target_id(platform_name="windows", architecture="amd64") == "windows-x86_64"


def test_managed_package_verifies_revision_hash_and_file_set(tmp_path: Path):
    target_id = host_target_id()
    package = _write_package(tmp_path, target_id)

    report = verify_managed_package(package, expected_target=target_id)

    assert report["valid"] is True
    assert report["status"] == "verified"
    assert report["sha256"] == file_sha256(package / str(report["executable"]))
    assert str(tmp_path) not in str(report)


def test_managed_package_rejects_tampering_and_revision_drift(tmp_path: Path):
    target_id = host_target_id()
    tampered = _write_package(tmp_path / "tampered", target_id)
    (tampered / load_lock()["targets"][target_id]["executable"]).write_bytes(b"changed")
    drifted = _write_package(tmp_path / "drifted", target_id, revision="0" * 40)

    tampered_report = verify_managed_package(tampered, expected_target=target_id)
    drifted_report = verify_managed_package(drifted, expected_target=target_id)

    assert tampered_report["valid"] is False
    assert "file_digest_mismatch" in {item["code"] for item in tampered_report["errors"]}
    assert drifted_report["valid"] is False
    assert "revision_mismatch" in {item["code"] for item in drifted_report["errors"]}


def test_resolver_prefers_verified_managed_package(tmp_path: Path, monkeypatch):
    target_id = host_target_id()
    package = _write_package(tmp_path, target_id)
    monkeypatch.setenv("QLH_MODEL_TOOLS_ROOT", str(tmp_path))
    monkeypatch.delenv("QLH_LLAMA_QUANTIZE", raising=False)

    executable, report = resolve_quantizer(project_root=tmp_path / "project")

    assert executable == package / load_lock()["targets"][target_id]["executable"]
    assert report["status"] == "available"
    assert report["provenance"] == "managed_package"
    assert report["verification"] == "verified"
    assert report["revision"] == expected_source_revision(load_lock())
    assert report["upstream_revision"] == load_lock()["upstream"]["revision"]
    assert report["source_dirty"] is False
    assert str(tmp_path) not in str(report)


def test_resolver_fails_closed_for_corrupt_managed_package(tmp_path: Path, monkeypatch):
    target_id = host_target_id()
    package = _write_package(tmp_path, target_id)
    (package / "unexpected.dll").write_bytes(b"not-listed")
    monkeypatch.setenv("QLH_MODEL_TOOLS_ROOT", str(tmp_path))

    executable, report = resolve_quantizer(project_root=tmp_path / "project")

    assert executable is None
    assert report["status"] == "invalid"
    assert report["verification"] == "failed"


def test_cmake_contract_is_offline_portable_and_platform_specific():
    common = cmake_definitions("linux-x86_64", "gnu")
    windows_gnu = cmake_definitions("windows-x86_64", "gnu")
    windows_msvc = cmake_definitions("windows-x86_64", "msvc")

    assert "-DLLAMA_USE_PREBUILT_UI=OFF" in common
    assert "-DLLAMA_OPENSSL=OFF" in common
    assert "-DGGML_CUDA=OFF" in common
    assert "-DGGML_NATIVE=OFF" in common
    assert "-DGGML_OPENMP=OFF" in common
    assert not any("CMAKE_EXE_LINKER_FLAGS" in item for item in common)
    assert any("-static-libstdc++" in item for item in windows_gnu)
    assert "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded" in windows_msvc


def test_release_packaging_requires_managed_quantizer():
    for relative in ("packaging/qlh-cpu.spec", "packaging/qlh-cuda.spec"):
        content = (RELEASE_ROOT / relative).read_text(encoding="utf-8")
        assert "verify_managed_package" in content
        assert "model-tools/llama-quantize/windows-x86_64" in content
    linux = (RELEASE_ROOT / "packaging/linux/build-deb.sh").read_text(encoding="utf-8")
    assert "scripts/build_llama_quantize.py" in linux
    assert "model-tools/llama-quantize/linux-x86_64" in linux


@pytest.mark.parametrize("local", [None, "", "784cbbb", "G" * 40, "A" * 40, 123, False])
def test_lock_rejects_invalid_explicit_local_revision(tmp_path, local):
    lock = copy.deepcopy(load_lock())
    lock["revision_local"] = local
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(LlamaQuantizeToolchainError, match="source revision"):
        load_lock(path)


def test_source_pin_falls_back_to_baseline_only_when_local_field_is_absent():
    lock = copy.deepcopy(load_lock())
    assert expected_source_revision(lock) == lock["revision_local"]
    del lock["revision_local"]
    assert expected_source_revision(lock) == lock["upstream"]["revision"]


def test_managed_package_distinguishes_upstream_from_actual_source(tmp_path):
    lock = load_lock()
    target_id = host_target_id()
    package = _write_package(tmp_path, target_id)
    manifest = json.loads((package / "manifest.json").read_text())
    report = verify_managed_package(package, expected_target=target_id)
    assert report["valid"] is True
    assert manifest["upstream"]["revision"] == lock["upstream"]["revision"]
    assert manifest["source_revision"] == lock["revision_local"]
    assert report["revision"] == lock["revision_local"]
    assert report["upstream_revision"] == lock["upstream"]["revision"]
    assert report["source_identity"] == "explicit_source"


@pytest.mark.parametrize("source_revision", ["0" * 40, "upstream"])
def test_managed_package_rejects_unpinned_or_upstream_only_source(tmp_path, source_revision):
    lock = load_lock()
    if source_revision == "upstream":
        source_revision = lock["upstream"]["revision"]
    target_id = host_target_id()
    package = _write_package(tmp_path, target_id, source_revision=source_revision)
    report = verify_managed_package(package, expected_target=target_id)
    assert report["valid"] is False
    assert "source_revision_mismatch" in {item["code"] for item in report["errors"]}
    assert report["revision"] == source_revision  # Observed claim, never relabelled to the expected HEAD.


def test_patched_source_cannot_be_mislabelled_as_upstream(tmp_path):
    lock = load_lock()
    target_id = host_target_id()
    package = _write_package(tmp_path, target_id, revision=lock["revision_local"])
    report = verify_managed_package(package, expected_target=target_id)
    assert report["valid"] is False
    assert "revision_mismatch" in {item["code"] for item in report["errors"]}


def test_legacy_package_cannot_prove_committed_patch_source_and_is_not_migrated(tmp_path):
    target_id = host_target_id()
    package = _write_package(tmp_path, target_id, legacy=True)
    before = (package / "manifest.json").read_bytes()
    report = verify_managed_package(package, expected_target=target_id)
    assert report["valid"] is False
    assert report["revision"] is None
    assert "source_revision_missing" in {item["code"] for item in report["errors"]}
    assert (package / "manifest.json").read_bytes() == before


def test_legacy_upstream_only_metadata_is_explicit_when_lock_has_no_local_pin(tmp_path):
    lock = copy.deepcopy(load_lock())
    del lock["revision_local"]
    target_id = host_target_id()
    package = _write_package(tmp_path, target_id, lock=lock, legacy=True)
    report = verify_managed_package(package, expected_target=target_id, lock=lock)
    assert report["valid"] is True
    assert report["revision"] == lock["upstream"]["revision"]
    assert report["source_identity"] == "legacy_upstream_only"
    assert report["source_dirty"] is None  # Old metadata did not observe this fact.


@pytest.mark.parametrize("change", ["dirty", "patch_digest", "patch_missing", "patch_duplicate", "patch_pending"])
def test_source_patch_receipts_are_verified(tmp_path, change):
    target_id = host_target_id()
    package = _write_package(tmp_path, target_id)
    path = package / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "dirty":
        manifest["source_dirty"] = True
    elif change == "patch_digest":
        manifest["source_patches"][0]["patch_sha256"] = "0" * 64
    elif change == "patch_missing":
        manifest.pop("source_patches")
    elif change == "patch_duplicate":
        manifest["source_patches"] *= 2
    else:
        manifest["source_patches"][0]["state"] = "ready_to_apply"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = verify_managed_package(package, expected_target=target_id)
    assert report["valid"] is False
    assert {item["code"] for item in report["errors"]} & {"source_contract_invalid", "source_patches_invalid"}


@pytest.fixture
def simulated_builder(tmp_path, monkeypatch):
    """Exercise the real packager/validator; CMake/compiler commands are simulated."""
    lock = copy.deepcopy(load_lock())
    project = tmp_path / "core"
    source = project / "android/app/src/main/cpp/llama.cpp"
    source.mkdir(parents=True)
    (source / "LICENSE").write_text("fixture MIT license", encoding="ascii")
    build_dir = project / "build/native"
    output = project / "build/model-tools/llama-quantize"
    calls = []
    target_id = host_target_id()
    state = {"revision": expected_source_revision(lock), "patch_state": "applied_committed"}
    monkeypatch.setattr(builder, "load_lock", lambda: lock)
    monkeypatch.setattr(builder, "llama_cpp_source", lambda _: source)
    monkeypatch.setattr(builder, "_source_revision", lambda *_: state["revision"])

    for item in lock["patches"]:
        patch = project / item["path"]
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_bytes((ROOT / item["path"]).read_bytes())
    monkeypatch.setattr(builder, "source_lock_contract", lambda: (expected_source_revision(lock), lock["patches"]))
    monkeypatch.setattr(builder, "find_cmake", lambda *_: (project / "cmake", project / "ninja"))
    monkeypatch.setattr(builder, "_compiler_pair", lambda *_: (project / "cc", project / "c++", "gnu"))
    monkeypatch.setattr(builder, "_runtime_dependencies", lambda *_: [])

    def run(command, **kwargs):
        calls.append(command)
        if "status" in command:
            changed = " M include/llama.h\n" if state["patch_state"] == "applied" else ""
            return subprocess.CompletedProcess(command, 0, changed)
        if "-S" in command:
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / "CMakeCache.txt").write_text(
                f"CMAKE_GENERATOR:INTERNAL={state.get('generator', 'Ninja')}\n"
                f"CMAKE_BUILD_TYPE:STRING={state.get('build_type', 'Release')}\n"
                f"CMAKE_CONFIGURATION_TYPES:STRING={state.get('configurations', '')}\n"
                f"CMAKE_C_COMPILER:FILEPATH={(project / 'cc').as_posix()}\n"
                f"CMAKE_CXX_COMPILER:FILEPATH={state.get('actual_cxx', (project / 'c++').as_posix())}\n",
                encoding="utf-8",
            )
            info_dir = build_dir / "CMakeFiles/3.22.1"
            info_dir.mkdir(parents=True, exist_ok=True)
            for language, requested in (("C", project / "cc"), ("CXX", project / "c++")):
                compiler = state.get("actual_cxx", requested.as_posix()) if language == "CXX" else requested.as_posix()
                (info_dir / f"CMake{language}Compiler.cmake").write_text(
                    f'set(CMAKE_{language}_COMPILER "{compiler}")\n'
                    f'set(CMAKE_{language}_COMPILER_ID "{state.get("compiler_id", "GNU")}")\n'
                    f'set(CMAKE_{language}_COMPILER_VERSION "15.2.0")\n',
                    encoding="utf-8",
                )
        if "--build" in command:
            executable = build_dir / "bin" / state.get("binary_configuration", "") / lock["targets"][target_id]["executable"]
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_bytes(b"simulated compiler output, never executed")
            if state.get("after_build_revision"):
                state["revision"] = state["after_build_revision"]
        text = "allowed quantization types: Q4_K_M\n" if "--help" in command else "cmake version 3.22.1\n"
        return subprocess.CompletedProcess(command, 0, text)

    monkeypatch.setattr(builder, "_run", run)
    return lock, state, calls, dict(project_root=project, output_root=output, build_dir=build_dir)


def test_builder_records_actual_patch_commit_and_true_upstream(simulated_builder):
    lock, _, calls, options = simulated_builder
    report = builder.build_and_package(**options)
    package = options["output_root"] / "packages" / host_target_id()
    manifest = json.loads((package / "manifest.json").read_text())
    assert report["valid"] is True
    assert report["revision"] == lock["revision_local"]
    assert report["upstream_revision"] == lock["upstream"]["revision"]
    assert manifest["upstream"]["revision"] == lock["upstream"]["revision"]
    assert manifest["source_revision"] == lock["revision_local"]
    assert manifest["source_dirty"] is False
    assert any("--build" in command for command in calls)  # Simulation only.
    assert verify_managed_package(package, expected_target=host_target_id())["valid"] is True
    assert report["archive"].startswith(f'llama-quantize-{lock["revision_local"][:8]}-')


@pytest.mark.parametrize("revision", ["0" * 40, "upstream"])
def test_builder_rejects_any_head_except_local_pin_before_compilation(simulated_builder, revision):
    lock, state, calls, options = simulated_builder
    state["revision"] = lock["upstream"]["revision"] if revision == "upstream" else revision
    with pytest.raises(builder.BuildError, match="source HEAD"):
        builder.build_and_package(**options)
    assert calls == []
    assert not options["output_root"].exists()


def test_builder_stops_when_head_changes_during_compilation(simulated_builder):
    lock, state, _, options = simulated_builder
    state["after_build_revision"] = lock["upstream"]["revision"]
    with pytest.raises(builder.BuildError, match="source HEAD"):
        builder.build_and_package(**options)
    assert not (options["output_root"] / "packages").exists()


def test_builder_rejects_worktree_patch_on_committed_source(simulated_builder):
    _, state, calls, options = simulated_builder
    state["patch_state"] = "applied"
    with pytest.raises(builder.BuildError, match="worktree patches"):
        builder.build_and_package(**options)
    assert all(command[0] == "git" for command in calls)


def test_builder_supports_worktree_patch_contract_without_local_pin(simulated_builder):
    lock, state, _, options = simulated_builder
    del lock["revision_local"]
    state["revision"] = lock["upstream"]["revision"]
    state["patch_state"] = "applied"
    report = builder.build_and_package(**options)
    package = options["output_root"] / "packages" / host_target_id()
    manifest = json.loads((package / "manifest.json").read_text())
    assert report["revision"] == lock["upstream"]["revision"]
    assert manifest["source_dirty"] is True
    assert manifest["source_patches"][0]["state"] == "applied"
    assert verify_managed_package(package, expected_target=host_target_id(), lock=lock)["valid"] is True


def test_source_patch_contract_rejection_stops_before_compilation(simulated_builder, monkeypatch):
    _, _, calls, options = simulated_builder

    def refused():
        raise builder.SyncError("fixture digest mismatch")

    monkeypatch.setattr(builder, "source_lock_contract", refused)
    with pytest.raises(builder.BuildError, match="source patch contract failed"):
        builder.build_and_package(**options)
    assert calls == []


def test_find_cmake_finds_sdk_ninja_independently_of_path_cmake(tmp_path, monkeypatch):
    cmake_name = "cmake.exe" if builder.os.name == "nt" else "cmake"
    ninja_name = "ninja.exe" if builder.os.name == "nt" else "ninja"
    path_cmake = tmp_path / "path-tools" / cmake_name
    path_cmake.parent.mkdir()
    path_cmake.touch()
    sdk_root = tmp_path / "android-sdk"
    sdk_ninja = sdk_root / "cmake/3.22.1/bin" / ninja_name
    sdk_ninja.parent.mkdir(parents=True)
    sdk_ninja.touch()
    monkeypatch.setattr(builder.shutil, "which", lambda name: str(path_cmake) if name == "cmake" else None)
    monkeypatch.setattr(builder, "_android_sdk_roots", lambda _: [sdk_root])
    assert builder.find_cmake(tmp_path) == (path_cmake.absolute(), sdk_ninja.absolute())


def test_windows_gnu_without_ninja_fails_before_cmake(simulated_builder, monkeypatch):
    _, _, calls, options = simulated_builder
    monkeypatch.setattr(builder, "host_target_id", lambda: "windows-x86_64")
    monkeypatch.setattr(builder, "find_cmake", lambda *_: (options["project_root"] / "cmake", None))
    with pytest.raises(builder.BuildError, match="Ninja is required for Windows GNU"):
        builder.build_and_package(**options)
    assert not any("-S" in command or "--build" in command for command in calls)
    assert not options["output_root"].exists()


def test_build_selects_release_and_records_actual_generator(simulated_builder):
    _, state, calls, options = simulated_builder
    report = builder.build_and_package(**options)
    build = next(command for command in calls if "--build" in command)
    assert build[build.index("--config") + 1] == "Release"
    configure = next(command for command in calls if "-S" in command)
    assert configure[configure.index("-G") + 1] == "Ninja"
    package = options["output_root"] / "packages" / host_target_id()
    manifest = json.loads((package / "manifest.json").read_text())
    assert manifest["build"]["generator"] == "Ninja"
    assert manifest["build"]["configuration"] == "Release"
    assert report["valid"] is True


def test_debug_only_binary_cannot_be_packaged_as_release(simulated_builder):
    _, state, _, options = simulated_builder
    state["binary_configuration"] = "Debug"
    with pytest.raises(builder.BuildError, match="did not produce the locked executable"):
        builder.build_and_package(**options)
    assert not (options["output_root"] / "packages").exists()


@pytest.mark.parametrize("state_change", [
    {"generator": "Visual Studio 18 2026"},
    {"actual_cxx": "C:/other-toolchain/cl.exe"},
])
def test_actual_cmake_toolchain_mismatch_stops_before_compilation(simulated_builder, state_change):
    _, state, calls, options = simulated_builder
    state.update(state_change)
    with pytest.raises(builder.BuildError, match="CMake's actual"):
        builder.build_and_package(**options)
    assert not any("--build" in command for command in calls)
    assert not (options["output_root"] / "packages").exists()


def test_msvc_multi_config_selects_release_without_calling_cl_version(simulated_builder, monkeypatch):
    _, state, calls, options = simulated_builder
    state.update(generator="Visual Studio 18 2026", compiler_id="MSVC", binary_configuration="Release",
                 configurations="Debug;Release;MinSizeRel;RelWithDebInfo")
    monkeypatch.setattr(builder, "find_cmake", lambda *_: (options["project_root"] / "cmake", None))
    monkeypatch.setattr(builder, "_compiler_pair", lambda *_: (None, None, "msvc"))
    report = builder.build_and_package(**options)
    package = options["output_root"] / "packages" / host_target_id()
    manifest = json.loads((package / "manifest.json").read_text())
    assert manifest["build"]["generator"] == "Visual Studio 18 2026"
    assert manifest["build"]["compiler"] == "MSVC 15.2.0"
    build = next(command for command in calls if "--build" in command)
    assert build[build.index("--config") + 1] == "Release"
    assert report["valid"] is True
    assert not any(command[0] == str(options["project_root"] / "c++") for command in calls)


def test_compiler_cache_claim_cannot_hide_generated_actual_compiler(simulated_builder, monkeypatch):
    _, state, calls, options = simulated_builder
    original_run = builder._run

    def cache_only_change(command, **kwargs):
        result = original_run(command, **kwargs)
        if "-S" in command:
            generated = options["build_dir"] / "CMakeFiles/3.22.1/CMakeCXXCompiler.cmake"
            generated.write_text(
                'set(CMAKE_CXX_COMPILER "C:/ignored-request/cl.exe")\n'
                'set(CMAKE_CXX_COMPILER_ID "MSVC")\n'
                'set(CMAKE_CXX_COMPILER_VERSION "19.51.0")\n', encoding="utf-8")
        return result

    monkeypatch.setattr(builder, "_run", cache_only_change)
    with pytest.raises(builder.BuildError, match="CMake's actual CMAKE_CXX_COMPILER"):
        builder.build_and_package(**options)
    assert not any("--build" in command for command in calls)


@pytest.mark.parametrize("multi_configuration", [False, True])
def test_build_never_selects_old_binary_from_another_configuration(simulated_builder, monkeypatch, multi_configuration):
    lock, state, _, options = simulated_builder
    executable_name = lock["targets"][host_target_id()]["executable"]
    if multi_configuration:
        state.update(generator="Visual Studio 18 2026", compiler_id="MSVC", binary_configuration="Release",
                     configurations="Debug;Release;MinSizeRel;RelWithDebInfo")
        monkeypatch.setattr(builder, "find_cmake", lambda *_: (options["project_root"] / "cmake", None))
        monkeypatch.setattr(builder, "_compiler_pair", lambda *_: (None, None, "msvc"))
        stale_binary = options["build_dir"] / "bin" / executable_name
    else:
        stale_binary = options["build_dir"] / "bin/Release" / executable_name
    stale_binary.parent.mkdir(parents=True)
    stale_binary.write_bytes(b"stale successful binary from a different build")
    report = builder.build_and_package(**options)
    packaged = options["output_root"] / "packages" / host_target_id() / executable_name
    assert report["valid"] is True
    assert packaged.read_bytes() == b"simulated compiler output, never executed"
    assert report["executable_sha256"] != file_sha256(stale_binary)


@pytest.mark.parametrize("configuration", [{"build_type": "Debug"}, {"configurations": "Debug;MinSizeRel"}])
def test_actual_cmake_configuration_must_support_release(simulated_builder, configuration):
    _, state, calls, options = simulated_builder
    state.update(configuration)
    with pytest.raises(builder.BuildError, match="actual configuration"):
        builder.build_and_package(**options)
    assert not any("--build" in command for command in calls)
