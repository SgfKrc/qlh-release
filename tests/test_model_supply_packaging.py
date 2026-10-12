"""MODEL-SUPPLY-01: release resources and fresh-process offline cutter contract."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from model_supply_resources import collect_model_supply_resources, source_model_supply_resources
from model_supply_runtime import CUT_LAYERS_FLAG

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
CORE = Path(os.environ["QLH_CORE_ROOT"])
LAUNCHER = PACKAGING / "launcher.py"
FORBIDDEN = (
    "api_server", "config", "model_host", "model_module", "torch", "transformers",
    "llama_cpp", "tui_textual",
)


def test_release_resources_include_executable_scripts_and_gguf_sources():
    datas, hiddenimports = collect_model_supply_resources(CORE)
    assert (str(CORE / "scripts" / "cut_layers.py"), "scripts") in datas
    assert (str(CORE / "src" / "llama_keep_head_worker.py"), ".") in datas
    for name in ("__init__.py", "gguf_reader.py", "gguf_writer.py", "constants.py"):
        assert any(Path(source).name == name and dest == "gguf" for source, dest in datas)
    assert all("__pycache__" not in Path(source).parts for source, _ in datas)
    assert {"gguf", "numpy", "yaml", "tqdm", "requests", "relay_contract",
            "llama_keep_head"} <= set(hiddenimports)
    assert not set(FORBIDDEN).intersection(hiddenimports)


@pytest.mark.parametrize("missing", ["scripts/cut_layers.py", "src/llama_keep_head_worker.py"])
def test_missing_required_script_fails_build(tmp_path, missing):
    for relative in ("scripts/cut_layers.py", "src/llama_keep_head_worker.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative != missing:
            path.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="model_supply_resource_missing"):
        collect_model_supply_resources(tmp_path)


def test_missing_gguf_fails_build(monkeypatch):
    import model_supply_resources

    monkeypatch.setattr(model_supply_resources.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(ModuleNotFoundError, match="model_supply_dependency_missing"):
        collect_model_supply_resources(CORE)


@pytest.mark.parametrize("name", ["qlh-edge.spec", "qlh-cpu.spec", "qlh-cuda.spec"])
def test_frozen_specs_collect_resources_and_preserve_existing_console_behavior(name):
    tree = ast.parse((PACKAGING / name).read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(isinstance(call.func, ast.Name)
               and call.func.id == "collect_model_supply_resources" for call in calls)
    analysis = next(call for call in calls if isinstance(call.func, ast.Name)
                    and call.func.id == "Analysis")
    keywords = {item.arg: item.value for item in analysis.keywords}
    hidden_names = {node.id for node in ast.walk(keywords["hiddenimports"])
                    if isinstance(node, ast.Name)}
    assert "_SUPPLY_IMPORTS" in hidden_names
    if name == "qlh-edge.spec":
        datas_node = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "_datas"
                                  for target in node.targets))
    else:
        datas_node = keywords["datas"]
    assert "_SUPPLY_DATAS" in {node.id for node in ast.walk(datas_node)
                              if isinstance(node, ast.Name)}
    exe = next(call for call in calls if isinstance(call.func, ast.Name) and call.func.id == "EXE")
    console = ast.literal_eval(next(item.value for item in exe.keywords if item.arg == "console"))
    assert console is (name == "qlh-edge.spec")
    if name == "qlh-edge.spec":
        assert "torch" in ast.literal_eval(keywords["excludes"])


def test_build_requirements_install_gguf_for_edge_and_full():
    for name in ("requirements-cpu.txt", "requirements-runtime-llama_cpp_only.txt",
                 "requirements-runtime-cpu.txt", "requirements-runtime-cuda.txt"):
        assert "gguf==0.19.0" in (PACKAGING / name).read_text(encoding="utf-8").splitlines()


def test_source_backed_packages_ship_cutter_without_copying_scripts_tree():
    slim = (PACKAGING / "qlh-slim.spec").read_text(encoding="utf-8")
    assert "collect_model_supply_resources(_ROOT)" in slim
    assert "] + _RUNTIME_REQS + _SUPPLY_DATAS" in slim
    deb = (PACKAGING / "linux" / "build-deb.sh").read_text(encoding="utf-8")
    assert 'python3 "$PACKAGING_DIR/model_supply_resources.py"' in deb
    assert '--stage-source-root "$BUILD_DIR/opt/qlh-edge-inference"' in deb
    assert 'cp -r "$CORE_ROOT/scripts"' not in deb
    assert 'cp -a "$CORE_ROOT/scripts"' not in deb


@pytest.mark.parametrize("profile, engine", [
    ("llama_cpp_only", "cpu"), ("torch_cpu", "cpu"), ("torch_cuda", "cuda"),
])
def test_existing_external_runtime_is_probed_for_cutter(profile, engine, tmp_path):
    import runtime_guard

    context = runtime_guard.RuntimeContext(root=tmp_path, engine=engine, profile=profile)
    assert "gguf" in runtime_guard.required_modules(context)
    commands = runtime_guard.build_pip_commands(tmp_path / "python", tmp_path / "missing.txt", context)
    assert any("gguf==0.19.0" in command for command in commands)


def test_source_application_and_slim_bootstrap_are_different_entrypoints():
    # Artifact code in the external venv executes the shipped script with that
    # venv's Python; --qlh-cut-layers is for frozen main-engine launchers only.
    slim = (PACKAGING / "qlh-slim.spec").read_text(encoding="utf-8")
    assert '"qlh_launcher.py"' in slim
    for name in ("qlh-edge.spec", "qlh-cpu.spec", "qlh-cuda.spec"):
        assert "launcher.py" in (PACKAGING / name).read_text(encoding="utf-8")


@pytest.fixture
def frozen_resources(tmp_path):
    root = tmp_path / "arbitrary install" / "_internal"
    datas, _ = collect_model_supply_resources(CORE)
    for source, destination in datas:
        target = root / destination / Path(source).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return root


@pytest.fixture
def source_gguf(tmp_path):
    import gguf
    import numpy as np

    path = tmp_path / "source model.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen2")
    writer.add_block_count(4)
    writer.add_embedding_length(8)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(["a", "b", "c", "d"])
    for layer in range(4):
        writer.add_tensor(f"blk.{layer}.attn_norm.weight", np.full(8, layer + 1, dtype=np.float32))
    writer.add_tensor("token_embd.weight", np.ones((4, 8), dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    writer.add_tensor("output.weight", np.ones((4, 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _run(tmp_path, *args, resource_root=None):
    # Run the real launcher in a new interpreter, with an import tripwire before
    # any launcher code.  Frozen-layout runs cannot create launcher log files.
    runner = """
import importlib.abc
import runpy
import sys
from pathlib import Path
packaging, core, launcher, resources, *arguments = sys.argv[1:]
sys.path[:0] = [packaging, str(Path(core) / 'src')]
forbidden = %r
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise AssertionError('cutter imported backend: ' + fullname)
sys.meta_path.insert(0, NoBackend())
if resources:
    sys.frozen = True
    sys._MEIPASS = resources
    sys.executable = str(Path(resources).parent / 'application.exe')
sys.argv = [launcher, '--qlh-cut-layers', *arguments]
try:
    runpy.run_path(launcher, run_name='__main__')
finally:
    assert not set(forbidden).intersection(sys.modules)
    if resources:
        assert not (Path(resources).parent / 'logs').exists()
""" % (FORBIDDEN,)
    return subprocess.run(
        [sys.executable, "-I", "-c", runner, str(PACKAGING), str(CORE), str(LAUNCHER),
         str(resource_root) if resource_root else "", *map(str, args)],
        cwd=tmp_path, env={**os.environ, "QLH_CORE_ROOT": str(CORE)},
        text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=30,
    )


@pytest.mark.parametrize("frozen", [False, True])
def test_help_dispatches_before_backend_and_log_setup(tmp_path, frozen_resources, frozen):
    result = _run(tmp_path, "--help", resource_root=frozen_resources if frozen else None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--keep-head" in result.stdout
    assert "--verify-manifest" in result.stdout
    assert CUT_LAYERS_FLAG == "--qlh-cut-layers"


@pytest.mark.parametrize("cut_args, expected_range, mode", [
    (["--keep-head", "2"], [0, 2], "head"),
    (["--k", "1", "--end", "3"], [1, 3], "middle"),
    (["--k", "2"], [2, 4], "tail"),
])
def test_frozen_layout_cuts_and_verifies_canonical_manifest(
    tmp_path, frozen_resources, source_gguf, cut_args, expected_range, mode,
):
    artifact = tmp_path / "segment.gguf"
    manifest = tmp_path / "segment.manifest.json"
    result = _run(
        tmp_path, "--src", source_gguf, "--dst", artifact, "--manifest", manifest,
        "--source-model-id", "tiny-source", "--tokenizer-sha256", "a" * 64,
        *cut_args, resource_root=frozen_resources,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    content = json.loads(manifest.read_text(encoding="utf-8"))
    assert content["source_layer_range"] == expected_range
    assert content["mode"] == mode
    assert content["source_model_id"] == "tiny-source"
    assert content["tokenizer_sha256"] == "a" * 64
    assert content["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert content["source_model_sha256"] == hashlib.sha256(source_gguf.read_bytes()).hexdigest()
    if mode == "tail":
        assert content["contract"]["trim_plan_valid"] is True
    verified = _run(tmp_path, "--src", artifact, "--verify-manifest", manifest,
                    resource_root=frozen_resources)
    assert verified.returncode == 0, verified.stdout + verified.stderr
    content["artifact_sha256"] = "0" * 64
    manifest.write_text(json.dumps(content), encoding="utf-8")
    rejected = _run(tmp_path, "--src", artifact, "--verify-manifest", manifest,
                    resource_root=frozen_resources)
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr


@pytest.mark.parametrize("extra", [["--unknown"], []])
def test_invalid_cli_preserves_exit_two(tmp_path, frozen_resources, extra):
    result = _run(tmp_path, *extra, resource_root=frozen_resources)
    assert result.returncode == 2


@pytest.mark.parametrize("suffix", [".safetensors", ".bin", ""])
def test_non_gguf_source_is_named_rejection(tmp_path, frozen_resources, suffix):
    result = _run(tmp_path, "--src", str(tmp_path / ("source" + suffix)), "--k", "2",
                  resource_root=frozen_resources)
    assert result.returncode == 2
    assert "model_supply_unsupported_source_format" in result.stderr


@pytest.mark.parametrize("missing, reason", [
    ("scripts/cut_layers.py", "model_supply_resource_missing"),
    ("gguf/__init__.py", "model_supply_dependency_missing"),
])
def test_missing_frozen_resource_is_named_rejection(tmp_path, frozen_resources, missing, reason):
    (frozen_resources / missing).unlink()
    result = _run(tmp_path, "--help", resource_root=frozen_resources)
    assert result.returncode == 2
    assert reason in result.stderr


def test_dry_run_is_read_only_and_still_uses_no_backend(tmp_path, frozen_resources, source_gguf):
    destination = tmp_path / "dry.gguf"
    manifest = tmp_path / "dry.manifest.json"
    result = _run(tmp_path, "--src", source_gguf, "--dst", destination,
                  "--manifest", manifest, "--k", "2", "--dry-run",
                  resource_root=frozen_resources)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not destination.exists()
    assert not manifest.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows inherited pipe handles")
def test_windowed_missing_streams_reconnect_in_fresh_process(tmp_path):
    runner = """
import sys
sys.path.insert(0, sys.argv[1])
import model_supply_runtime
sys.frozen = True
sys.stdout = sys.stderr = None
assert model_supply_runtime._restore_helper_stdio()
print('restored stdout: \u2713', flush=True)
print('restored stderr: \u2713', file=sys.stderr, flush=True)
# Closing the duplicate must leave GetStdHandle's original handle usable.
sys.stdout.close()
sys.stdout = None
assert model_supply_runtime._restore_helper_stdio()
print('reopened duplicate', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", runner, str(PACKAGING)], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "restored stdout: \u2713" in result.stdout
    assert "reopened duplicate" in result.stdout
    assert "restored stderr: \u2713" in result.stderr


def test_console_streams_are_preserved_without_reopening(monkeypatch):
    import model_supply_runtime

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    stdout, stderr = sys.stdout, sys.stderr
    def unexpected(_):
        raise AssertionError("existing console stream was reopened")
    monkeypatch.setattr(model_supply_runtime, "_open_inherited_output", unexpected)
    assert model_supply_runtime._restore_helper_stdio()
    assert sys.stdout is stdout
    assert sys.stderr is stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows inherited pipe handles")
@pytest.mark.parametrize("missing", ["stdout", "stderr"])
def test_handle_recovery_failure_is_named_and_cutter_does_not_run(monkeypatch, missing):
    import io
    import model_supply_runtime

    output = io.StringIO()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, missing, None)
    monkeypatch.setattr(sys, "stderr" if missing == "stdout" else "stdout", output)
    def unavailable(_):
        raise OSError("inherited pipe unavailable")
    def unexpected(*args, **kwargs):
        raise AssertionError("cutter ran after failed handle restoration")
    monkeypatch.setattr(model_supply_runtime, "_open_inherited_output", unavailable)
    monkeypatch.setattr(model_supply_runtime.runpy, "run_path", unexpected)
    assert model_supply_runtime.run_cut_layers(["--help"]) == 2
    assert "model_supply_stdio_unavailable" in output.getvalue()
    assert missing in output.getvalue()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows inherited pipe handles")
def test_both_missing_outputs_report_named_failure_without_console(monkeypatch):
    import ctypes
    import model_supply_runtime

    messages = []
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    for name in ("stdout", "stderr", "__stdout__", "__stderr__"):
        monkeypatch.setattr(sys, name, None)
    def unavailable(_):
        raise OSError("inherited pipe unavailable")
    monkeypatch.setattr(model_supply_runtime, "_open_inherited_output", unavailable)
    monkeypatch.setattr(ctypes.windll.kernel32, "OutputDebugStringW", messages.append)
    assert model_supply_runtime.run_cut_layers(["--help"]) == 2
    assert len(messages) == 1
    assert messages[0].startswith("model_supply_stdio_unavailable:")
    assert "stdout:" in messages[0] and "stderr:" in messages[0]


def test_source_supply_whitelist_preserves_canonical_modules_and_converter():
    resources = source_model_supply_resources(CORE)
    by_target = {str(Path(destination) / source.name).replace("\\", "/"): source
                 for source, destination in resources}
    assert by_target["scripts/model_tools/gguf.py"] == CORE / "scripts/model_tools/gguf.py"
    assert by_target["scripts/model_tools/_canonical/gguf_convert.py"] == CORE / "scripts/model_tools/gguf_convert.py"
    assert by_target["scripts/model_tools/llama_quantize.lock.json"] == CORE / "scripts/model_tools/llama_quantize.lock.json"
    assert by_target["scripts/model_tools/__init__.py"] != CORE / "scripts/model_tools/__init__.py"
    assert by_target["scripts/__init__.py"] == PACKAGING / "model_supply_package/scripts/__init__.py"
    prefix = "android/app/src/main/cpp/llama.cpp/"
    for resource in ("convert_hf_to_gguf.py", "conversion/base.py", "conversion/qwen.py",
                     "gguf-py/gguf/__init__.py", "requirements/requirements-convert_hf_to_gguf.txt",
                     "requirements/requirements-convert_legacy_llama.txt"):
        assert by_target[prefix + resource] == CORE / (prefix + resource)
    assert not any("pipeline_adapter" in target or "sidecar_probe" in target for target in by_target)


@pytest.mark.parametrize("missing", [
    "scripts/model_tools/gguf.py", "scripts/model_tools/llama_quantize.lock.json",
    "android/app/src/main/cpp/llama.cpp/conversion/base.py",
    "android/app/src/main/cpp/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt",
])
def test_missing_supply_conversion_resources_fail_build(tmp_path, missing):
    # Copy only the canonical whitelist, preserving original paths. Adapters are
    # build-owned and checked separately by the collector.
    for source, _ in source_model_supply_resources(CORE):
        try:
            relative = source.relative_to(CORE)
        except ValueError:
            continue
        if str(relative).replace("\\", "/") == missing:
            continue
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    with pytest.raises(FileNotFoundError, match="model_supply_resource_missing"):
        collect_model_supply_resources(tmp_path)


def _run_supply(tmp_path, resources, body, *args, frozen=False):
    runner = """
import importlib.abc
import importlib.util
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
if sys.argv[2] == 'frozen':
    sys.frozen = True
    sys._MEIPASS = str(root)
# Preflight may discover installed capabilities but must never import them.
original_find_spec = importlib.util.find_spec
missing = ('torch', 'safetensors', 'transformers', 'tokenizers', 'sentencepiece', 'google.protobuf')
def find_spec(name, *args, **kwargs):
    return None if name in missing else original_find_spec(name, *args, **kwargs)
importlib.util.find_spec = find_spec
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in %r:
            raise AssertionError('supply helper imported backend: ' + fullname)
sys.meta_path.insert(0, NoBackend())
try:
%s
finally:
    assert not set(%r).intersection(sys.modules)
""" % (FORBIDDEN, "\n".join("    " + line for line in body.splitlines()), FORBIDDEN)
    return subprocess.run(
        [sys.executable, "-I", "-c", runner, str(resources), "frozen" if frozen else "source", *map(str, args)],
        cwd=tmp_path, env={**os.environ, "QLH_ANDROID_ROOT": str(tmp_path / "untrusted android")},
        capture_output=True, text=True, encoding="utf-8", errors="strict", timeout=30,
    )


@pytest.mark.parametrize("frozen", [False, True])
def test_packaged_verifier_is_pure_read_with_no_repo_on_sys_path(tmp_path, frozen_resources, source_gguf, frozen):
    before = hashlib.sha256(source_gguf.read_bytes()).hexdigest()
    result = _run_supply(tmp_path, frozen_resources, """
from scripts.model_tools.gguf import verify_gguf
report = verify_gguf(Path(sys.argv[3]), full_hash=True)
assert report['valid'], report
assert report['sha256'] == sys.argv[4]
print(json.dumps(report))
""", source_gguf, before, frozen=frozen)
    assert result.returncode == 0, result.stdout + result.stderr
    assert hashlib.sha256(source_gguf.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("frozen", [False, True])
def test_edge_conversion_refuses_before_subprocess_or_staging(tmp_path, frozen_resources, frozen):
    source = tmp_path / "hf model"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"]}), encoding="utf-8")
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"preflight never reads tensor content")
    result = _run_supply(tmp_path, frozen_resources, """
from scripts.model_tools import gguf_convert
converter, state = gguf_convert._find_converter(None)
assert state == 'available' and converter.is_relative_to(root)
assert 'Qwen2ForCausalLM' in gguf_convert._supported_architectures(converter)
def unexpected(*args, **kwargs):
    raise AssertionError('Edge conversion started a process')
gguf_convert.subprocess.run = unexpected
report = gguf_convert.execute_conversion(source=Path(sys.argv[3]), target=Path(sys.argv[4]),
                                        outtype='f16', confirmation='CONVERT')
assert report['execution']['started'] is False
assert report['writes_performed'] is False
assert report['toolchain']['dependencies']['status'] == 'missing'
assert 'torch' in report['toolchain']['dependencies']['missing']
assert 'converter_dependencies_missing' in [e['code'] for e in report['errors']]
assert not Path(sys.argv[4]).exists()
print(json.dumps(report))
""", source, tmp_path / "converted.gguf", frozen=frozen)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.glob(".qlh*"))


@pytest.mark.parametrize("frozen", [False, True])
def test_conversion_python_command_uses_correct_process_entry(tmp_path, frozen_resources, frozen):
    result = _run_supply(tmp_path, frozen_resources, """
from scripts.model_tools import gguf_convert
converter, _ = gguf_convert._find_converter(None)
command = gguf_convert._tool_command(converter, '--help')
if getattr(sys, 'frozen', False):
    assert command == [sys.executable, '--qlh-convert-hf-to-gguf', '--help'], command
    assert gguf_convert._tool_command(Path('quantizer.exe'), 'input') == ['quantizer.exe', 'input']
    try:
        gguf_convert._tool_command(Path('untrusted.py'), '--help')
    except ValueError as exc:
        assert 'model_supply_converter_untrusted' in str(exc)
    else:
        raise AssertionError('untrusted converter accepted')
else:
    assert command == [sys.executable, str(converter), '--help'], command
print(json.dumps(command))
""", frozen=frozen)
    assert result.returncode == 0, result.stdout + result.stderr


def test_full_specs_explicitly_collect_conversion_dependency_closure():
    for name in ("qlh-cpu.spec", "qlh-cuda.spec"):
        text = (PACKAGING / name).read_text(encoding="utf-8")
        assert "freeze_conversion=True" in text
        assert '"scripts.model_tools"' in text
    edge = (PACKAGING / "qlh-edge.spec").read_text(encoding="utf-8")
    assert "freeze_conversion=True" not in edge
    assert '"scripts.model_tools"' in edge
    for name in ("requirements-cpu.txt", "requirements-runtime-cpu.txt", "requirements-runtime-cuda.txt"):
        text = (PACKAGING / name).read_text(encoding="utf-8")
        for dependency in ("safetensors>=", "sentencepiece>=", "protobuf>="):
            assert dependency in text


def test_full_build_rejects_missing_conversion_native_dependency(monkeypatch):
    import model_supply_resources

    original = model_supply_resources.importlib.util.find_spec
    monkeypatch.setattr(model_supply_resources.importlib.util, "find_spec",
                        lambda name: None if name == "torch" else original(name))
    with pytest.raises(ModuleNotFoundError, match="conversion requires torch"):
        collect_model_supply_resources(CORE, freeze_conversion=True)


@pytest.mark.parametrize("profile, engine", [("torch_cpu", "cpu"), ("torch_cuda", "cuda"), ("llama_cpp_only", "cpu")])
def test_external_conversion_runtime_dependencies_follow_profile(tmp_path, profile, engine):
    import runtime_guard

    context = runtime_guard.RuntimeContext(root=tmp_path, profile=profile, engine=engine)
    required = set(runtime_guard.required_modules(context))
    assert ({"safetensors", "sentencepiece", "google.protobuf"} <= required) is (profile != "llama_cpp_only")
    commands = runtime_guard.build_pip_commands(tmp_path / "python", None, context)
    has_conversion_deps = any(any("sentencepiece" in part for part in command) for command in commands)
    assert has_conversion_deps is (profile != "llama_cpp_only")


def test_frozen_conversion_flag_is_dispatched_before_backend(tmp_path, frozen_resources):
    runner = """
import runpy
import sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
sys.frozen = True
sys._MEIPASS = sys.argv[2]
import importlib.util
original_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args, **kwargs: None if name in ('torch', 'safetensors', 'transformers', 'tokenizers', 'sentencepiece', 'google.protobuf') else original_find_spec(name, *args, **kwargs)
sys.argv = [sys.argv[3], '--qlh-convert-hf-to-gguf', '--help']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    assert not set(%r).intersection(sys.modules)
""" % (FORBIDDEN,)
    result = subprocess.run([sys.executable, "-I", "-c", runner, str(PACKAGING), str(frozen_resources), str(LAUNCHER)],
                            cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "converter_dependencies_missing" in result.stderr
    assert "torch" in result.stderr


def test_source_resource_staging_uses_same_whitelist(tmp_path):
    from model_supply_resources import stage_source_resources

    stage_source_resources(CORE, tmp_path)
    for source, relative in source_model_supply_resources(CORE):
        assert (tmp_path / relative / source.name).read_bytes() == source.read_bytes()


def test_production_source_metadata_uses_trusted_frozen_resources(tmp_path, frozen_resources, source_gguf):
    body = """
import model_supply_sources as source
assert Path(source.__file__).parent != root
assert Path(sys.executable).parent != root
module = source._model_tools('gguf')
assert Path(module.__file__).is_relative_to(root)
metadata = source._gguf_metadata(Path(sys.argv[3]))
assert len(metadata['artifact_sha256']) == 64
assert len(metadata['tokenizer_sha256']) == 64
print(json.dumps(metadata))
"""
    # The source module's real production function runs with its normal imports.
    # Only the subprocess import path supplies src; the resource root is the
    # simulated _MEIPASS. Neither the executable directory nor cwd has tools.
    body = "sys.path.append(" + repr(str(CORE / "src")) + ")\n" + body
    result = _run_supply(tmp_path, frozen_resources, body, source_gguf, frozen=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['artifact_sha256'] == hashlib.sha256(source_gguf.read_bytes()).hexdigest()


def test_frozen_source_ignores_hostile_developer_scripts_package(tmp_path, frozen_resources, source_gguf):
    body = """
import importlib.abc
import types
sys.path.append(%r)
import model_supply_sources as source
# Simulate a developer package already visible through the frozen archive.
poisoned = types.ModuleType('scripts.model_tools')
poisoned.__path__ = []
poisoned.verify_gguf = lambda *a, **k: (_ for _ in ()).throw(AssertionError('developer verifier'))
sys.modules['scripts.model_tools'] = poisoned
class DeveloperArchive(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'scripts' or name.startswith('scripts.model_tools'):
            raise AssertionError('frozen source entered developer archive: ' + name)
sys.meta_path.insert(0, DeveloperArchive())
module = source._model_tools('gguf')
assert module.__name__ == '_qlh_supply_tools.gguf'
assert type(module.__loader__).__name__ == 'SourceFileLoader'
assert Path(module.__file__).is_relative_to(root)
assert source._gguf_metadata(Path(sys.argv[3]))['artifact_sha256']
converter = source._model_tools('gguf_convert')
assert converter.__name__ == '_qlh_supply_tools.gguf_convert'
assert type(converter.__loader__).__name__ == 'SourceFileLoader'
assert type(converter._supply_runtime.__loader__).__name__ == 'SourceFileLoader'
assert converter._supply_runtime.__name__ == '_qlh_supply_tools.conversion_runtime'
assert converter.verify_gguf is module.verify_gguf
assert converter._tool_command(converter._find_converter(None)[0], '--help')[1] == '--qlh-convert-hf-to-gguf'
assert sys.modules['scripts.model_tools'] is poisoned
print(module.__name__)
""" % str(CORE / "src")
    result = _run_supply(tmp_path, frozen_resources, body, source_gguf, frozen=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '_qlh_supply_tools.gguf' in result.stdout


def test_developer_source_preserves_existing_module_identity(tmp_path):
    runner = """
import sys
import types
sys.path.insert(0, sys.argv[1])
import model_supply_sources as source
fake = types.ModuleType('scripts.model_tools.gguf')
sys.modules['scripts.model_tools.gguf'] = fake
assert source._model_tools('gguf') is fake
assert '_qlh_supply_tools' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-I', '-c', runner, str(CORE / 'src')],
                            cwd=tmp_path, capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('flag, function', [('--qlh-cut-layers', 'run_cut_layers'),
                                          ('--qlh-convert-hf-to-gguf', 'run_hf_conversion')])
def test_helper_unexpected_exception_is_traceback_and_nonzero(monkeypatch, capsys, flag, function):
    import model_supply_runtime as runtime

    def failed(_):
        raise RuntimeError('intentional helper boundary failure')
    monkeypatch.setattr(runtime, function, failed)
    assert runtime.dispatch_helper(flag, []) == 1
    stderr = capsys.readouterr().err
    assert 'model_supply_helper_failed:' in stderr
    assert 'Traceback' in stderr
    assert 'intentional helper boundary failure' in stderr


@pytest.mark.parametrize('code', [0, 1, 2])
def test_helper_preserves_cli_system_exit(monkeypatch, code):
    import model_supply_runtime as runtime

    def stopped(_):
        raise SystemExit(code)
    monkeypatch.setattr(runtime, 'run_cut_layers', stopped)
    assert runtime.dispatch_helper('--qlh-cut-layers', []) == code
