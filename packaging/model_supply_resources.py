"""Explicit release resources for GGUF verification, cutting and HF conversion."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
import sys
from pathlib import Path

CONVERTER_DIRECTORY = Path("android/app/src/main/cpp/llama.cpp")


def source_model_supply_resources(project_root: str | Path) -> list[tuple[Path, str]]:
    """Share a product whitelist between frozen, source-backed and Linux bundles."""
    root = Path(project_root)
    adapters = Path(__file__).parent / "model_supply_package"
    resources = [
        (root / "scripts" / "cut_layers.py", "scripts"),
        (root / "src" / "llama_keep_head_worker.py", "."),
        (root / "scripts/model_tools/gguf.py", "scripts/model_tools"),
        (root / "scripts/model_tools/llama_quantize_toolchain.py", "scripts/model_tools"),
        (root / "scripts/model_tools/llama_quantize.lock.json", "scripts/model_tools"),
        (root / "scripts/model_tools/gguf_convert.py", "scripts/model_tools/_canonical"),
        (adapters / "scripts/__init__.py", "scripts"),
        (adapters / "__init__.py", "scripts/model_tools"),
        (adapters / "gguf_convert.py", "scripts/model_tools"),
        (adapters / "conversion_runtime.py", "scripts/model_tools"),
        (root / CONVERTER_DIRECTORY / "convert_hf_to_gguf.py", CONVERTER_DIRECTORY.as_posix()),
        (root / CONVERTER_DIRECTORY / "LICENSE", CONVERTER_DIRECTORY.as_posix()),
        (root / CONVERTER_DIRECTORY / "gguf-py/LICENSE", (CONVERTER_DIRECTORY / "gguf-py").as_posix()),
    ]
    for relative, required in (
        ("conversion", ("__init__.py", "base.py")),
        ("gguf-py/gguf", ("__init__.py", "gguf_reader.py", "gguf_writer.py")),
        ("requirements", ("requirements-convert_hf_to_gguf.txt", "requirements-convert_legacy_llama.txt")),
    ):
        directory = root / CONVERTER_DIRECTORY / relative
        for name in required:
            if not (directory / name).is_file():
                raise FileNotFoundError(f"model_supply_resource_missing: {directory / name}")
        if relative == "requirements":
            resources.extend((directory / name, (CONVERTER_DIRECTORY / relative).as_posix())
                             for name in required)
        else:
            # Only module sources/typing data, never caches, test fixtures or
            # unrelated llama.cpp binaries. Conversion modules import lazily.
            for source in sorted(directory.rglob("*")):
                if source.is_file() and source.suffix in {".py", ".typed"}:
                    destination = CONVERTER_DIRECTORY / relative / source.parent.relative_to(directory)
                    resources.append((source, destination.as_posix()))
    for source, _ in resources:
        if not source.is_file():
            raise FileNotFoundError(f"model_supply_resource_missing: {source}")
    return resources


def _resource_imports(resources, *, freeze_conversion: bool) -> list[str]:
    """Disk-executed sources still need their standard/native import dependencies."""
    imports = {"gguf", "numpy", "yaml", "tqdm", "requests", "relay_contract", "llama_keep_head",
               "model_supply_sources", "relay_segment_info"}
    conversion_packages = {
        "torch", "transformers", "safetensors", "tokenizers", "sentencepiece",
        "huggingface_hub", "google", "tiktoken",
    }
    for source, _ in resources:
        if source.suffix != ".py":
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                package = name.split(".")[0]
                if package in sys.stdlib_module_names or (freeze_conversion and package in conversion_packages):
                    if package != "__future__":
                        imports.add(name)
    if freeze_conversion:
        imports.update({"tokenizers", "google.protobuf", "sentencepiece.sentencepiece_model_pb2"})
    return sorted(imports)


def collect_model_supply_resources(project_root: str | Path, *, freeze_conversion: bool = False) -> tuple[list, list[str]]:
    """Fail an incomplete build; resource copies alone cannot ship native imports."""
    from PyInstaller.utils.hooks import collect_data_files

    resources = source_model_supply_resources(project_root)
    if importlib.util.find_spec("gguf") is None:
        raise ModuleNotFoundError("model_supply_dependency_missing: install gguf==0.19.0 in the build venv")
    if freeze_conversion:
        for name in ("torch", "transformers", "safetensors", "tokenizers", "sentencepiece", "google.protobuf"):
            try:
                available = importlib.util.find_spec(name) is not None
            except ModuleNotFoundError:
                available = False
            if not available:
                raise ModuleNotFoundError(f"model_supply_dependency_missing: conversion requires {name}")
    gguf_sources = collect_data_files(
        "gguf", include_py_files=True,
        excludes=["**/__pycache__/**", "**/*.pyc", "**/*.pyo"],
    )
    if not any(Path(source).name == "__init__.py" and dest == "gguf" for source, dest in gguf_sources):
        raise FileNotFoundError("model_supply_resource_missing: gguf/__init__.py")
    datas = [(str(source), dest) for source, dest in resources] + gguf_sources
    return datas, _resource_imports(resources, freeze_conversion=freeze_conversion)


def stage_source_resources(project_root: str | Path, destination: str | Path) -> None:
    """Copy the same source whitelist without requiring PyInstaller on Linux."""
    for source, relative in source_model_supply_resources(project_root):
        target = Path(destination) / relative / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--stage-source-root", required=True, type=Path)
    args = parser.parse_args()
    stage_source_resources(args.project_root, args.stage_source_root)
