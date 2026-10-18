"""Validated Windows x64 CPU keep-head closure for the edge product.

The existing build_keep_head_shim.ps1 produces this MinGW group. Never search
PATH, collect an entire build tree, or substitute pip llama_cpp DLLs. Inspect PE
metadata without loading DLLs: the build process may already use pip's ABI.
"""
from __future__ import annotations

from pathlib import Path

NATIVE_RELATIVE = Path("build/keephead/build-cpu/bin")
PRIVATE_NATIVE_DIR = "keep_head"
REQUIRED_DLLS = (
    "qlh_keep_head.dll",
    "libllama.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu.dll",
    "libgcc_s_seh-1.dll",
    "libstdc++-6.dll",
    "libwinpthread-1.dll",
    "libgomp-1.dll",
)
REQUIRED_EXPORTS = {
    "qlh_keep_head.dll": {
        "qlh_kh_load", "qlh_kh_forward", "qlh_kh_forward_embd",
        "qlh_kh_forward_embd_seq", "qlh_kh_reset", "qlh_kh_n_embd",
        "qlh_kh_n_layer", "qlh_kh_close",
    },
    "libllama.dll": {
        "llama_set_embeddings_layer_inp", "llama_get_embeddings_layer_inp",
        "llama_set_embeddings_nextn", "llama_get_embeddings_nextn_ith",
    },
}
# This collector supports the existing UCRT MinGW CPU build, not arbitrary
# backends. API sets are provided by the supported Windows OS, not the build host.
_SYSTEM_DLLS = {"kernel32.dll", "advapi32.dll", "ntdll.dll", "ucrtbase.dll"}


def _inspect_pe(path: Path) -> tuple[dict[str, set[str]], set[str]]:
    import pefile  # PyInstaller's Windows build dependency; never ctypes-load.

    try:
        pe = pefile.PE(str(path), fast_load=True)
        try:
            if pe.FILE_HEADER.Machine != 0x8664:
                raise RuntimeError(f"keep_head_native_invalid_architecture: {path}: expected Windows x64")
            pe.parse_data_directories(directories=[
                pefile.DIRECTORY_ENTRY[name] for name in (
                    "IMAGE_DIRECTORY_ENTRY_IMPORT", "IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT",
                    "IMAGE_DIRECTORY_ENTRY_EXPORT",
                )
            ])
            imports: dict[str, set[str]] = {}
            for table in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
                for entry in getattr(pe, table, ()):
                    symbols = imports.setdefault(entry.dll.decode("ascii").lower(), set())
                    symbols.update(
                        item.name.decode("ascii") if item.name else f"#{item.ordinal}"
                        for item in entry.imports
                    )
            exports: set[str] = set()
            for item in getattr(getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", ()):
                exports.add(f"#{item.ordinal}")
                if item.name:
                    exports.add(item.name.decode("ascii"))
            return imports, exports
        finally:
            pe.close()
    except (pefile.PEFormatError, OSError, UnicodeError) as exc:
        raise RuntimeError(f"keep_head_native_invalid: {path}: {exc}") from exc


def collect_keep_head_binaries(project_root: str | Path) -> list[tuple[str, str, str]]:
    """Return a validated COLLECT TOC, bypassing Analysis's DLL relocation.

    Analysis recursively adds non-site-packages dependencies at the shared bundle
    root. Supplying this closed group directly to COLLECT preserves the private
    destinations and prevents same-name ggml dependencies leaking into pip's ABI.
    """
    native_dir = Path(project_root).resolve() / NATIVE_RELATIVE
    missing = [str(native_dir / name) for name in REQUIRED_DLLS
               if not (native_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "keep_head_native_missing: " + ", ".join(missing)
            + "; build patched llama.cpp and scripts/model_tools/build_keep_head_shim.ps1 first"
        )
    metadata = {name: _inspect_pe(native_dir / name) for name in REQUIRED_DLLS}
    for name, required in REQUIRED_EXPORTS.items():
        absent = required - metadata[name][1]
        if absent:
            raise RuntimeError(
                f"keep_head_native_symbol_missing: {native_dir / name}: {', '.join(sorted(absent))}"
            )
    for name, (imports, _) in metadata.items():
        for dependency, required in imports.items():
            if dependency in _SYSTEM_DLLS or dependency.startswith(("api-ms-win-", "ext-ms-win-")):
                continue
            if dependency not in metadata:
                raise FileNotFoundError(
                    f"keep_head_native_dependency_missing: {native_dir / name} requires {dependency}; "
                    "outside the approved CPU native group"
                )
            absent = required - metadata[dependency][1]
            if absent:
                raise RuntimeError(
                    f"keep_head_native_import_missing: {native_dir / name} -> {dependency}: "
                    + ", ".join(sorted(absent))
                )
    return [(f"{PRIVATE_NATIVE_DIR}/{name}", str(native_dir / name), "BINARY")
            for name in REQUIRED_DLLS]
