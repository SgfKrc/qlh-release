"""包体边界复验：构建后逐文件扫描发行目录，确认没有打进不属于该 profile 的东西。

## 为什么需要它（与 `install_manifest.py` 的分工）

`install_manifest.py` 产出并校验**签名清单** —— 它证明的是**文件完整性**
（磁盘上的文件与清单记载一致、未被篡改）。它**不证明依赖闭包**：清单里本就不该
出现的文件，只要构建时被顺手拷进去，签名照样通过。

本脚本补的正是后一半：拿**实际磁盘内容**去比对 profile 的**禁品集合**，并解析二进制
的实际导入表（PE / ELF），确认它没有链到不该链的库里。两者缺一不可 —— 前者的产物
是发布附件里的 manifest，后者是这里的 JSON 报告。

## 判据来源

* 禁品集合按 profile 分档，与 `runtime_guard.RUNTIME_PROFILES` 同源：
  - `llama_cpp_only`：**不得**出现 torch / transformers / accelerate 及其原生库
    （`torch_cpu` 档允许 torch，但两档都不得出现 CUDA 运行库）。
  - `torch_cuda`：允许 CUDA 运行库，仍不得出现开发残留。
* 「开发残留」对**所有** profile 一视同仁：构建脚本、测试、agent 配置、
  `__pycache__` 都不属于发行物。

## 用法

    python packaging/packaging/bundle_boundary_scan.py \\
        --bundle dist/QLH-Edge-Inference --profile llama_cpp_only \\
        --report build/audits/bundle-boundary-<date>.json

退出码：0 = 无违规；1 = 有违规；2 = 用法/IO 错误。
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA_VERSION = 1
REPORT_TYPE = "qlh_bundle_boundary"

#: 与 `runtime_guard.RUNTIME_PROFILES` 同集合（两边都改就一起改）。
RUNTIME_PROFILES: tuple[str, ...] = ("llama_cpp_only", "torch_cpu", "torch_cuda")

#: torch 系的模块名与原生库前缀。出现在文件路径或二进制的导入表里都算违规
#: （后者是"包体里没这个名字的文件，但某个 DLL 链到了它"—— 只扫文件名会漏掉）。
TORCH_MODULE_NAMES: tuple[str, ...] = ("torch", "torchvision", "torchaudio", "transformers", "accelerate", "safetensors")
TORCH_NATIVE_PREFIXES: tuple[str, ...] = (
    "torch_", "torch-", "libtorch", "_C.cp", "libc10", "c10.dll",
    "torch_cpu.dll", "torch_cuda.dll", "torch_python.dll",
)

#: CUDA 运行库前缀。任何 profile 都不该出现在**发行包体**里 —— 边缘包靠运行时
#: 外部 runtime 目录提供，PC CUDA 档则由安装器按需装到 runtime 目录外。
CUDA_NATIVE_PREFIXES: tuple[str, ...] = (
    "cudart", "cublas", "cudnn", "cufft", "curand", "cusolver", "cusparse",
    "nvrtc", "npp", "nccl", "nvToolsExt", "cupti", "cuda.dll", "nvcuda",
)

#: 构建/开发残留（目录名，按 basename 匹配，大小写不敏感）。
DEV_PATH_NAMES: frozenset[str] = frozenset({
    "__pycache__", ".agents", ".claude", ".github", ".vscode",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
})
#: 构建/开发残留（文件名 glob，小写比较）。**只对我们自己的源码树生效** —— 第三方
#: 包（PyInstaller 收在 `_internal/`）自带 `test_*.py` 是它的正常组成部分，报出来
#: 就是噪声（实测：torch 的 `_dynamo/test_case.py` 之类被误判了 18 条）。
DEV_FILE_PATTERNS: tuple[str, ...] = (
    "*.spec", "build-*.bat", "build-*.sh", "makefile", "install-manifest.json.tmp",
    "conftest.py", "test_*.py", "*_test.py", "pull_request_template.md",
)
#: 第三方包根目录：这棵子树里的一切都按"上游内容"对待，不做源码残留判据。
THIRD_PARTY_ROOT = "_internal"
#: 二进制扩展名。CUDA/torch 的**原生库**判据只对它们生效 —— `cudart.py` / `nccl.py`
#: 是 torch 的 Python 封装模块，不是运行库（实测误报）。
NATIVE_EXTENSIONS: frozenset[str] = frozenset({".dll", ".so", ".dylib", ".pyd"})
#: agent / 协作配置文件名。
AGENT_FILE_NAMES: frozenset[str] = frozenset({
    "agents.md", "claude.md", "copilot-instructions.md", ".cursorrules",
})

#: PE / ELF magic。
_PE_MAGIC = b"MZ"
_ELF_MAGIC = b"\x7fELF"


@dataclass
class Violation:
    kind: str
    path: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "path": self.path, "detail": self.detail}


@dataclass
class ScanResult:
    bundle: str
    profile: str
    file_count: int = 0
    total_bytes: int = 0
    directory_count: int = 0
    binary_count: int = 0
    imports_checked: int = 0
    violations: list[Violation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "report_type": REPORT_TYPE,
            "bundle": self.bundle,
            "profile": self.profile,
            "ok": self.ok,
            "file_count": self.file_count,
            "directory_count": self.directory_count,
            "total_bytes": self.total_bytes,
            "binary_count": self.binary_count,
            "imports_checked": self.imports_checked,
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "notes": self.notes,
        }


def _lower_names(path_parts: Sequence[str]) -> list[str]:
    return [part.lower() for part in path_parts]


def _matches_module_name(lowered: str, module: str) -> bool:
    """`torch` 命中 `torch/`、`torch-2.4.dist-info`、`torch.dll`，但不命中 `torchlike`。

    ⚠️ **不**匹配 `torch_<something>.py`：那是我们自己的模块名（`src/torch_runtime.py`
    检测 torch 可用性、`src/torch_hetero_plan.py` 之类），把它们当 torch 包报出来是
    误报（实测 4 条）。原生库的 `torch_*.dll` 由 `NATIVE_EXTENSIONS` 分支单独判。
    """
    return (
        lowered == module
        or lowered.startswith(module + ".")
        or lowered.startswith(module + "-")
        or lowered == module + ".dll"
        or lowered == module + ".so"
    )


def classify_path(relative: Path, profile: str) -> tuple[str, str] | None:
    """返回 `(kind, detail)`；`None` 表示这条路径没有违规。

    只看**路径**（文件名 + 目录名）。二进制内部的导入闭包由 `scan_binary_imports`
    负责 —— 两者互补：路径扫描抓"文件本身不该在"，导入扫描抓"文件在但链到了禁品"。
    """
    parts = list(relative.parts)
    lowered_parts = _lower_names(parts)
    basename = lowered_parts[-1] if lowered_parts else ""
    in_third_party = (
        len(lowered_parts) > 1 and lowered_parts[0] == THIRD_PARTY_ROOT.lower()
    )
    suffix = relative.suffix.lower()
    # `.so.12` 这类版本化后缀的 `suffix` 是 `.12` ⇒ 额外判 `.so` 系，否则 Linux 的
    # 共享库全部漏判（实测：`libcudart.so.12` 没被认出来）。
    is_native = (
        suffix in NATIVE_EXTENSIONS
        or basename.endswith(".so")
        or ".so." in basename
    )

    for part in lowered_parts[:-1]:
        if part in DEV_PATH_NAMES:
            return "dev_residue", f"目录 `{part}` 不属于发行物"
    if basename in DEV_PATH_NAMES:
        return "dev_residue", f"`{basename}` 不属于发行物"

    # 源码残留模式只对我们自己的树生效；`_internal/` 下是上游包的原样内容。
    if not in_third_party:
        for pattern in DEV_FILE_PATTERNS:
            if _glob_match(basename, pattern):
                return "dev_residue", f"匹配开发残留模式 `{pattern}`"
    if basename in AGENT_FILE_NAMES:
        return "agent_config", f"agent 协作配置 `{basename}` 不应随包发行"

    if profile == "llama_cpp_only":
        for module in TORCH_MODULE_NAMES:
            if any(_matches_module_name(part, module) for part in lowered_parts):
                return "torch_in_llama_cpp_only", (
                    f"`llama_cpp_only` profile 不得含 {module} 系文件（命中 `{basename}`）"
                )
        if is_native and any(
            _native_name_matches(basename, prefix)
            for prefix in TORCH_NATIVE_PREFIXES
        ):
            return "torch_in_llama_cpp_only", f"torch 原生库 `{basename}`"

    # CUDA 运行库只可能是**原生库**；`cudart.py` / `nccl.py` 是 torch 的 Python 封装。
    if is_native and any(
        _native_name_matches(basename, prefix) for prefix in CUDA_NATIVE_PREFIXES
    ):
        return "cuda_runtime_in_bundle", (
            f"CUDA 运行库 `{basename}` 不应打进包体（应由外部 runtime 目录提供）"
        )
    return None


def _glob_match(name: str, pattern: str) -> bool:
    if "*" not in pattern and "?" not in pattern:
        return name == pattern
    regex = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return re.fullmatch(regex, name) is not None


def _iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _dir_count(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_dir())


def is_pe(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(2) == _PE_MAGIC
    except OSError:
        return False


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(4) == _ELF_MAGIC
    except OSError:
        return False


def _native_name_matches(basename: str, prefix: str) -> bool:
    """原生库名前缀匹配，同时接受 Unix 的 `lib` 前缀。

    `cudart64_12.dll`（Windows）与 `libcudart.so.12`（Linux）是同一个库的两种命名，
    只匹配前者会让 Linux 包体的 CUDA 检出**静默失效**（实测：测试里 `libcudart.so.12`
    没被认出来）。
    """
    return basename.startswith(prefix) or basename.startswith("lib" + prefix)


def pe_imported_dlls(path: Path) -> list[str]:
    """解析 PE 的导入表，返回被导入的 DLL 名（小写）。

    用 `pefile`（可选依赖）。不可用时抛 `RuntimeError`，由调用方决定降级策略 ——
    静默返回空列表会让"没装 pefile"伪装成"没有违规"，正是本脚本要防的错。
    """
    try:
        import pefile  # type: ignore import-not-found
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise RuntimeError("pefile 不可用，无法解析 PE 导入表") from exc
    try:
        pe = pefile.PE(str(path), fast_load=True)
    except Exception as exc:  # noqa: BLE001 - pefile 抛的异常类型很杂
        raise RuntimeError(f"PE 解析失败: {exc}") from exc
    names: list[str] = []
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]],
        )
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
            dll = entry.dll
            if isinstance(dll, bytes):
                dll = dll.decode("utf-8", "replace")
            names.append(str(dll).lower())
    finally:
        pe.close()
    return names


def elf_imported_libs(path: Path) -> list[str]:
    """解析 ELF 的动态依赖（`DT_NEEDED`），返回库名（小写）。

    只做最小解析：定位 section header 里的 `.dynamic` 段，读 `DT_NEEDED` 指向的
    `.dynstr` 字符串。不引入外部依赖，够用于"链没链到 libtorch/cudart"这一判据。
    """
    data = path.read_bytes()
    if data[:4] != _ELF_MAGIC:
        return []
    is64 = data[4] == 2
    little = data[5] == 1
    endian = "<" if little else ">"
    if is64:
        e_shoff, = struct.unpack_from(endian + "Q", data, 0x28)
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(endian + "HHH", data, 0x3A)
        sh_fmt, sh_size = endian + "IIQQQQIIQQ", 64
    else:
        e_shoff, = struct.unpack_from(endian + "I", data, 0x20)
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(endian + "HHH", data, 0x2E)
        sh_fmt, sh_size = endian + "IIIIIIIIII", 40
    if not e_shoff or not e_shnum:
        return []

    sections: list[tuple[int, int, int, int, int]] = []
    for index in range(e_shnum):
        off = e_shoff + index * e_shentsize
        if off + sh_size > len(data):
            return []
        fields = struct.unpack_from(sh_fmt, data, off)
        # name, type, flags, addr, offset, size, link, info, align, entsize
        sections.append((fields[0], fields[1], fields[4], fields[5], fields[6]))

    # `.dynamic` 的 type 是 6（SHT_DYNAMIC），它的 sh_link 指向字符串表。
    dyn = next((s for s in sections if s[1] == 6), None)
    if dyn is None:
        return []
    strtab_index = dyn[4]
    if not (0 <= strtab_index < len(sections)):
        return []
    strtab_off, strtab_size = sections[strtab_index][2], sections[strtab_index][3]
    strtab = data[strtab_off:strtab_off + strtab_size]

    names: list[str] = []
    off, size = dyn[2], dyn[3]
    step = 16 if is64 else 8
    d_tag_off, d_val_off = (0, 8) if is64 else (0, 4)
    d_fmt = endian + ("Qq" if is64 else "Ii")
    for pos in range(off, off + size - step + 1, step):
        tag, val = struct.unpack_from(d_fmt, data, pos)
        if tag == 0:  # DT_NULL
            break
        if tag != 1:  # DT_NEEDED
            continue
        if 0 <= val < len(strtab):
            end = strtab.find(b"\x00", val)
            raw = strtab[val:end if end != -1 else len(strtab)]
            names.append(raw.decode("utf-8", "replace").lower())
    return names


def forbidden_import(name: str, profile: str) -> tuple[str, str] | None:
    """判断一个被导入的库名是否违规。"""
    if profile == "llama_cpp_only":
        for module in TORCH_MODULE_NAMES:
            if _matches_module_name(name, module) or name.startswith(module):
                return "torch_import_in_llama_cpp_only", f"导入 `{name}`"
        if any(
            _native_name_matches(name, prefix) for prefix in TORCH_NATIVE_PREFIXES
        ):
            return "torch_import_in_llama_cpp_only", f"导入 `{name}`"
    if any(
        _native_name_matches(name, prefix) for prefix in CUDA_NATIVE_PREFIXES
    ):
        return "cuda_import_in_bundle", f"导入 `{name}`"
    return None


def scan_bundle(
    root: Path,
    *,
    profile: str,
    scan_imports: bool = True,
    max_binaries: int | None = None,
) -> ScanResult:
    result = ScanResult(bundle=str(root), profile=profile)
    if not root.is_dir():
        raise FileNotFoundError(f"不是目录: {root}")
    result.directory_count = _dir_count(root)

    for path in _iter_files(root):
        relative = path.relative_to(root)
        result.file_count += 1
        try:
            result.total_bytes += path.stat().st_size
        except OSError:
            pass
        hit = classify_path(relative, profile)
        if hit is not None:
            kind, detail = hit
            result.violations.append(
                Violation(kind=kind, path=relative.as_posix(), detail=detail),
            )

    if not scan_imports:
        result.notes.append("已跳过二进制导入表扫描（--no-imports）")
        return result

    binaries = 0
    for path in _iter_files(root):
        if max_binaries is not None and binaries >= max_binaries:
            result.notes.append(f"导入扫描在 {max_binaries} 个二进制处截断")
            break
        relative = path.relative_to(root)
        try:
            if is_pe(path):
                imports = pe_imported_dlls(path)
            elif is_elf(path):
                imports = elf_imported_libs(path)
            else:
                continue
        except RuntimeError as exc:
            result.violations.append(
                Violation(
                    kind="import_scan_unavailable",
                    path=relative.as_posix(),
                    detail=str(exc),
                ),
            )
            continue
        binaries += 1
        result.binary_count += 1
        for name in imports:
            result.imports_checked += 1
            hit = forbidden_import(name, profile)
            if hit is not None:
                kind, detail = hit
                result.violations.append(
                    Violation(kind=kind, path=relative.as_posix(), detail=detail),
                )
    return result


def _render_text(result: ScanResult) -> str:
    lines = [
        f"包体边界复验: {result.bundle}",
        f"  profile={result.profile}  files={result.file_count}  "
        f"dirs={result.directory_count}  "
        f"size={result.total_bytes / (1024 * 1024):.1f}MB  "
        f"binaries={result.binary_count}  imports={result.imports_checked}",
    ]
    if result.ok:
        lines.append("  [OK] 无违规")
    else:
        lines.append(f"  [FAIL] {len(result.violations)} 项违规:")
        for violation in result.violations[:40]:
            lines.append(f"    [{violation.kind}] {violation.path} — {violation.detail}")
        if len(result.violations) > 40:
            lines.append(f"    …另有 {len(result.violations) - 40} 项")
    for note in result.notes:
        lines.append(f"  · {note}")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="包体边界复验：逐文件扫发行目录，比对 profile 禁品集合与二进制导入表",
    )
    parser.add_argument("--bundle", required=True, help="发行目录（解包后的包体根）")
    parser.add_argument(
        "--profile", required=True, choices=RUNTIME_PROFILES,
        help="该包体声明的 runtime profile",
    )
    parser.add_argument("--report", help="输出 JSON 报告路径（建议纳入发布附件）")
    parser.add_argument(
        "--no-imports", action="store_true",
        help="跳过二进制导入表扫描（只做路径级检查；不建议用于发布门）",
    )
    parser.add_argument(
        "--max-binaries", type=int, default=None,
        help="最多解析多少个二进制（调试用；发布门不要设）",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    # Windows 控制台默认 GBK：报告里既有中文（违规说明）又有 emoji 类字符时，
    # `print` 会 `UnicodeEncodeError` 直接把整份报告吞掉（实测踩到）。改成
    # 无法编码时替换，保证报告永远打得出来。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass

    args = _parse_args(argv)
    root = Path(args.bundle)
    try:
        result = scan_bundle(
            root,
            profile=args.profile,
            scan_imports=not args.no_imports,
            max_binaries=args.max_binaries,
        )
    except FileNotFoundError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    print(_render_text(result))
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  报告: {report}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
