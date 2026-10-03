"""包体边界复验的回归测试。

判据分两类，各有用例：
* **路径级**：profile 禁品集合 + 开发残留 —— 覆盖 `llama_cpp_only` 与 `torch_cpu`
  的分档，以及"第三方包自带 `test_*.py` 不算我们的残留"这条（实测误报过 18 条）。
* **导入级**：解析 PE 的导入表，确认"文件在但链到了禁品"也能被抓到 —— 只扫文件名的
  做法漏这一类。
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packaging'))

import bundle_boundary_scan as scan  # noqa: E402


def _write(root: Path, relative: str, content: bytes = b"") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class TestProfileGating:
    def test_torch_is_a_violation_in_llama_cpp_only(self, tmp_path):
        _write(tmp_path, "_internal/torch/lib/torch.dll", b"MZ")
        hit = scan.classify_path(
            Path("_internal/torch/lib/torch.dll"), "llama_cpp_only",
        )
        assert hit is not None and hit[0] == "torch_in_llama_cpp_only"

    def test_torch_is_fine_in_torch_cpu(self, tmp_path):
        hit = scan.classify_path(
            Path("_internal/torch/lib/torch.dll"), "torch_cpu",
        )
        assert hit is None

    def test_transformers_and_accelerate_are_also_gated(self):
        for name in ("transformers", "accelerate", "safetensors"):
            hit = scan.classify_path(
                Path(f"_internal/{name}/__init__.py"), "llama_cpp_only",
            )
            assert hit is not None, name

    def test_dist_info_of_torch_is_hit(self):
        hit = scan.classify_path(
            Path("_internal/torch-2.12.0+cpu.dist-info/METADATA"), "llama_cpp_only",
        )
        assert hit is not None and hit[0] == "torch_in_llama_cpp_only"


class TestDevResidue:
    def test_our_own_tests_are_residue(self):
        hit = scan.classify_path(Path("src/test_scheduler.py"), "torch_cpu")
        assert hit is not None and hit[0] == "dev_residue"

    def test_third_party_tests_are_not_residue(self):
        # torch 自带 `_dynamo/test_case.py` 是上游包的正常内容，不是我们的残留。
        hit = scan.classify_path(
            Path("_internal/torch/_dynamo/test_case.py"), "torch_cpu",
        )
        assert hit is None

    def test_spec_and_build_scripts_are_residue(self):
        for name in ("qlh-slim.spec", "build-cpu.bat", "conftest.py"):
            assert scan.classify_path(Path(name), "torch_cpu") is not None, name

    def test_agent_config_is_rejected(self):
        hit = scan.classify_path(Path("AGENTS.md"), "torch_cpu")
        assert hit is not None and hit[0] == "agent_config"

    def test_pycache_is_residue(self):
        hit = scan.classify_path(Path("src/__pycache__/x.pyc"), "torch_cpu")
        assert hit is not None and hit[0] == "dev_residue"


class TestCudaNativeOnly:
    def test_cuda_python_module_is_not_a_runtime_lib(self):
        # `cudart.py` / `nccl.py` 是 torch 的 Python 封装，不是运行库（实测误报）。
        assert scan.classify_path(Path("_internal/torch/cuda/nccl.py"), "torch_cpu") is None

    def test_cuda_native_library_is_rejected(self):
        for name in ("cudart64_12.dll", "cublas64_12.dll", "libcudart.so.12"):
            hit = scan.classify_path(Path("_internal") / name, "torch_cpu")
            assert hit is not None and hit[0] == "cuda_runtime_in_bundle", name


class TestImportScan:
    def test_pe_imports_are_read(self):
        """用系统 dll 验证 PE 导入表能真读出来（不依赖发行产物）。"""
        candidates = [
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "kernel32.dll",
        ]
        target = next((p for p in candidates if p.is_file()), None)
        if target is None:
            pytest.skip("系统 dll 不可用")
        imports = scan.pe_imported_dlls(target)
        assert imports, "kernel32.dll 应当有导入"

    def test_forbidden_import_detects_torch_and_cuda(self):
        assert scan.forbidden_import("torch_cpu.dll", "llama_cpp_only") is not None
        assert scan.forbidden_import("libtorch.so", "llama_cpp_only") is not None
        assert scan.forbidden_import("cudart64_12.dll", "torch_cpu") is not None
        assert scan.forbidden_import("kernel32.dll", "torch_cpu") is None
        # torch_cpu 档下 torch 自身不算违规
        assert scan.forbidden_import("torch_cpu.dll", "torch_cpu") is None


class TestElfImportScan:
    def test_elf_imports_are_read(self):
        """若能找到系统 .so，验证 ELF 的 DT_NEEDED 解析。"""
        candidates = [Path("/usr/lib/x86_64-linux-gnu/libc.so.6"),
                      Path("/lib/x86_64-linux-gnu/libc.so.6")]
        target = next((p for p in candidates if p.is_file()), None)
        if target is None:
            pytest.skip("系统 .so 不可用（非 Linux）")
        libs = scan.elf_imported_libs(target)
        assert libs, "libc.so.6 应当有 DT_NEEDED"


class TestScanBundleEndToEnd:
    def test_clean_bundle_reports_ok(self, tmp_path):
        _write(tmp_path, "src/api_server.py", b"x = 1\n")
        _write(tmp_path, "run.bat", b"@echo off\n")
        result = scan.scan_bundle(tmp_path, profile="torch_cpu", scan_imports=False)
        assert result.ok, [v.to_dict() for v in result.violations]
        assert result.file_count == 2

    def test_violating_bundle_is_reported(self, tmp_path):
        _write(tmp_path, "_internal/torch/__init__.py", b"")
        result = scan.scan_bundle(tmp_path, profile="llama_cpp_only", scan_imports=False)
        assert not result.ok
        assert any(v.kind == "torch_in_llama_cpp_only" for v in result.violations)

    def test_missing_bundle_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scan.scan_bundle(tmp_path / "nope", profile="torch_cpu")

    def test_report_shape_is_stable(self, tmp_path):
        _write(tmp_path, "src/a.py", b"")
        report = scan.scan_bundle(
            tmp_path, profile="torch_cpu", scan_imports=False,
        ).to_dict()
        assert report["schema_version"] == scan.SCHEMA_VERSION
        assert report["report_type"] == scan.REPORT_TYPE
        assert report["ok"] is True
        assert report["violation_count"] == 0
        assert isinstance(report["violations"], list)
