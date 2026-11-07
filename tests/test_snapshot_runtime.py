"""A maintenance refresh must not destroy the standalone runtime on bad input."""
from pathlib import Path
import shutil
import subprocess
import sys


TOOL = Path(__file__).resolve().parents[1] / "tools/snapshot_runtime.py"


def project(tmp_path):
    kit = tmp_path / "standalone"
    (kit / "tools").mkdir(parents=True)
    shutil.copy2(TOOL, kit / "tools/snapshot_runtime.py")
    (kit / "vendor/jade_lotus").mkdir(parents=True)
    (kit / "vendor/jade_lotus/__init__.py").write_text("# keep existing runtime\n")
    (kit / "runtime_manifest.json").write_text('{"original": true}\n')
    return kit


def invoke(kit, *args):
    return subprocess.run([sys.executable, str(kit / "tools/snapshot_runtime.py"), *args],
                          capture_output=True, text=True)


def test_refresh_requires_explicit_source_and_preserves_existing_runtime(tmp_path):
    kit = project(tmp_path)
    before = (kit / "vendor/jade_lotus/__init__.py").read_bytes()
    for args in (("--refresh",), ("--refresh", "--source-root", str(tmp_path / "missing"))):
        result = invoke(kit, *args)
        assert result.returncode != 0
        assert (kit / "vendor/jade_lotus/__init__.py").read_bytes() == before
        assert (kit / "runtime_manifest.json").read_text() == '{"original": true}\n'


def test_explicit_valid_source_replaces_and_verifies_the_snapshot(tmp_path):
    kit = project(tmp_path)
    source = tmp_path / "explicit-source"
    (source / "jade_lotus").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "jade_lotus/__init__.py").write_text("# new runtime\n")
    (source / "scripts/probe_resident_openpi_fault.py").write_text("# new controller\n")
    result = invoke(kit, "--refresh", "--source-root", str(source))
    assert result.returncode == 0, result.stderr
    assert (kit / "vendor/jade_lotus/__init__.py").read_text() == "# new runtime\n"
    verifier = TOOL.with_name("verify_package.py")
    shutil.copy2(verifier, kit / "tools/verify_package.py")
    result = subprocess.run([sys.executable, str(kit / "tools/verify_package.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
