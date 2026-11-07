#!/usr/bin/env python3
"""Refresh a portable Python source snapshot; no runs, data or credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


KIT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Explicit source tree containing jade_lotus/ and scripts/probe_resident_openpi_fault.py")
    parser.add_argument("--refresh", action="store_true", help="Explicitly replace only the vendored source snapshot")
    args = parser.parse_args()
    root = args.source_root.resolve()
    vendor = KIT / "vendor"
    if vendor.exists() and not args.refresh:
        parser.error("vendor exists; use --refresh only when intentionally updating the runtime")
    controller = root / "scripts/probe_resident_openpi_fault.py"
    modules = sorted((root / "jade_lotus").glob("*.py"))
    inputs = [*modules, controller]
    if (not (root / "jade_lotus/__init__.py").is_file() or not modules
            or any(not source.is_file() or source.is_symlink() for source in inputs)):
        parser.error("source-root must contain ordinary jade_lotus/*.py sources and the resident controller; existing vendor is unchanged")
    # Build everything before replacing the working snapshot. An invalid source
    # or failed copy must never remove the runtime of the independent project.
    with tempfile.TemporaryDirectory(prefix=".runtime-refresh-", dir=KIT) as temporary:
        staging = Path(temporary)
        staged_vendor = staging / "vendor"
        (staged_vendor / "jade_lotus").mkdir(parents=True)
        (staged_vendor / "scripts").mkdir()
        sources = {}
        for source in inputs:
            relative = source.relative_to(root)
            destination = staged_vendor / relative
            shutil.copy2(source, destination)
            sources[str(relative)] = sha(destination)
        manifest = {
            "format": "fragment-training-kit-source-snapshot-v1",
            "files": sources,
            "source_set_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            "scope": "Python sources only. No runs, weights, datasets, inventory, cookie or password files.",
            "openpi": "Optional upstream environment/model/data are supplied separately.",
        }
        staged_manifest = staging / "runtime_manifest.json"
        staged_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        backup = staging / "previous-vendor"
        if vendor.exists():
            os.replace(vendor, backup)
        try:
            os.replace(staged_vendor, vendor)
            os.replace(staged_manifest, KIT / "runtime_manifest.json")
        except BaseException:
            if vendor.exists():
                shutil.rmtree(vendor)
            if backup.exists():
                os.replace(backup, vendor)
            raise
    print(json.dumps({"vendor": str(vendor), "source_files": len(sources), "source_set_sha256": manifest["source_set_sha256"]}))


if __name__ == "__main__":
    main()
