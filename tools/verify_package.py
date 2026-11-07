#!/usr/bin/env python3
"""Verify the portable runtime snapshot without importing torch or a platform."""
import hashlib
import json
from pathlib import Path


def main():
    kit = Path(__file__).resolve().parents[1]
    manifest = json.loads((kit / "runtime_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = (kit / "vendor" / name).resolve()
        if not path.is_relative_to((kit / "vendor").resolve()):
            raise ValueError("Snapshot path escapes vendor")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Runtime source changed: {name}")
    actual = {str(p.relative_to(kit / "vendor")) for p in (kit / "vendor").rglob("*.py")}
    if actual != set(manifest["files"]):
        raise ValueError("Runtime Python inventory differs")
    release_count = None
    release = kit / "package_manifest.json"
    if release.is_file():
        package = json.loads(release.read_text())
        for name, digest in package["files"].items():
            path = (kit / name).resolve()
            if not path.is_relative_to(kit.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Release file changed: {name}")
        release_count = len(package["files"])
    print(json.dumps({"passed": True, "runtime_sources_verified": len(actual), "release_files_verified": release_count}))


if __name__ == "__main__":
    main()
