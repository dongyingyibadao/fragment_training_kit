#!/usr/bin/env python3
"""Record release-file hashes after integration and acceptance have finished."""
import hashlib
import json
from pathlib import Path


def main():
    kit = Path(__file__).resolve().parents[1]
    files = {}
    for path in sorted(kit.rglob("*")):
        if not path.is_file() or path.name == "package_manifest.json":
            continue
        relative = path.relative_to(kit)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts) or path.suffix == ".pyc":
            continue
        files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"format": "fragment-training-kit-release-v1", "files": files,
              "file_set_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
              "scope": "Runtime, adapters, launcher, docs, examples, tests and acceptance evidence including tiny validation state. Credentials, external datasets and large model weights excluded."}
    (kit / "package_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"sealed_files": len(files), "file_set_sha256": result["file_set_sha256"]}))


if __name__ == "__main__":
    main()
