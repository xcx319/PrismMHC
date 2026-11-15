"""Verify installed scientific data against the release SHA256 manifest."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    root = args.data_dir.resolve()
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text())
    failures = []
    for relative, entry in manifest.items():
        parts = PurePosixPath(relative)
        path = (root / relative).resolve()
        if parts.is_absolute() or ".." in parts.parts or not path.is_relative_to(root):
            raise ValueError(f"Unsafe manifest path {relative!r}")
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            failures.append({"path": relative, "error": "hash_or_size_mismatch"})
    print(json.dumps({"files": len(manifest), "failures": failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()

