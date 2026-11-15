"""Download and verify the versioned scientific data assets."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shutil
import stat
import urllib.request
import zipfile

ASSETS = (
    "PrismMHC_Data_1_2.zip",
    "PrismMHC_Data_3_4.zip",
    "PrismMHC_Data_5.zip",
    "PrismMHC_Data_6.zip",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "PrismMHC-reproduction"})
    temporary = destination.with_name(destination.name + ".partial")
    with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 << 20)
    os.replace(temporary, destination)


def extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        for member in members:
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe archive path {member.filename!r}")
            if not path.parts or path.parts[0] != "supplementary_data":
                raise ValueError(f"Unexpected archive root {member.filename!r}")
            if stat.S_ISLNK(mode):
                raise ValueError(f"Archive symlink {member.filename!r}")
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive path escapes destination {member.filename!r}")
        source.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="xcx319/PrismMHC")
    parser.add_argument("--tag", default="manuscript-2026-11-16")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    cache = args.data_dir / ".downloads" / args.tag
    cache.mkdir(parents=True, exist_ok=True)
    api_url = f"https://api.github.com/repos/{args.repository}/releases/tags/{args.tag}"
    request = urllib.request.Request(api_url, headers={"User-Agent": "PrismMHC-reproduction"})
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    urls = {asset["name"]: asset["browser_download_url"] for asset in release["assets"]}
    required = (*ASSETS, "SHA256SUMS.txt", "RELEASE_MANIFEST.json")
    missing = [name for name in required if name not in urls]
    if missing:
        raise RuntimeError(f"Release is missing required assets: {missing}")
    download(urls["SHA256SUMS.txt"], cache / "SHA256SUMS.txt")
    expected = {}
    for line in (cache / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            expected[name.lstrip("*")] = digest
    for name in ASSETS:
        if name not in expected:
            raise RuntimeError(f"No checksum for {name}")
        target = cache / name
        if not target.exists() or sha256(target) != expected[name]:
            print(f"Downloading {name}", flush=True)
            download(urls[name], target)
        if sha256(target) != expected[name]:
            raise RuntimeError(f"Checksum mismatch for {name}")
        extract(target, args.data_dir)
        print(f"Installed {name}", flush=True)
    target = args.data_dir / "RELEASE_MANIFEST.json"
    download(urls["RELEASE_MANIFEST.json"], target)
    if target.name in expected and sha256(target) != expected[target.name]:
        raise RuntimeError("Release manifest checksum mismatch")
    print(f"Data installed in {args.data_dir.resolve()}")


if __name__ == "__main__":
    main()

