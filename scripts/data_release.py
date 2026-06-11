"""Package and publish the data/ tree as GitHub Release assets.

Why this exists: data/ is ~4 GB (datasets, golden, models, captures). It is
gitignored by policy (docs/DATA_POLICY.md) but a fresh clone still needs to
reproduce a known-good snapshot to train or run inference. GitHub Releases
gives us versioned 2 GB-per-file storage for free, separate from the code
history.

Three subcommands form the round-trip:

    # On the machine holding the source-of-truth data/:
    python scripts/data_release.py pack --version 0.1.0
    python scripts/data_release.py publish --version 0.1.0 --notes "First cut"

    # On a fresh clone:
    python scripts/data_release.py fetch --version 0.1.0

Tag scheme: ``data-v<MAJOR>.<MINOR>.<PATCH>`` keeps data versions independent
from code release tags (see docs/DATA_RELEASES.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("data_release")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
STAGING_ROOT = PROJECT_ROOT / "dist" / "releases"
DEFAULT_DIRS = ("datasets", "golden", "models", "captures")
MANIFEST_NAME = "MANIFEST.json"
ASSET_EXT = ".tar"
CHUNK = 1024 * 1024


@dataclass
class AssetEntry:
    name: str
    size: int
    sha256: str

    def to_dict(self) -> dict:
        return {"name": self.name, "size": self.size, "sha256": self.sha256}


def tag_for(version: str) -> str:
    return f"data-v{version}"


def staging_dir_for(version: str) -> Path:
    return STAGING_ROOT / tag_for(version)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _relativize_symlinks(src_root: Path):
    """Rewrite absolute symlinks that point inside DATA_ROOT into paths relative
    to the symlink's own location, so the archive is portable to any machine
    (and survives tarfile's `data` extraction filter).

    Source symlinks are absolute on this machine (e.g. anomalib `latest -> /home/.../v0`
    and prepare_anomaly_dataset's panel-mirror links). Same content; just rewrite
    the linkname for the archive — the on-disk tree is untouched.
    """
    abs_src = src_root.resolve()
    data_abs = DATA_ROOT.resolve()

    def _filter(info: tarfile.TarInfo):
        if info.issym() and info.linkname.startswith("/"):
            try:
                target_in_data = PurePosixPath(info.linkname).relative_to(data_abs)
            except ValueError:
                logger.warning(
                    "Symlink points outside data/, leaving absolute: %s -> %s",
                    info.name, info.linkname,
                )
                return info
            arc_dir = PurePosixPath(info.name).parent
            info.linkname = os.path.relpath(str(target_in_data), str(arc_dir))
        return info

    _ = abs_src  # quiet linters; kept for future per-src checks
    return _filter


def tar_directory(src: Path, dst: Path) -> None:
    """Uncompressed tar — PNG/ckpt barely compress, and tar preserves symlinks
    (the dataset trees use them to share captures across panels)."""
    with tarfile.open(dst, "w") as tf:
        tf.add(src, arcname=src.name, recursive=True, filter=_relativize_symlinks(src))


def extract_tar(src: Path, dst_root: Path) -> None:
    with tarfile.open(src) as tf:
        try:
            tf.extractall(dst_root, filter="data")
        except TypeError:
            # Python <3.12 — no `filter` kwarg. Trusted internal release, safe.
            tf.extractall(dst_root)


def cmd_pack(args: argparse.Namespace) -> int:
    requested = list(args.dirs or DEFAULT_DIRS)
    missing = [d for d in requested if not (DATA_ROOT / d).is_dir()]
    if missing:
        logger.error("Missing data dirs: %s", ", ".join(missing))
        return 1

    staging = staging_dir_for(args.version)
    if staging.exists():
        if not args.force:
            logger.error("%s exists. Pass --force to overwrite.", staging)
            return 1
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    assets: list[AssetEntry] = []
    for name in requested:
        src = DATA_ROOT / name
        dst = staging / f"{name}{ASSET_EXT}"
        logger.info("Packing %s -> %s", src, dst.name)
        tar_directory(src, dst)
        size = dst.stat().st_size
        digest = sha256_of(dst)
        assets.append(AssetEntry(dst.name, size, digest))
        logger.info("  %s  %s", human_bytes(size), digest[:12])

    manifest = {
        "version": args.version,
        "tag": tag_for(args.version),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": "data/",
        "assets": [a.to_dict() for a in assets],
    }
    (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    total = sum(a.size for a in assets)
    over_2gb = [a for a in assets if a.size > 2 * 1024**3]
    logger.info("Packed %d asset(s), total %s -> %s", len(assets), human_bytes(total), staging)
    if over_2gb:
        logger.warning(
            "Assets exceed GitHub Release 2GB-per-file cap: %s",
            ", ".join(a.name for a in over_2gb),
        )
        return 2
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    staging = staging_dir_for(args.version)
    manifest_path = staging / MANIFEST_NAME
    if not manifest_path.exists():
        logger.error("No manifest at %s. Run pack first.", manifest_path)
        return 1

    manifest = json.loads(manifest_path.read_text())
    tag = manifest["tag"]
    asset_paths = [staging / a["name"] for a in manifest["assets"]] + [manifest_path]
    missing = [p for p in asset_paths if not p.exists()]
    if missing:
        logger.error("Missing assets: %s", ", ".join(str(p) for p in missing))
        return 1

    notes_body = args.notes or f"Data snapshot {tag}\n\nSee docs/DATA_RELEASES.md for restore instructions."
    cmd = [
        "gh", "release", "create", tag,
        "--title", tag,
        "--notes", notes_body,
    ]
    if args.draft:
        cmd.append("--draft")
    if args.prerelease:
        cmd.append("--prerelease")
    cmd += [str(p) for p in asset_paths]

    logger.info("Creating release %s with %d asset(s)", tag, len(asset_paths))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def cmd_fetch(args: argparse.Namespace) -> int:
    tag = tag_for(args.version)
    staging = staging_dir_for(args.version)
    staging.mkdir(parents=True, exist_ok=True)

    patterns = ["MANIFEST.json"]
    if args.dirs:
        patterns += [f"{d}{ASSET_EXT}" for d in args.dirs]
    else:
        patterns.append(f"*{ASSET_EXT}")

    dl_cmd = ["gh", "release", "download", tag, "--dir", str(staging), "--clobber"]
    for pat in patterns:
        dl_cmd += ["-p", pat]
    logger.info("Downloading %s to %s", tag, staging)
    rc = subprocess.run(dl_cmd, cwd=PROJECT_ROOT).returncode
    if rc != 0:
        return rc

    manifest_path = staging / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    by_name = {a["name"]: a for a in manifest["assets"]}
    requested = [f"{d}{ASSET_EXT}" for d in args.dirs] if args.dirs else list(by_name)

    for name in requested:
        archive_path = staging / name
        if name not in by_name:
            logger.warning("%s not in manifest — skipping", name)
            continue
        expected = by_name[name]["sha256"]
        actual = sha256_of(archive_path)
        if actual != expected:
            logger.error("Checksum mismatch for %s\n  expected %s\n  got      %s", name, expected, actual)
            return 1
        logger.info("Verified %s (%s)", name, human_bytes(archive_path.stat().st_size))

    DATA_ROOT.mkdir(exist_ok=True)
    for name in requested:
        archive_path = staging / name
        logger.info("Extracting %s -> %s", name, DATA_ROOT)
        extract_tar(archive_path, DATA_ROOT)

    if not args.keep_archives:
        for name in requested:
            (staging / name).unlink(missing_ok=True)
        logger.info("Removed local archives (pass --keep-archives to retain)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    common_version = {"required": True, "help": "Semantic version, e.g. 0.1.0"}

    pp = sub.add_parser("pack", help="Zip data/ subdirs and write MANIFEST.json")
    pp.add_argument("--version", **common_version)
    pp.add_argument("--dirs", nargs="+", choices=DEFAULT_DIRS, help="Subset of data/ dirs to pack")
    pp.add_argument("--force", action="store_true", help="Overwrite existing staging dir")
    pp.set_defaults(func=cmd_pack)

    pu = sub.add_parser("publish", help="Create GitHub Release and upload zips")
    pu.add_argument("--version", **common_version)
    pu.add_argument("--notes", help="Release notes body (markdown)")
    pu.add_argument("--draft", action="store_true")
    pu.add_argument("--prerelease", action="store_true")
    pu.set_defaults(func=cmd_publish)

    pf = sub.add_parser("fetch", help="Download release assets and extract into data/")
    pf.add_argument("--version", **common_version)
    pf.add_argument("--dirs", nargs="+", choices=DEFAULT_DIRS, help="Subset to fetch")
    pf.add_argument("--keep-archives", action="store_true", help="Don't delete archives after extracting")
    pf.set_defaults(func=cmd_fetch)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
