from __future__ import annotations

import argparse
import hashlib
import importlib
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


PATCH_FILENAMES = (
    "bins_arena.xml",
    "empty_arena.xml",
    "multi_table_arena.xml",
    "pegs_arena.xml",
    "table_arena.xml",
)
PATCH_ROOT = REPO_ROOT / "assets" / "robosuite_patch" / "arenas"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_robosuite_assets_dir() -> Path:
    robosuite = importlib.import_module("robosuite")
    return Path(robosuite.__file__).resolve().parent / "models" / "assets" / "arenas"


def get_patch_status() -> tuple[Path, list[dict[str, object]]]:
    target_dir = get_robosuite_assets_dir()
    statuses: list[dict[str, object]] = []

    for name in PATCH_FILENAMES:
        source = PATCH_ROOT / name
        target = target_dir / name
        exists = target.exists()
        matches = exists and source.exists() and sha256(source) == sha256(target)
        statuses.append(
            {
                "name": name,
                "source": source,
                "target": target,
                "exists": exists,
                "matches": matches,
            }
        )
    return target_dir, statuses


def patch_is_applied() -> bool:
    _, statuses = get_patch_status()
    return all(bool(status["matches"]) for status in statuses)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the VILA robosuite arena camera patch to the active Python environment."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify whether the patch is already installed.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .pre_vila_patch.bak backups before overwriting files.",
    )
    args = parser.parse_args()

    target_dir, statuses = get_patch_status()

    missing_sources = [str(status["source"]) for status in statuses if not Path(status["source"]).exists()]
    if missing_sources:
        raise SystemExit(
            "Missing robosuite patch assets in the repository:\n" + "\n".join(missing_sources)
        )

    if args.check:
        if all(bool(status["matches"]) for status in statuses):
            print(f"[OK] robosuite patch present in {target_dir}")
            return
        print(f"[FAIL] robosuite patch missing or stale in {target_dir}")
        for status in statuses:
            if bool(status["matches"]):
                continue
            target = Path(status["target"])
            state = "missing" if not bool(status["exists"]) else "mismatch"
            print(f"  - {status['name']}: {state} ({target})")
        raise SystemExit(1)

    updated = 0
    for status in statuses:
        source = Path(status["source"])
        target = Path(status["target"])

        if bool(status["matches"]):
            print(f"[SKIP] {target.name} already matches")
            continue

        if target.exists() and not args.no_backup:
            backup = target.with_suffix(target.suffix + ".pre_vila_patch.bak")
            if not backup.exists():
                shutil.copy2(target, backup)
                print(f"[BACKUP] {backup}")

        shutil.copy2(source, target)
        print(f"[PATCH] {source.name} -> {target}")
        updated += 1

    print(f"[OK] robosuite patch ready in {target_dir} ({updated} file(s) updated)")


if __name__ == "__main__":
    main()
