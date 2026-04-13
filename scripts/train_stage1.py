import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_RELEASES = (
    {
        "public_path": "lift/lift_25v.hdf5",
        "source_path": "robomimic/lift/ph/lift_multiview_25v_hard_rand_64x64.hdf5",
    },
)


def expand_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_vars(v) for v in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def find_unresolved_vars(value: Any, prefix: str = "") -> list[str]:
    issues: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            issues.extend(find_unresolved_vars(nested, child_prefix))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            issues.extend(find_unresolved_vars(nested, child_prefix))
    elif isinstance(value, str) and "$" in value:
        issues.append(f"{prefix}={value}")
    return issues


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_temp_yaml(data: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
        return handle.name


def resolve_dataset_path(path: str) -> str:
    candidate = Path(path)
    if candidate.exists() or candidate.suffix != ".hdf5":
        return path

    for entry in DATASET_RELEASES:
        public_rel = Path(entry["public_path"])
        source_rel = Path(entry["source_path"])
        if len(candidate.parts) < len(public_rel.parts):
            continue
        if candidate.parts[-len(public_rel.parts):] != public_rel.parts:
            continue
        root = Path(*candidate.parts[:-len(public_rel.parts)])
        legacy_path = root / source_rel
        if legacy_path.exists():
            return str(legacy_path)

    return path


def resolve_dataset_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: resolve_dataset_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_dataset_paths(v) for v in value]
    if isinstance(value, str):
        return resolve_dataset_path(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a stage-1 YAML config")
    parser.add_argument("--dataset-path", help="Override dataset path for all stage-1 stages")
    parser.add_argument("--output-root", help="Override root output directory for checkpoints")
    parser.add_argument("--run-name", help="Optional fixed run name")
    parser.add_argument("--seed", type=int, help="Optional seed override")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    args = parser.parse_args()

    config = expand_vars(load_yaml(args.config))

    if args.seed is not None:
        config["seed"] = args.seed
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.output_root is not None:
        config["output_root"] = args.output_root

    if args.dataset_path is not None:
        config.setdefault("lapo", {})
        config["lapo"]["data_path"] = args.dataset_path
        config.setdefault("bc", {})
        config["bc"]["data_path"] = args.dataset_path

    config = resolve_dataset_paths(config)

    unresolved = find_unresolved_vars(config)
    if unresolved:
        preview = ", ".join(unresolved[:5])
        raise ValueError(f"Unresolved config variables remain: {preview}")

    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        return

    resolved_config = write_temp_yaml(config)
    cmd = [
        sys.executable,
        "-m",
        "vila_stage1.train",
        "--config_path",
        resolved_config,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
