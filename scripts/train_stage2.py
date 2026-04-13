import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


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


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_temp_json(data: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
        return handle.name


def set_nested(mapping: dict, path: list[str], value: Any) -> None:
    cursor = mapping
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


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
    parser.add_argument("--config", required=True, help="Path to a stage-2 JSON config")
    parser.add_argument("--dataset-path", help="Override dataset path")
    parser.add_argument("--encoder-checkpoint", help="Override VILA encoder checkpoint path")
    parser.add_argument("--output-dir", help="Override training output directory")
    parser.add_argument("--name", help="Override experiment name")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    args = parser.parse_args()

    config = expand_vars(load_json(args.config))

    if args.dataset_path is not None:
        for dataset_entry in config["train"]["data"]:
            dataset_entry["path"] = args.dataset_path
    if args.encoder_checkpoint is not None:
        set_nested(
            config,
            ["observation", "encoder", "rgb", "core_kwargs", "backbone_kwargs", "model_path"],
            args.encoder_checkpoint,
        )
    if args.output_dir is not None:
        config["train"]["output_dir"] = args.output_dir
    if args.name is not None:
        config["experiment"]["name"] = args.name
    config = resolve_dataset_paths(config)

    unresolved = find_unresolved_vars(config)
    if unresolved:
        preview = ", ".join(unresolved[:5])
        raise ValueError(f"Unresolved config variables remain: {preview}")

    if args.dry_run:
        print(json.dumps(config, indent=4))
        return

    resolved_config = write_temp_json(config)
    cmd = [
        sys.executable,
        "-m",
        "robomimic.scripts.train_multiview",
        "--config",
        resolved_config,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
