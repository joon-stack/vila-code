import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

from apply_robosuite_patch import get_patch_status

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def import_and_report(module_name: str) -> object:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    if version is not None:
        ok(f"import {module_name} ({version})")
    else:
        ok(f"import {module_name}")
    return module


def run_dry_run(args: list[str]) -> None:
    cmd = [sys.executable, *args, "--dry-run"]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    ok(f"dry-run {' '.join(args)}")


def find_lift_dataset(root: Path) -> tuple[Path | None, str | None]:
    public_path = root / "lift" / "lift_25v.hdf5"
    if public_path.exists():
        return public_path, None

    legacy_path = root / "robomimic" / "lift" / "ph" / "lift_multiview_25v_hard_rand_64x64.hdf5"
    if legacy_path.exists():
        return legacy_path, "legacy internal dataset layout detected"

    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Do not fail if CUDA is unavailable.",
    )
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 10):
        fail(f"Python 3.10 is required, found {sys.version.split()[0]}")
    ok(f"python {sys.version.split()[0]}")

    torch = import_and_report("torch")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        ok(f"CUDA available ({torch.cuda.device_count()} device(s))")
    elif args.allow_no_cuda:
        warn("CUDA is not available")
    else:
        fail("CUDA is not available")

    for module_name in [
        "numpy",
        "h5py",
        "yaml",
        "pyrallis",
        "wandb",
        "torchinfo",
        "matplotlib",
        "imageio",
        "PIL",
        "einops",
        "einx",
        "diffusers",
        "transformers",
        "huggingface_hub",
        "tensorboardX",
        "termcolor",
        "gymnasium",
        "robosuite",
    ]:
        import_and_report(module_name)

    robosuite_assets_dir, patch_statuses = get_patch_status()
    stale_patch_files = [str(status["name"]) for status in patch_statuses if not bool(status["matches"])]
    if stale_patch_files:
        fail(
            "robosuite multiview camera patch is not installed for the active environment. "
            "Run `python scripts/apply_robosuite_patch.py` and try again. "
            f"Missing or stale files in {robosuite_assets_dir}: {', '.join(stale_patch_files)}"
        )
    ok(f"robosuite multiview camera patch {robosuite_assets_dir}")

    if cuda_available:
        for module_name in ["shimmy", "dm_control", "mujoco", "egl_probe"]:
            import_and_report(module_name)
    else:
        warn("skipping GPU/EGL imports: shimmy, dm_control, mujoco, egl_probe")

    robomimic = import_and_report("robomimic")
    robomimic_path = Path(robomimic.__file__).resolve()
    if not robomimic_path.is_relative_to(REPO_ROOT):
        fail(f"robomimic import escaped repo: {robomimic_path}")
    ok(f"local robomimic import {robomimic_path}")

    if cuda_available:
        import_and_report("vila_stage1.train")
    else:
        warn("skipping vila_stage1.train import on non-GPU node")
    import_and_report("robomimic.scripts.train_multiview")
    import_and_report("robomimic.scripts.run_trained_agent")

    robotics_data_root = os.environ.get("ROBOTICS_DATA_ROOT")
    if not robotics_data_root:
        fail("ROBOTICS_DATA_ROOT is not set")
    robotics_data_root_path = Path(robotics_data_root).expanduser()
    if not robotics_data_root_path.exists():
        fail(f"ROBOTICS_DATA_ROOT does not exist: {robotics_data_root_path}")
    ok(f"ROBOTICS_DATA_ROOT={robotics_data_root_path}")

    lift_dataset, lift_dataset_note = find_lift_dataset(robotics_data_root_path)
    if lift_dataset is None:
        fail(
            "lift dataset not found under either the public layout "
            "(`lift/lift_25v.hdf5`) or the internal layout "
            "(`robomimic/lift/ph/lift_multiview_25v_hard_rand_64x64.hdf5`)."
        )
    if lift_dataset_note is not None:
        warn(lift_dataset_note)
    ok(f"canonical lift dataset {lift_dataset}")

    encoder_checkpoint = os.environ.get("VILA_ENCODER_CHECKPOINT")
    if not encoder_checkpoint:
        fail("VILA_ENCODER_CHECKPOINT is not set")
    encoder_checkpoint_path = Path(encoder_checkpoint).expanduser()
    if not encoder_checkpoint_path.exists():
        fail(f"VILA encoder checkpoint missing: {encoder_checkpoint_path}")
    ok(f"VILA_ENCODER_CHECKPOINT={encoder_checkpoint_path}")

    mujoco_gl = os.environ.get("MUJOCO_GL")
    if mujoco_gl != "egl":
        warn(f"MUJOCO_GL is '{mujoco_gl}', expected 'egl' for cluster evaluation")
    else:
        ok("MUJOCO_GL=egl")

    wandb_mode = os.environ.get("WANDB_MODE")
    if wandb_mode != "disabled":
        warn(f"WANDB_MODE is '{wandb_mode}', recommended 'disabled'")
    else:
        ok("WANDB_MODE=disabled")

    run_dry_run([
        "scripts/train_stage1.py",
        "--config",
        "configs/stage1/lift.yaml",
    ])
    run_dry_run([
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2/lift/finetune_10v.json",
        "--encoder-checkpoint",
        str(encoder_checkpoint_path),
    ])
    run_dry_run([
        "scripts/eval_stage2.py",
        "--checkpoint",
        "outputs/stage2/lift/dummy/models/model_epoch_1.pth",
        "--camera-names",
        "view_00",
    ])

    ok("environment check passed")


if __name__ == "__main__":
    main()
