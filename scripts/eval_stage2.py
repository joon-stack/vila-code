import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to a trained diffusion-policy checkpoint")
    parser.add_argument("--camera-names", nargs="+", default=["agentview"], help="Camera name(s) for evaluation")
    parser.add_argument("--n-rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--video-path", help="Optional output video path")
    parser.add_argument("--dataset-path", help="Optional output rollout dataset path")
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit")
    args = parser.parse_args()

    checkpoint_stem = Path(args.checkpoint).stem
    if args.video_path is None:
        args.video_path = os.path.join("outputs", "eval", f"{checkpoint_stem}_{args.camera_names[0]}.mp4")

    cmd = [
        sys.executable,
        "-m",
        "robomimic.scripts.run_trained_agent",
        "--agent",
        args.checkpoint,
        "--n_rollouts",
        str(args.n_rollouts),
        "--horizon",
        str(args.horizon),
        "--seed",
        str(args.seed),
        "--video_path",
        args.video_path,
        "--camera_names",
        *args.camera_names,
    ]
    if args.dataset_path is not None:
        cmd.extend(["--dataset_path", args.dataset_path])

    if args.dry_run:
        print(" ".join(cmd))
        return

    Path(args.video_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
