from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2: normal-speed fine-tuning from Phase 1 checkpoints.")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--trainer-pool", default="league_pool_filtered",
                        help="Opponent pool folder. Default is the balanced 40-model runtime pool. Use league_pool_all_submitted only for the full 211-model member archive.")
    parser.add_argument("--build-path", default=None,
                        help="Path to the Unity Training build dp.exe (or set DPICKLEBALL_TRAINING_EXE).")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable, "run_6_learners_parallel.py",
        "--max-parallel", str(args.max_parallel),
        "--trainer-pool", args.trainer_pool,
        "--start-source", "latest",
        "--resume-root", "outputs/six_learners_tscale2_800k",
        "--output-root", "outputs/six_learners_tscale1_finetune_200k",
        "--steps", str(args.steps),
        "--chunk-steps", "40000",
        "--monitor",
        "--monitor-interval", "60",
        "--time-scale", "1",
        "--target-frame-rate", "-1",
        "--learning-rate", "0.00005",
        "--ent-coef", "0.003",
        "--n-steps", "2048",
        "--batch-size", "512",
        "--n-epochs", "2",
        "--save-every-chunks", "1",
        "--add-selfplay-to-pool",
    ]
    if args.build_path:
        cmd += ["--build-path", args.build_path]
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
