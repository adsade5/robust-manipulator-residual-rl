from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [7, 17, 27]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce final Residual PPO v3 results.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--evaluate-only", action="store_true", help="Only aggregate existing per-seed evaluations.")
    parser.add_argument("--train-missing", action="store_true", help="Train missing seed artifacts before aggregation.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    seed_args = [str(seed) for seed in args.seeds]
    if not args.evaluate_only:
        runner_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_final_multiseed.py"),
            "--seeds",
            *seed_args,
            "--reuse-existing",
        ]
        if args.train_missing:
            run(runner_cmd)
        else:
            run(runner_cmd)
    run([sys.executable, str(PROJECT_ROOT / "evaluate" / "summarize_multiseed.py"), "--seeds", *seed_args])


if __name__ == "__main__":
    main()
