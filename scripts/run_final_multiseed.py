from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.residual_tracking_env import ResidualTrackingEnv


DEFAULT_SEEDS = [7, 17, 27]
RESULT_ROOT = PROJECT_ROOT / "results" / "final_multiseed"
SEED7_EXISTING_DIR = PROJECT_ROOT / "results" / "residual_ppo_v3"
SCALES = [1.00, 1.25, 1.50, 1.75, 2.00]
ACTION_PENALTY_WEIGHT = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final Residual PPO v3 multi-seed evaluation.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def seed_dir(seed: int) -> Path:
    return RESULT_ROOT / f"seed_{seed}"


def source_dir_for_seed(seed: int, reuse_existing: bool) -> Path:
    if seed == 7 and reuse_existing and (SEED7_EXISTING_DIR / "checkpoints" / "final_model.zip").exists():
        return SEED7_EXISTING_DIR
    return seed_dir(seed)


def final_artifacts_exist(result_dir: Path) -> bool:
    ckpt = result_dir / "checkpoints"
    return (ckpt / "final_model.zip").exists() and (ckpt / "vecnormalize_final.pkl").exists()


def train_seed(seed: int, result_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "train" / "train_residual_ppo_v3.py"),
        "--seed",
        str(seed),
        "--result-dir",
        str(result_dir),
    ]
    print(f"Training seed {seed}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def run_ctc_only(scale: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = ResidualTrackingEnv(
        fixed_inertial_scale=scale,
        randomize_scale=False,
        action_penalty_weight=ACTION_PENALTY_WEIGHT,
        seed=2000 + int(scale * 100),
    )
    env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(np.zeros(7, dtype=np.float32))
    metrics = dict(env.last_episode_metrics)
    metrics["terminated"] = bool(terminated)
    metrics["truncated"] = bool(truncated)
    metrics["nan_or_inf"] = False
    metrics["instability"] = bool(terminated)
    log = env.export_episode_log()
    env.close()
    return metrics, log


def make_vec_env(scale: float, vecnormalize_path: Path) -> VecNormalize:
    def _make() -> ResidualTrackingEnv:
        return ResidualTrackingEnv(
            fixed_inertial_scale=scale,
            randomize_scale=False,
            action_penalty_weight=ACTION_PENALTY_WEIGHT,
            seed=1000 + int(scale * 100),
        )

    env = DummyVecEnv([_make])
    vec = VecNormalize.load(vecnormalize_path, env)
    vec.training = False
    vec.norm_reward = False
    return vec


def run_residual(model: PPO, vecnormalize_path: Path, scale: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = make_vec_env(scale, vecnormalize_path)
    obs = env.reset()
    done = np.array([False])
    metrics = None
    log = None
    while not bool(done[0]):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        if bool(done[0]):
            metrics = infos[0].get("episode_metrics")
            log = infos[0].get("episode_log")
            terminal_info = infos[0]
    if metrics is None or log is None:
        raise RuntimeError("Residual evaluation ended without episode metrics")
    metrics = dict(metrics)
    metrics["terminated"] = bool(terminal_info.get("terminal_observation") is not None and not metrics)
    metrics["truncated"] = True
    metrics["nan_or_inf"] = False
    metrics["instability"] = False
    env.close()
    return metrics, log


def safe_scale(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def save_npz(path: Path, log: dict[str, np.ndarray]) -> None:
    np.savez(path, **log)


def evaluate_seed(seed: int, artifact_dir: Path, output_dir: Path) -> None:
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "checkpoints" / "final_model.zip"
    vecnormalize_path = artifact_dir / "checkpoints" / "vecnormalize_final.pkl"
    if not model_path.exists() or not vecnormalize_path.exists():
        raise FileNotFoundError(f"Missing final artifacts for seed {seed}: {artifact_dir}")

    model = PPO.load(model_path)
    rows = []
    results = []
    for scale in SCALES:
        ctc_metrics, ctc_log = run_ctc_only(scale)
        ppo_metrics, ppo_log = run_residual(model, vecnormalize_path, scale)
        for controller, metrics, log in [
            ("ctc", ctc_metrics, ctc_log),
            ("ppo_v3", ppo_metrics, ppo_log),
        ]:
            metrics = dict(metrics)
            metrics["seed"] = seed
            metrics["scale"] = scale
            metrics["controller"] = controller
            metrics_path = eval_dir / f"scale_{safe_scale(scale)}_{controller}_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            save_npz(eval_dir / f"scale_{safe_scale(scale)}_{controller}.npz", log)
        improvement = 100.0 * (ctc_metrics["motion_rmse"] - ppo_metrics["motion_rmse"]) / ctc_metrics["motion_rmse"]
        row = {
            "seed": seed,
            "scale": scale,
            "ctc_motion_rmse": ctc_metrics["motion_rmse"],
            "ppo_motion_rmse": ppo_metrics["motion_rmse"],
            "improvement_percent": improvement,
            "ppo_overall_rmse": ppo_metrics["overall_rmse"],
            "ppo_max_error": ppo_metrics["max_tracking_error"],
            "ppo_torque_rms": ppo_metrics["torque_rms"],
            "ppo_residual_torque_rms": ppo_metrics["residual_torque_rms"],
            "ppo_max_residual_torque": ppo_metrics["max_residual_torque"],
            "ppo_action_rms": ppo_metrics["action_rms"],
            "ppo_total_torque_clipping_count": ppo_metrics["total_torque_clipping_count"],
            "ppo_residual_action_clipping_count": ppo_metrics["residual_action_clipping_count"],
            "ppo_nan_or_inf": ppo_metrics["nan_or_inf"],
            "ppo_instability": ppo_metrics["instability"],
        }
        rows.append(row)
        results.append({"scale": scale, "ctc": ctc_metrics, "ppo_v3": ppo_metrics, "improvement_percent": improvement})
        print(
            f"seed={seed} scale={scale:.2f} CTC={ctc_metrics['motion_rmse']:.8f} "
            f"PPO={ppo_metrics['motion_rmse']:.8f} improvement={improvement:.2f}%"
        )

    (eval_dir / "seed_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (eval_dir / "seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def guard_summary(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "training_eval_log.json"
    if not path.exists():
        return {"guard_passing_checkpoint_exists": False, "best_checkpoint_timestep": 0}
    records = json.loads(path.read_text(encoding="utf-8"))
    passing = [record for record in records if not record.get("nominal_guard_failed", True)]
    best_timestep = 0
    summary_path = result_dir / "training_summary.json"
    if summary_path.exists():
        best_timestep = int(json.loads(summary_path.read_text(encoding="utf-8")).get("best_checkpoint_timestep", 0))
    return {
        "guard_passing_checkpoint_exists": bool(passing),
        "guard_passing_timesteps": [int(record["timesteps"]) for record in passing],
        "best_checkpoint_timestep": best_timestep,
    }


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    run_summary = []
    for seed in args.seeds:
        out_dir = seed_dir(seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = source_dir_for_seed(seed, args.reuse_existing)
        reused = artifact_dir != out_dir
        if not final_artifacts_exist(artifact_dir):
            if args.reuse_existing and seed == 7:
                print("Seed 7 existing v3 artifacts incomplete; training into final_multiseed/seed_7.")
                artifact_dir = out_dir
                reused = False
            if not final_artifacts_exist(artifact_dir):
                train_seed(seed, artifact_dir)
        else:
            print(f"Seed {seed}: using existing artifacts from {artifact_dir}")

        evaluate_seed(seed, artifact_dir, out_dir)
        summary = {
            "seed": seed,
            "artifact_dir": repo_relative(artifact_dir),
            "output_dir": repo_relative(out_dir),
            "reused_existing": reused,
            **guard_summary(artifact_dir),
        }
        train_summary_path = artifact_dir / "training_summary.json"
        if train_summary_path.exists():
            train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
            summary["requested_timesteps"] = train_summary.get("total_timesteps")
            summary["actual_timesteps"] = train_summary.get("actual_model_timesteps")
            summary["nominal_episode_fraction"] = train_summary.get("sample_fractions", {}).get("nominal_episode_fraction")
            summary["mismatch_episode_fraction"] = train_summary.get("sample_fractions", {}).get("mismatch_episode_fraction")
        run_summary.append(summary)

    (RESULT_ROOT / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"saved run summary = {RESULT_ROOT / 'run_summary.json'}")


if __name__ == "__main__":
    main()
