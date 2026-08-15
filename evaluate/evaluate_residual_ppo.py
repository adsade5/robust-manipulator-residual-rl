from __future__ import annotations

import json
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


RESULT_DIR = PROJECT_ROOT / "results" / "residual_ppo"
EVAL_DIR = RESULT_DIR / "evaluation"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
SCALES = [1.00, 1.25, 1.50, 1.75, 2.00]


def make_vec_env(scale: float) -> VecNormalize:
    def _make() -> ResidualTrackingEnv:
        return ResidualTrackingEnv(fixed_inertial_scale=scale, randomize_scale=False, seed=1000 + int(scale * 100))

    env = DummyVecEnv([_make])
    vec = VecNormalize.load(CHECKPOINT_DIR / "vecnormalize.pkl", env)
    vec.training = False
    vec.norm_reward = False
    return vec


def run_ctc_only(scale: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = ResidualTrackingEnv(fixed_inertial_scale=scale, randomize_scale=False, seed=2000 + int(scale * 100))
    obs, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
    metrics = dict(env.last_episode_metrics)
    log = env.export_episode_log()
    env.close()
    return metrics, log


def run_residual(model: PPO, scale: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = make_vec_env(scale)
    obs = env.reset()
    done = np.array([False])
    metrics = None
    log = None
    while not bool(done[0]):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        if bool(done[0]):
            metrics = infos[0].get("episode_metrics")
            log = infos[0].get("episode_log")
    if metrics is None or log is None:
        raise RuntimeError("Residual evaluation ended without episode metrics")
    metrics = dict(metrics)
    env.close()
    return metrics, log


def _save(scale: float, controller: str, metrics: dict[str, Any], log: dict[str, np.ndarray]) -> None:
    safe_scale = f"{scale:.2f}".replace(".", "p")
    metrics = dict(metrics)
    metrics["scale"] = scale
    metrics["controller"] = controller
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"scale_{safe_scale}_{controller}_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    np.savez(EVAL_DIR / f"scale_{safe_scale}_{controller}.npz", **log)


def main() -> None:
    best_path = CHECKPOINT_DIR / "best_model.zip"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best model: {best_path}")
    model = PPO.load(best_path)

    comparison = []
    for scale in SCALES:
        ctc_metrics, ctc_log = run_ctc_only(scale)
        rl_metrics, rl_log = run_residual(model, scale)
        _save(scale, "ctc", ctc_metrics, ctc_log)
        _save(scale, "rl", rl_metrics, rl_log)
        absolute_change = ctc_metrics["motion_rmse"] - rl_metrics["motion_rmse"]
        relative_improvement = 100.0 * absolute_change / ctc_metrics["motion_rmse"] if ctc_metrics["motion_rmse"] > 0 else 0.0
        row = {
            "scale": scale,
            "ctc": ctc_metrics,
            "rl": rl_metrics,
            "motion_rmse_absolute_improvement": absolute_change,
            "motion_rmse_relative_improvement_percent": relative_improvement,
        }
        comparison.append(row)
        print(
            f"scale={scale:.2f} CTC motion={ctc_metrics['motion_rmse']:.8f} "
            f"RL motion={rl_metrics['motion_rmse']:.8f} "
            f"improvement={absolute_change:.8f} ({relative_improvement:.2f}%)"
        )

    (EVAL_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"saved comparison = {EVAL_DIR / 'comparison.json'}")


if __name__ == "__main__":
    main()
