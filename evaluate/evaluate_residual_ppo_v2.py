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


V1_DIR = PROJECT_ROOT / "results" / "residual_ppo_v1"
V2_DIR = PROJECT_ROOT / "results" / "residual_ppo_v2"
EVAL_DIR = V2_DIR / "evaluation"
V1_CHECKPOINT_DIR = V1_DIR / "checkpoints"
V2_CHECKPOINT_DIR = V2_DIR / "checkpoints"
SCALES = [1.00, 1.25, 1.50, 1.75, 2.00]


def make_vec_env(scale: float, vecnormalize_path: Path) -> VecNormalize:
    def _make() -> ResidualTrackingEnv:
        return ResidualTrackingEnv(fixed_inertial_scale=scale, randomize_scale=False, seed=1000 + int(scale * 100))

    env = DummyVecEnv([_make])
    vec = VecNormalize.load(vecnormalize_path, env)
    vec.training = False
    vec.norm_reward = False
    return vec


def run_ctc_only(scale: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = ResidualTrackingEnv(fixed_inertial_scale=scale, randomize_scale=False, seed=2000 + int(scale * 100))
    env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(np.zeros(7, dtype=np.float32))
    metrics = dict(env.last_episode_metrics)
    log = env.export_episode_log()
    env.close()
    return metrics, log


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
    if metrics is None or log is None:
        raise RuntimeError("Residual evaluation ended without episode metrics")
    env.close()
    return dict(metrics), log


def _safe(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def _save(scale: float, controller: str, metrics: dict[str, Any], log: dict[str, np.ndarray]) -> None:
    metrics = dict(metrics)
    metrics["scale"] = scale
    metrics["controller"] = controller
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    safe_scale = _safe(scale)
    (EVAL_DIR / f"scale_{safe_scale}_{controller}_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    np.savez(EVAL_DIR / f"scale_{safe_scale}_{controller}.npz", **log)


def main() -> None:
    v1_best = V1_CHECKPOINT_DIR / "best_model.zip"
    v2_best = V2_CHECKPOINT_DIR / "best_model.zip"
    v2_final = V2_CHECKPOINT_DIR / "final_model.zip"
    v2_best_vecnorm = V2_CHECKPOINT_DIR / "vecnormalize.pkl"
    v2_final_vecnorm = V2_CHECKPOINT_DIR / "vecnormalize_final.pkl"
    if not v1_best.exists():
        raise FileNotFoundError(f"Missing v1 best model: {v1_best}")
    v1_model = PPO.load(v1_best)
    v1_vecnormalize_path = V1_CHECKPOINT_DIR / "vecnormalize.pkl"
    if v2_best.exists():
        v2_model_path = v2_best
        v2_vecnormalize_path = v2_best_vecnorm
        v2_model_is_guard_eligible = True
    else:
        if not v2_final.exists() or not v2_final_vecnorm.exists():
            raise FileNotFoundError("Missing both v2 best model and final diagnostic model")
        v2_model_path = v2_final
        v2_vecnormalize_path = v2_final_vecnorm
        v2_model_is_guard_eligible = False
        print("WARNING: no nominal-guard-passing v2 best model exists; evaluating final model as diagnostic only.")
    v2_model = PPO.load(v2_model_path)

    comparison = []
    for scale in SCALES:
        ctc_metrics, ctc_log = run_ctc_only(scale)
        v1_metrics, v1_log = run_residual(v1_model, v1_vecnormalize_path, scale)
        v2_metrics, v2_log = run_residual(v2_model, v2_vecnormalize_path, scale)
        _save(scale, "ctc", ctc_metrics, ctc_log)
        _save(scale, "ppo_v1", v1_metrics, v1_log)
        _save(scale, "ppo_v2", v2_metrics, v2_log)
        v2_vs_ctc = v2_metrics["motion_rmse"] - ctc_metrics["motion_rmse"]
        v2_vs_v1 = v2_metrics["motion_rmse"] - v1_metrics["motion_rmse"]
        relative_change = 100.0 * v2_vs_ctc / ctc_metrics["motion_rmse"] if ctc_metrics["motion_rmse"] > 0 else 0.0
        row = {
            "scale": scale,
            "ctc": ctc_metrics,
            "ppo_v1": v1_metrics,
            "ppo_v2": v2_metrics,
            "ppo_v2_model_path": str(v2_model_path),
            "ppo_v2_model_is_guard_eligible": v2_model_is_guard_eligible,
            "v2_vs_ctc_motion_rmse_difference": v2_vs_ctc,
            "v2_vs_ctc_motion_rmse_relative_change_percent": relative_change,
            "v2_vs_v1_motion_rmse_difference": v2_vs_v1,
        }
        comparison.append(row)
        print(
            f"scale={scale:.2f} "
            f"CTC={ctc_metrics['motion_rmse']:.8f} "
            f"v1={v1_metrics['motion_rmse']:.8f} "
            f"v2={v2_metrics['motion_rmse']:.8f} "
            f"v2-CTC={v2_vs_ctc:+.8f} "
            f"v2-v1={v2_vs_v1:+.8f}"
        )

    (EVAL_DIR / "comparison_v1_v2.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (EVAL_DIR / "v2_model_selection.json").write_text(
        json.dumps(
            {
                "model_path": str(v2_model_path),
                "vecnormalize_path": str(v2_vecnormalize_path),
                "guard_eligible_best_model": v2_model_is_guard_eligible,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved comparison = {EVAL_DIR / 'comparison_v1_v2.json'}")


if __name__ == "__main__":
    main()
