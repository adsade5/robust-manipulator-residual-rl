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
V3_DIR = PROJECT_ROOT / "results" / "residual_ppo_v3"
EVAL_DIR = V3_DIR / "evaluation"
SCALES = [1.00, 1.25, 1.50, 1.75, 2.00]
V3_ACTION_PENALTY_WEIGHT = 0.03


def _safe(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def choose_model(result_dir: Path, label: str) -> dict[str, object]:
    checkpoint_dir = result_dir / "checkpoints"
    best_model = checkpoint_dir / "best_model.zip"
    best_vecnorm = checkpoint_dir / "vecnormalize.pkl"
    final_model = checkpoint_dir / "final_model.zip"
    final_vecnorm = checkpoint_dir / "vecnormalize_final.pkl"
    if best_model.exists() and best_vecnorm.exists():
        return {
            "label": label,
            "model_path": best_model,
            "vecnormalize_path": best_vecnorm,
            "guard_eligible_best_model": True,
        }
    if final_model.exists() and final_vecnorm.exists():
        print(f"WARNING: no guard-passing {label} best model exists; evaluating final model as diagnostic only.")
        return {
            "label": label,
            "model_path": final_model,
            "vecnormalize_path": final_vecnorm,
            "guard_eligible_best_model": False,
        }
    raise FileNotFoundError(f"Missing usable model for {label} under {checkpoint_dir}")


def make_vec_env(scale: float, vecnormalize_path: Path, action_penalty_weight: float | None) -> VecNormalize:
    def _make() -> ResidualTrackingEnv:
        return ResidualTrackingEnv(
            fixed_inertial_scale=scale,
            randomize_scale=False,
            action_penalty_weight=action_penalty_weight,
            seed=1000 + int(scale * 100),
        )

    env = DummyVecEnv([_make])
    vec = VecNormalize.load(vecnormalize_path, env)
    vec.training = False
    vec.norm_reward = False
    return vec


def run_ctc_only(scale: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = ResidualTrackingEnv(
        fixed_inertial_scale=scale,
        randomize_scale=False,
        action_penalty_weight=V3_ACTION_PENALTY_WEIGHT,
        seed=2000 + int(scale * 100),
    )
    env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(np.zeros(7, dtype=np.float32))
    metrics = dict(env.last_episode_metrics)
    log = env.export_episode_log()
    env.close()
    return metrics, log


def run_residual(
    model: PPO,
    vecnormalize_path: Path,
    scale: float,
    action_penalty_weight: float | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = make_vec_env(scale, vecnormalize_path, action_penalty_weight)
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
    v1_selection = choose_model(V1_DIR, "ppo_v1")
    v2_selection = choose_model(V2_DIR, "ppo_v2")
    v3_selection = choose_model(V3_DIR, "ppo_v3")
    v1_model = PPO.load(v1_selection["model_path"])
    v2_model = PPO.load(v2_selection["model_path"])
    v3_model = PPO.load(v3_selection["model_path"])

    comparison = []
    for scale in SCALES:
        ctc_metrics, ctc_log = run_ctc_only(scale)
        v1_metrics, v1_log = run_residual(v1_model, v1_selection["vecnormalize_path"], scale, None)
        v2_metrics, v2_log = run_residual(v2_model, v2_selection["vecnormalize_path"], scale, None)
        v3_metrics, v3_log = run_residual(v3_model, v3_selection["vecnormalize_path"], scale, V3_ACTION_PENALTY_WEIGHT)
        _save(scale, "ctc", ctc_metrics, ctc_log)
        _save(scale, "ppo_v1", v1_metrics, v1_log)
        _save(scale, "ppo_v2", v2_metrics, v2_log)
        _save(scale, "ppo_v3", v3_metrics, v3_log)
        v3_vs_ctc = v3_metrics["motion_rmse"] - ctc_metrics["motion_rmse"]
        v3_vs_v2 = v3_metrics["motion_rmse"] - v2_metrics["motion_rmse"]
        relative_change = 100.0 * v3_vs_ctc / ctc_metrics["motion_rmse"] if ctc_metrics["motion_rmse"] > 0 else 0.0
        row = {
            "scale": scale,
            "ctc": ctc_metrics,
            "ppo_v1": v1_metrics,
            "ppo_v2": v2_metrics,
            "ppo_v3": v3_metrics,
            "ppo_v2_model_is_guard_eligible": v2_selection["guard_eligible_best_model"],
            "ppo_v3_model_is_guard_eligible": v3_selection["guard_eligible_best_model"],
            "v3_vs_ctc_motion_rmse_difference": v3_vs_ctc,
            "v3_vs_ctc_motion_rmse_relative_change_percent": relative_change,
            "v3_vs_v2_motion_rmse_difference": v3_vs_v2,
        }
        comparison.append(row)
        print(
            f"scale={scale:.2f} "
            f"CTC={ctc_metrics['motion_rmse']:.8f} "
            f"v1={v1_metrics['motion_rmse']:.8f} "
            f"v2={v2_metrics['motion_rmse']:.8f} "
            f"v3={v3_metrics['motion_rmse']:.8f} "
            f"v3-CTC={v3_vs_ctc:+.8f} "
            f"v3-v2={v3_vs_v2:+.8f}"
        )

    (EVAL_DIR / "comparison_v1_v2_v3.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (EVAL_DIR / "model_selection.json").write_text(
        json.dumps(
            {
                "ppo_v1": {key: str(value) if isinstance(value, Path) else value for key, value in v1_selection.items()},
                "ppo_v2": {key: str(value) if isinstance(value, Path) else value for key, value in v2_selection.items()},
                "ppo_v3": {key: str(value) if isinstance(value, Path) else value for key, value in v3_selection.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved comparison = {EVAL_DIR / 'comparison_v1_v2_v3.json'}")


if __name__ == "__main__":
    main()
