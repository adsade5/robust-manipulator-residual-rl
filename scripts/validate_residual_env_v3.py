from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.residual_tracking_env import ResidualTrackingEnv, print_environment_timing


RESULT_DIR = PROJECT_ROOT / "results" / "residual_ppo_v3"
ACTION_PENALTY_WEIGHT = 0.03
PHASE5_MEDIUM = {
    "overall_rmse": 0.025917733700213982,
    "motion_rmse": 0.026674150644443333,
    "max_error": 0.05868046973970009,
    "max_torque": 31.711473271254405,
}
PHASE5_NOMINAL = {
    "overall_rmse": 2.3894461172206165e-05,
    "motion_rmse": 2.7923387438357618e-05,
    "max_error": 5.411024484985871e-05,
    "max_torque": 29.484790281506562,
}
TOL = 1e-12


def run_episode(scale: float, policy: str, seed: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = ResidualTrackingEnv(
        fixed_inertial_scale=scale,
        randomize_scale=False,
        action_penalty_weight=ACTION_PENALTY_WEIGHT,
        seed=seed,
    )
    env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        if policy == "zero":
            action = np.zeros(7, dtype=np.float32)
        elif policy == "random":
            action = env.action_space.sample()
        else:
            raise ValueError(policy)
        _, _, terminated, truncated, _ = env.step(action)
    metrics = dict(env.last_episode_metrics)
    log = env.export_episode_log()
    env.close()
    return metrics, log


def compare(metrics: dict[str, Any], baseline: dict[str, float]) -> dict[str, float | bool]:
    differences = {
        "overall_rmse_difference": abs(float(metrics["overall_rmse"]) - baseline["overall_rmse"]),
        "motion_rmse_difference": abs(float(metrics["motion_rmse"]) - baseline["motion_rmse"]),
        "max_error_difference": abs(float(metrics["max_tracking_error"]) - baseline["max_error"]),
        "max_torque_difference": abs(float(metrics["total_max_torque"]) - baseline["max_torque"]),
    }
    differences["pass"] = all(value <= TOL for value in differences.values())
    return differences


def reward_stats(log: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "reward_position_mean": float(np.mean(log["reward_position"])),
        "reward_velocity_mean": float(np.mean(log["reward_velocity"])),
        "reward_action_mean": float(np.mean(log["reward_action"])),
        "reward_smoothness_mean": float(np.mean(log["reward_smoothness"])),
        "total_reward_mean": float(np.mean(log["reward"])),
        "total_reward_sum": float(np.sum(log["reward"])),
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    probe_env = ResidualTrackingEnv(
        fixed_inertial_scale=1.50,
        randomize_scale=False,
        action_penalty_weight=ACTION_PENALTY_WEIGHT,
        seed=0,
    )
    print_environment_timing(probe_env)
    probe_env.close()

    medium_metrics, medium_log = run_episode(1.50, "zero", seed=0)
    nominal_metrics, nominal_log = run_episode(1.00, "zero", seed=0)
    medium_check = compare(medium_metrics, PHASE5_MEDIUM)
    nominal_check = compare(nominal_metrics, PHASE5_NOMINAL)

    sanity: dict[str, dict[str, float]] = {
        "scale_1p00_zero": reward_stats(nominal_log),
        "scale_1p50_zero": reward_stats(medium_log),
    }
    for scale in [1.00, 1.50]:
        _, random_log = run_episode(scale, "random", seed=100 + int(scale * 100))
        sanity[f"scale_{scale:.2f}".replace(".", "p") + "_random"] = reward_stats(random_log)

    output = {
        "action_penalty_weight": ACTION_PENALTY_WEIGHT,
        "scale_1p50_zero_action_metrics": medium_metrics,
        "scale_1p50_phase5_baseline": PHASE5_MEDIUM,
        "scale_1p50_check": medium_check,
        "scale_1p00_zero_action_metrics": nominal_metrics,
        "scale_1p00_phase5_baseline": PHASE5_NOMINAL,
        "scale_1p00_check": nominal_check,
        "reward_sanity": sanity,
        "pass": bool(medium_check["pass"] and nominal_check["pass"]),
    }
    (RESULT_DIR / "zero_action_and_reward_validation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Residual Env v3 validation")
    print(
        "scale=1.00 "
        f"overall_diff={nominal_check['overall_rmse_difference']:.12g} "
        f"motion_diff={nominal_check['motion_rmse_difference']:.12g} "
        f"max_error_diff={nominal_check['max_error_difference']:.12g} "
        f"max_torque_diff={nominal_check['max_torque_difference']:.12g} "
        f"PASS={nominal_check['pass']}"
    )
    print(
        "scale=1.50 "
        f"overall_diff={medium_check['overall_rmse_difference']:.12g} "
        f"motion_diff={medium_check['motion_rmse_difference']:.12g} "
        f"max_error_diff={medium_check['max_error_difference']:.12g} "
        f"max_torque_diff={medium_check['max_torque_difference']:.12g} "
        f"PASS={medium_check['pass']}"
    )
    print("Reward sanity means")
    for name, stats in sanity.items():
        print(
            f"{name}: position={stats['reward_position_mean']:.6f} "
            f"velocity={stats['reward_velocity_mean']:.6f} "
            f"action={stats['reward_action_mean']:.6f} "
            f"smooth={stats['reward_smoothness_mean']:.6f} "
            f"total={stats['total_reward_mean']:.6f}"
        )
    print(f"overall PASS={output['pass']}")
    if not output["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
