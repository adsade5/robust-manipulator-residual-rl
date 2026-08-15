from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.residual_tracking_env import ResidualTrackingEnv, print_environment_timing


RESULT_DIR = PROJECT_ROOT / "results" / "residual_ppo"
PHASE5_METRICS = PROJECT_ROOT / "results" / "model_mismatch" / "inertial_medium_metrics.json"


def run_zero_action(scale: float = 1.50) -> dict[str, object]:
    env = ResidualTrackingEnv(fixed_inertial_scale=scale, randomize_scale=False, seed=0)
    obs, info = env.reset(seed=0)
    print_environment_timing(env)
    if obs.shape != (49,):
        raise RuntimeError(f"Unexpected obs shape: {obs.shape}")
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
    return env.last_episode_metrics


def main() -> None:
    if not PHASE5_METRICS.exists():
        raise FileNotFoundError(f"Missing Phase 5 metrics: {PHASE5_METRICS}")
    phase5 = json.loads(PHASE5_METRICS.read_text(encoding="utf-8"))
    residual = run_zero_action(1.50)

    diffs = {
        "overall_rmse_difference": abs(residual["overall_rmse"] - phase5["overall_rmse"]),
        "motion_rmse_difference": abs(residual["motion_rmse"] - phase5["motion_rmse"]),
        "max_error_difference": abs(residual["max_tracking_error"] - phase5["overall_max_error"]),
        "max_torque_difference": abs(residual["total_max_torque"] - phase5["max_abs_tau"]),
    }
    passed = all(value < 1e-10 for value in diffs.values())
    result = {
        "residual_zero_action_metrics": residual,
        "phase5_inertial_medium": {
            "overall_rmse": phase5["overall_rmse"],
            "motion_rmse": phase5["motion_rmse"],
            "max_error": phase5["overall_max_error"],
            "max_torque": phase5["max_abs_tau"],
        },
        **diffs,
        "pass": passed,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / "zero_action_validation.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Residual Env zero-action vs Phase 5 CTC")
    print(f"zero-action overall RMSE = {residual['overall_rmse']:.12e}")
    print(f"Phase 5 medium overall RMSE = {phase5['overall_rmse']:.12e}")
    print(f"RMSE difference = {diffs['overall_rmse_difference']:.12e}")
    print(f"motion RMSE difference = {diffs['motion_rmse_difference']:.12e}")
    print(f"max error difference = {diffs['max_error_difference']:.12e}")
    print(f"max torque difference = {diffs['max_torque_difference']:.12e}")
    print(f"saved = {output_path}")
    print("PASS" if passed else "FAIL")
    if not passed:
        raise RuntimeError("Residual environment zero-action validation failed")


if __name__ == "__main__":
    main()
