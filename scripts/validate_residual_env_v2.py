from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.residual_tracking_env import ResidualTrackingEnv, print_environment_timing


RESULT_DIR = PROJECT_ROOT / "results" / "residual_ppo_v2"
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


def run_zero_action(scale: float) -> dict[str, object]:
    env = ResidualTrackingEnv(fixed_inertial_scale=scale, randomize_scale=False, seed=0)
    obs, info = env.reset()
    del obs, info
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(np.zeros(7, dtype=np.float32))
    metrics = dict(env.last_episode_metrics)
    env.close()
    return metrics


def compare(metrics: dict[str, object], baseline: dict[str, float]) -> dict[str, float | bool]:
    differences = {
        "overall_rmse_difference": abs(float(metrics["overall_rmse"]) - baseline["overall_rmse"]),
        "motion_rmse_difference": abs(float(metrics["motion_rmse"]) - baseline["motion_rmse"]),
        "max_error_difference": abs(float(metrics["max_tracking_error"]) - baseline["max_error"]),
        "max_torque_difference": abs(float(metrics["total_max_torque"]) - baseline["max_torque"]),
    }
    differences["pass"] = all(value <= TOL for value in differences.values())
    return differences


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    probe_env = ResidualTrackingEnv(fixed_inertial_scale=1.50, randomize_scale=False, seed=0)
    print_environment_timing(probe_env)
    probe_env.close()

    medium_metrics = run_zero_action(1.50)
    nominal_metrics = run_zero_action(1.00)
    medium_check = compare(medium_metrics, PHASE5_MEDIUM)
    nominal_check = compare(nominal_metrics, PHASE5_NOMINAL)
    output = {
        "scale_1p50_zero_action_metrics": medium_metrics,
        "scale_1p50_phase5_baseline": PHASE5_MEDIUM,
        "scale_1p50_check": medium_check,
        "scale_1p00_zero_action_metrics": nominal_metrics,
        "scale_1p00_phase5_baseline": PHASE5_NOMINAL,
        "scale_1p00_check": nominal_check,
        "pass": bool(medium_check["pass"] and nominal_check["pass"]),
    }
    (RESULT_DIR / "zero_action_validation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Residual Env v2 zero-action validation")
    print(
        "scale=1.50 "
        f"overall_diff={medium_check['overall_rmse_difference']:.12g} "
        f"motion_diff={medium_check['motion_rmse_difference']:.12g} "
        f"max_error_diff={medium_check['max_error_difference']:.12g} "
        f"max_torque_diff={medium_check['max_torque_difference']:.12g} "
        f"PASS={medium_check['pass']}"
    )
    print(
        "scale=1.00 "
        f"overall_diff={nominal_check['overall_rmse_difference']:.12g} "
        f"motion_diff={nominal_check['motion_rmse_difference']:.12g} "
        f"max_error_diff={nominal_check['max_error_difference']:.12g} "
        f"max_torque_diff={nominal_check['max_torque_difference']:.12g} "
        f"PASS={nominal_check['pass']}"
    )
    print(f"overall PASS={output['pass']}")
    if not output["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
