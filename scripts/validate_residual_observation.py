from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.residual_tracking_env import ResidualTrackingEnv, print_environment_timing


def main() -> None:
    env = ResidualTrackingEnv(fixed_inertial_scale=1.50, randomize_scale=False, seed=1)
    obs, info = env.reset(seed=1)
    action = env.action_space.sample()

    print_environment_timing(env)
    print(f"observation shape = {obs.shape}")
    print(f"action shape = {action.shape}")
    print(f"observation finite = {np.all(np.isfinite(obs))}")
    print(f"action within [-1, 1] = {np.all(action >= -1.0) and np.all(action <= 1.0)}")
    print(f"inertial scale in info only = {info['inertial_scale']:.6f}")

    if obs.shape != (49,):
        raise RuntimeError(f"Observation shape mismatch: {obs.shape}")
    if action.shape != (7,):
        raise RuntimeError(f"Action shape mismatch: {action.shape}")
    if not np.all(np.isfinite(obs)):
        raise RuntimeError("Observation contains NaN or Inf")
    if not (np.all(action >= -1.0) and np.all(action <= 1.0)):
        raise RuntimeError("Sampled action is outside [-1, 1]")

    # The observation has only q/qdot/reference/error terms. The fixed scale
    # should not be directly present as a privileged scalar.
    if np.any(np.isclose(obs, info["inertial_scale"], atol=1e-8)):
        raise RuntimeError("Plant inertial scale appears to be present in observation")

    check_env(env, skip_render_check=True)
    print("gymnasium check_env passed.")
    print("PASS")


if __name__ == "__main__":
    main()
