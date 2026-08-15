from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.residual_tracking_env import ResidualTrackingEnv


RESULT_DIR = PROJECT_ROOT / "results" / "residual_ppo"


def main() -> None:
    env = ResidualTrackingEnv(randomize_scale=True, seed=42)
    results = []
    for episode in range(5):
        obs, info = env.reset(seed=42 + episode)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
            total_reward += reward
        metrics = dict(env.last_episode_metrics)
        metrics.update(
            {
                "episode": episode,
                "terminated": terminated,
                "truncated": truncated,
                "rollout_reward": total_reward,
            }
        )
        results.append(metrics)
        print(
            f"episode={episode} scale={metrics['inertial_scale']:.4f} "
            f"RMSE={metrics['overall_rmse']:.6f} max_torque={metrics['total_max_torque']:.6f} "
            f"residual_RMS={metrics['residual_torque_rms']:.6f} "
            f"clip_count={metrics['total_torque_clipping_count']} "
            f"terminated={terminated} truncated={truncated}"
        )
        if terminated:
            raise RuntimeError("Random residual policy caused termination/instability")

    # Reset to the same fixed scale twice to verify perturbation does not accumulate.
    env.fixed_inertial_scale = 1.50
    env.randomize_scale = False
    _, info1 = env.reset(seed=100)
    mass1 = float(env.plant_model.body_mass[9])
    _, info2 = env.reset(seed=101)
    mass2 = float(env.plant_model.body_mass[9])
    if abs(mass1 - mass2) > 1e-12:
        raise RuntimeError("Inertial scale appears to accumulate across resets")
    if abs(float(env.nominal_model.body_mass[9]) - 0.73) > 1e-12:
        raise RuntimeError("Nominal model was perturbed")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / "random_policy_safety.json"
    output_path.write_text(json.dumps({"episodes": results, "reset_mass_check": [mass1, mass2]}, indent=2), encoding="utf-8")
    print(f"saved = {output_path}")
    print("PASS")


if __name__ == "__main__":
    main()
