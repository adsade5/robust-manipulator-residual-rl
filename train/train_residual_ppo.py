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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.residual_tracking_env import ResidualTrackingEnv, print_environment_timing


RESULT_DIR = PROJECT_ROOT / "results" / "residual_ppo"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
TENSORBOARD_DIR = RESULT_DIR / "tensorboard"
EVAL_LOG = RESULT_DIR / "training_eval_log.json"
SEED = 7
TOTAL_TIMESTEPS = 30_000
EVAL_FREQ = 4_500
EVAL_SCALES = [1.25, 1.50, 1.75]
POLICY_KWARGS = {"net_arch": [128, 128]}
PPO_KWARGS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
}


def make_training_env() -> VecNormalize:
    def _make() -> Monitor:
        return Monitor(ResidualTrackingEnv(randomize_scale=True, seed=SEED))

    env = DummyVecEnv([_make])
    return VecNormalize(env, norm_obs=True, norm_reward=False, training=True)


def make_eval_env(scale: float, obs_rms: Any) -> VecNormalize:
    def _make() -> Monitor:
        return Monitor(ResidualTrackingEnv(fixed_inertial_scale=scale, randomize_scale=False, seed=SEED + int(scale * 100)))

    env = VecNormalize(DummyVecEnv([_make]), norm_obs=True, norm_reward=False, training=False)
    env.obs_rms = obs_rms
    return env


def run_eval_episode(model: PPO, scale: float, obs_rms: Any) -> dict[str, float]:
    env = make_eval_env(scale, obs_rms)
    obs = env.reset()
    done = np.array([False])
    episode_reward = 0.0
    metrics = None
    while not bool(done[0]):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        episode_reward += float(reward[0])
        if bool(done[0]):
            metrics = infos[0].get("episode_metrics")
    if metrics is None:
        raise RuntimeError("Evaluation episode ended without episode_metrics in info")
    env.close()
    return {
        "scale": float(scale),
        "reward": episode_reward,
        "overall_rmse": float(metrics["overall_rmse"]),
        "motion_rmse": float(metrics["motion_rmse"]),
        "residual_torque_rms": float(metrics["residual_torque_rms"]),
    }


class MotionRmseEvalCallback(BaseCallback):
    def __init__(self, eval_freq: int, verbose: int = 1) -> None:
        super().__init__(verbose=verbose)
        self.eval_freq = eval_freq
        self.best_motion_rmse = float("inf")
        self.best_timestep = 0
        self.records: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        assert isinstance(self.training_env, VecNormalize)
        evals = [run_eval_episode(self.model, scale, self.training_env.obs_rms) for scale in EVAL_SCALES]
        mean_motion_rmse = float(np.mean([item["motion_rmse"] for item in evals]))
        mean_overall_rmse = float(np.mean([item["overall_rmse"] for item in evals]))
        mean_reward = float(np.mean([item["reward"] for item in evals]))
        mean_residual_rms = float(np.mean([item["residual_torque_rms"] for item in evals]))
        record = {
            "timesteps": int(self.num_timesteps),
            "mean_motion_rmse": mean_motion_rmse,
            "mean_overall_rmse": mean_overall_rmse,
            "mean_reward": mean_reward,
            "mean_residual_torque_rms": mean_residual_rms,
            "per_scale": evals,
        }
        self.records.append(record)
        EVAL_LOG.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
        if self.verbose:
            print(
                f"eval step={self.num_timesteps} mean_motion_rmse={mean_motion_rmse:.8f} "
                f"mean_reward={mean_reward:.4f} residual_rms={mean_residual_rms:.6f}"
            )
        if mean_motion_rmse < self.best_motion_rmse:
            self.best_motion_rmse = mean_motion_rmse
            self.best_timestep = int(self.num_timesteps)
            self.model.save(CHECKPOINT_DIR / "best_model")
            assert isinstance(self.training_env, VecNormalize)
            self.training_env.save(CHECKPOINT_DIR / "vecnormalize.pkl")
        return True


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    probe_env = ResidualTrackingEnv(randomize_scale=True, seed=SEED)
    print_environment_timing(probe_env)
    probe_env.close()

    env = make_training_env()
    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=1,
        tensorboard_log=str(TENSORBOARD_DIR),
        policy_kwargs=POLICY_KWARGS,
        device="auto",
        **PPO_KWARGS,
    )
    callback = MotionRmseEvalCallback(eval_freq=EVAL_FREQ)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, tb_log_name="ppo_residual")
    model.save(CHECKPOINT_DIR / "final_model")
    env.save(CHECKPOINT_DIR / "vecnormalize.pkl")

    final_train_reward = None
    if model.ep_info_buffer:
        final_train_reward = float(np.mean([item["r"] for item in model.ep_info_buffer]))

    summary = {
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "ppo_hyperparameters": PPO_KWARGS,
        "policy_kwargs": POLICY_KWARGS,
        "eval_freq": EVAL_FREQ,
        "eval_scales": EVAL_SCALES,
        "best_checkpoint_timestep": callback.best_timestep,
        "best_evaluation_motion_rmse": callback.best_motion_rmse,
        "final_training_reward_mean_ep_info_buffer": final_train_reward,
    }
    (RESULT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"best checkpoint timestep = {callback.best_timestep}")
    print(f"best evaluation motion RMSE = {callback.best_motion_rmse:.8f}")
    print(f"final training reward = {final_train_reward}")
    print(f"saved final model = {CHECKPOINT_DIR / 'final_model.zip'}")
    print(f"saved best model = {CHECKPOINT_DIR / 'best_model.zip'}")
    print(f"saved vecnormalize = {CHECKPOINT_DIR / 'vecnormalize.pkl'}")
    env.close()


if __name__ == "__main__":
    main()
