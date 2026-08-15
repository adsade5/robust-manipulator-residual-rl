from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.computed_torque import computed_torque_control
from experiments.computed_torque_tracking import KD_ACC, KP_ACC
from experiments.joint_trajectory_tracking import ARM_TAU_LIMITS, HOME_CTRL, HOME_Q, HOME_QPOS, TARGET_Q, desired_state
from robustness.dual_model import NominalDynamicsModel, arm_qpos_dof_addresses
from robustness.perturbations import set_ee_inertial_scale_from_nominal


MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
TOTAL_DURATION = 9.0
TARGET_POLICY_DT = 0.02
RESIDUAL_TORQUE_LIMIT = np.array([8.0, 8.0, 8.0, 8.0, 1.2, 1.2, 1.2], dtype=np.float64)
POSITION_ERROR_SCALE = 0.05
VELOCITY_ERROR_SCALE = 0.25
REWARD_WEIGHTS = {
    "position": 1.0,
    "velocity": 0.1,
    "action": 0.01,
    "smoothness": 0.005,
}
TRAIN_SCALE_LOW = 1.25
TRAIN_SCALE_HIGH = 1.75
MAX_ABS_QVEL = 80.0
JOINT_LIMIT_MARGIN = 0.25
FAILURE_REWARD = -100.0


class ResidualTrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        fixed_inertial_scale: float | None = None,
        randomize_scale: bool = True,
        nominal_sample_probability: float = 0.0,
        action_penalty_weight: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.fixed_inertial_scale = fixed_inertial_scale
        self.randomize_scale = randomize_scale
        if not 0.0 <= nominal_sample_probability <= 1.0:
            raise ValueError("nominal_sample_probability must be in [0, 1]")
        self.nominal_sample_probability = float(nominal_sample_probability)
        self.reward_weights = dict(REWARD_WEIGHTS)
        if action_penalty_weight is not None:
            self.reward_weights["action"] = float(action_penalty_weight)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.plant_model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.plant_data = mujoco.MjData(self.plant_model)
        self.nominal = NominalDynamicsModel.from_xml_path(MODEL_XML, plant_model=self.plant_model)
        self.nominal_model = self.nominal.model
        self.nominal_data = self.nominal.data
        self.arm_qposadr, self.arm_dofadr = arm_qpos_dof_addresses(self.plant_model)

        self.physics_dt = float(self.plant_model.opt.timestep)
        self.action_repeat = max(1, int(round(TARGET_POLICY_DT / self.physics_dt)))
        self.policy_dt = self.action_repeat * self.physics_dt
        self.policy_frequency = 1.0 / self.policy_dt
        self.max_policy_steps = int(math.ceil(TOTAL_DURATION / self.policy_dt))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(49,), dtype=np.float32)

        self.previous_action = np.zeros(7, dtype=np.float64)
        self.current_inertial_scale = 1.0
        self.current_sample_source = "nominal"
        self.sample_counts = {"nominal": 0, "mismatch": 0, "fixed": 0}
        self.policy_step_count = 0
        self.last_episode_metrics: dict[str, Any] = {}
        self._reset_logs()

    def _reset_logs(self) -> None:
        self.logs: dict[str, list[np.ndarray | float | bool | dict[str, float]]] = {
            "time": [],
            "inertial_scale": [],
            "sample_source": [],
            "q": [],
            "qdot": [],
            "q_des": [],
            "qdot_des": [],
            "qddot_des": [],
            "q_error": [],
            "qdot_error": [],
            "tau_ctc": [],
            "delta_tau_rl": [],
            "tau_total": [],
            "action": [],
            "reward_position": [],
            "reward_velocity": [],
            "reward_action": [],
            "reward_smoothness": [],
            "reward": [],
            "total_torque_clipping_flag": [],
            "residual_action_clipping_flag": [],
        }

    def _sample_scale(self) -> tuple[float, str]:
        if self.fixed_inertial_scale is not None:
            return float(self.fixed_inertial_scale), "fixed"
        if self.randomize_scale:
            if self.nominal_sample_probability > 0.0 and self.np_random.random() < self.nominal_sample_probability:
                return 1.0, "nominal"
            return float(self.np_random.uniform(TRAIN_SCALE_LOW, TRAIN_SCALE_HIGH)), "mismatch"
        return 1.0, "nominal"

    def _init_home(self) -> None:
        self.plant_data.qpos[: len(HOME_QPOS)] = HOME_QPOS
        self.plant_data.qvel[:] = 0.0
        self.plant_data.ctrl[: len(HOME_CTRL)] = HOME_CTRL
        mujoco.mj_forward(self.plant_model, self.plant_data)
        self.nominal_data.qpos[: len(HOME_QPOS)] = HOME_QPOS
        self.nominal_data.qvel[:] = 0.0
        self.nominal_data.ctrl[: len(HOME_CTRL)] = HOME_CTRL
        mujoco.mj_forward(self.nominal_model, self.nominal_data)

    def _get_desired(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_des, qdot_des, qddot_des, _ = desired_state(float(self.plant_data.time), 1.0)
        return q_des, qdot_des, qddot_des

    def _get_obs(self) -> np.ndarray:
        q = self.plant_data.qpos[self.arm_qposadr].copy()
        qdot = self.plant_data.qvel[self.arm_dofadr].copy()
        q_des, qdot_des, qddot_des = self._get_desired()
        q_error = q_des - q
        qdot_error = qdot_des - qdot
        obs = np.concatenate([q, qdot, q_des, qdot_des, qddot_des, q_error, qdot_error])
        return obs.astype(np.float32)

    def _check_failure(self) -> bool:
        arrays = [self.plant_data.qpos, self.plant_data.qvel, self.plant_data.ctrl, self.plant_data.qacc]
        if any(not np.all(np.isfinite(array)) for array in arrays):
            return True
        if float(np.max(np.abs(self.plant_data.qvel[self.arm_dofadr]))) > MAX_ABS_QVEL:
            return True
        joint_ids = [mujoco.mj_name2id(self.plant_model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
        for idx, joint_id in enumerate(joint_ids):
            low, high = self.plant_model.jnt_range[joint_id]
            q = self.plant_data.qpos[self.arm_qposadr[idx]]
            if q < low - JOINT_LIMIT_MARGIN or q > high + JOINT_LIMIT_MARGIN:
                return True
        return False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        options = options or {}
        if "fixed_inertial_scale" in options:
            self.fixed_inertial_scale = float(options["fixed_inertial_scale"])

        self.plant_model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.plant_data = mujoco.MjData(self.plant_model)
        self.nominal = NominalDynamicsModel.from_xml_path(MODEL_XML, plant_model=self.plant_model)
        self.nominal_model = self.nominal.model
        self.nominal_data = self.nominal.data
        self.arm_qposadr, self.arm_dofadr = arm_qpos_dof_addresses(self.plant_model)

        self.current_inertial_scale, self.current_sample_source = self._sample_scale()
        self.sample_counts[self.current_sample_source] = self.sample_counts.get(self.current_sample_source, 0) + 1
        set_ee_inertial_scale_from_nominal(
            self.plant_model,
            self.plant_data,
            self.nominal_model,
            self.current_inertial_scale,
        )
        self.plant_data = mujoco.MjData(self.plant_model)
        self._init_home()
        self.previous_action[:] = 0.0
        self.policy_step_count = 0
        self.last_episode_metrics = {}
        self._reset_logs()
        return self._get_obs(), {
            "inertial_scale": self.current_inertial_scale,
            "sample_source": self.current_sample_source,
        }

    def _ctc_torque(self, q_des: np.ndarray, qdot_des: np.ndarray, qddot_des: np.ndarray):
        self.nominal.sync_state_from_plant(self.plant_data)
        self.nominal.update_dynamics()
        return computed_torque_control(
            self.nominal_model,
            self.nominal_data,
            self.nominal.arm_dofadr,
            q_des,
            qdot_des,
            qddot_des,
            KP_ACC,
            KD_ACC,
        )

    def step(self, action: np.ndarray):
        raw_action = np.asarray(action, dtype=np.float64).reshape(7)
        clipped_action = np.clip(raw_action, -1.0, 1.0)
        residual_action_clipped = bool(np.any(np.abs(clipped_action - raw_action) > 1e-8))
        delta_tau_rl = clipped_action * RESIDUAL_TORQUE_LIMIT

        pos_terms = []
        vel_terms = []
        action_terms = []
        smooth_terms = []
        total_torque_clipped = False
        terminated = False

        for _ in range(self.action_repeat):
            mujoco.mj_forward(self.plant_model, self.plant_data)
            q = self.plant_data.qpos[self.arm_qposadr].copy()
            qdot = self.plant_data.qvel[self.arm_dofadr].copy()
            q_des, qdot_des, qddot_des = self._get_desired()
            q_error = q_des - q
            qdot_error = qdot_des - qdot
            ctc = self._ctc_torque(q_des, qdot_des, qddot_des)
            tau_unclipped = ctc.tau_unclipped + delta_tau_rl
            tau_total = np.clip(tau_unclipped, -ARM_TAU_LIMITS, ARM_TAU_LIMITS)
            total_torque_clipped = total_torque_clipped or bool(np.any(np.abs(tau_total - tau_unclipped) > 1e-9))

            p_q = float(np.mean((q_error / POSITION_ERROR_SCALE) ** 2))
            p_v = float(np.mean((qdot_error / VELOCITY_ERROR_SCALE) ** 2))
            p_action = float(np.mean(clipped_action**2))
            p_smooth = float(np.mean((clipped_action - self.previous_action) ** 2))
            reward = (
                -self.reward_weights["position"] * p_q
                -self.reward_weights["velocity"] * p_v
                -self.reward_weights["action"] * p_action
                -self.reward_weights["smoothness"] * p_smooth
            )

            self.plant_data.ctrl[:7] = tau_total
            self.plant_data.ctrl[7] = HOME_CTRL[7]

            self.logs["time"].append(float(self.plant_data.time))
            self.logs["inertial_scale"].append(float(self.current_inertial_scale))
            self.logs["sample_source"].append(self.current_sample_source)
            self.logs["q"].append(q)
            self.logs["qdot"].append(qdot)
            self.logs["q_des"].append(q_des.copy())
            self.logs["qdot_des"].append(qdot_des.copy())
            self.logs["qddot_des"].append(qddot_des.copy())
            self.logs["q_error"].append(q_error.copy())
            self.logs["qdot_error"].append(qdot_error.copy())
            self.logs["tau_ctc"].append(ctc.tau_unclipped.copy())
            self.logs["delta_tau_rl"].append(delta_tau_rl.copy())
            self.logs["tau_total"].append(tau_total.copy())
            self.logs["action"].append(clipped_action.copy())
            self.logs["reward_position"].append(-self.reward_weights["position"] * p_q)
            self.logs["reward_velocity"].append(-self.reward_weights["velocity"] * p_v)
            self.logs["reward_action"].append(-self.reward_weights["action"] * p_action)
            self.logs["reward_smoothness"].append(-self.reward_weights["smoothness"] * p_smooth)
            self.logs["reward"].append(float(reward))
            self.logs["total_torque_clipping_flag"].append(bool(np.any(np.abs(tau_total - tau_unclipped) > 1e-9)))
            self.logs["residual_action_clipping_flag"].append(residual_action_clipped)

            pos_terms.append(p_q)
            vel_terms.append(p_v)
            action_terms.append(p_action)
            smooth_terms.append(p_smooth)

            mujoco.mj_step(self.plant_model, self.plant_data)
            if self._check_failure():
                terminated = True
                break
            if self.plant_data.time >= TOTAL_DURATION:
                break

        self.previous_action = clipped_action.copy()
        self.policy_step_count += 1
        truncated = bool(self.plant_data.time >= TOTAL_DURATION and not terminated)
        avg_reward = (
            -self.reward_weights["position"] * float(np.mean(pos_terms))
            -self.reward_weights["velocity"] * float(np.mean(vel_terms))
            -self.reward_weights["action"] * float(np.mean(action_terms))
            -self.reward_weights["smoothness"] * float(np.mean(smooth_terms))
        )
        if terminated:
            avg_reward += FAILURE_REWARD

        if terminated or truncated:
            self.last_episode_metrics = self.compute_episode_metrics()

        info = {
            "inertial_scale": self.current_inertial_scale,
            "sample_source": self.current_sample_source,
            "reward_position": -self.reward_weights["position"] * float(np.mean(pos_terms)),
            "reward_velocity": -self.reward_weights["velocity"] * float(np.mean(vel_terms)),
            "reward_action": -self.reward_weights["action"] * float(np.mean(action_terms)),
            "reward_smoothness": -self.reward_weights["smoothness"] * float(np.mean(smooth_terms)),
            "total_torque_clipped": total_torque_clipped,
            "residual_action_clipped": residual_action_clipped,
            "episode_metrics": self.last_episode_metrics if (terminated or truncated) else None,
            "episode_log": self.export_episode_log() if (terminated or truncated) else None,
        }
        return self._get_obs(), float(avg_reward), terminated, truncated, info

    def compute_episode_metrics(self) -> dict[str, Any]:
        if not self.logs["time"]:
            return {}
        time_log = np.asarray(self.logs["time"], dtype=float)
        q_error = np.asarray(self.logs["q_error"], dtype=float)
        tau_total = np.asarray(self.logs["tau_total"], dtype=float)
        delta_tau = np.asarray(self.logs["delta_tau_rl"], dtype=float)
        actions = np.asarray(self.logs["action"], dtype=float)
        motion_mask = ((time_log >= 1.0) & (time_log <= 4.0)) | ((time_log >= 5.0) & (time_log <= 8.0))
        action_diff = np.diff(actions, axis=0) if len(actions) > 1 else np.zeros_like(actions)
        return {
            "inertial_scale": float(self.current_inertial_scale),
            "sample_source": self.current_sample_source,
            "overall_rmse": float(np.sqrt(np.mean(q_error**2))),
            "motion_rmse": float(np.sqrt(np.mean(q_error[motion_mask] ** 2))),
            "per_joint_rmse": np.sqrt(np.mean(q_error**2, axis=0)).tolist(),
            "max_tracking_error": float(np.max(np.abs(q_error))),
            "torque_rms": float(np.sqrt(np.mean(np.sum(tau_total**2, axis=1)))),
            "residual_torque_rms": float(np.sqrt(np.mean(np.sum(delta_tau**2, axis=1)))),
            "max_residual_torque": float(np.max(np.abs(delta_tau))),
            "action_rms": float(np.sqrt(np.mean(actions**2))),
            "total_max_torque": float(np.max(np.abs(tau_total))),
            "action_smoothness_rms": float(np.sqrt(np.mean(action_diff**2))) if action_diff.size else 0.0,
            "total_torque_clipping_count": int(np.sum(self.logs["total_torque_clipping_flag"])),
            "residual_action_clipping_count": int(np.sum(self.logs["residual_action_clipping_flag"])),
            "total_reward": float(np.sum(self.logs["reward"])),
            "mean_reward": float(np.mean(self.logs["reward"])),
        }

    def export_episode_log(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(value) for key, value in self.logs.items()}


def print_environment_timing(env: ResidualTrackingEnv) -> None:
    print(f"physics dt = {env.physics_dt:.8f} s")
    print(f"action_repeat = {env.action_repeat}")
    print(f"actual policy dt = {env.policy_dt:.8f} s")
    print(f"actual policy frequency = {env.policy_frequency:.4f} Hz")
