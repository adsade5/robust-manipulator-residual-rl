from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.computed_torque import computed_torque_control
from experiments.computed_torque_tracking import KD_ACC, KP_ACC
from experiments.joint_trajectory_tracking import ARM_TAU_LIMITS, HOME_CTRL, HOME_Q, HOME_QPOS, TARGET_Q, desired_state
from robustness.dual_model import NominalDynamicsModel, arm_qpos_dof_addresses
from robustness.perturbations import (
    DISTURBANCE_END,
    DISTURBANCE_START,
    apply_ee_inertial_mismatch,
    apply_external_disturbance,
    apply_joint_damping_mismatch,
    disturbance_force_at_time,
)


MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
RESULT_DIR = PROJECT_ROOT / "results" / "model_mismatch"
TOTAL_DURATION = 9.0
MAX_ABS_QVEL = 80.0
MAX_ABS_QACC = 300.0
RECOVERY_THRESHOLD = 0.01
RECOVERY_HOLD = 0.2


def _init_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    data.qvel[:] = 0.0
    data.ctrl[: len(HOME_CTRL)] = HOME_CTRL
    mujoco.mj_forward(model, data)


def _check_stability(data: mujoco.MjData, arm_dofadr: np.ndarray, sim_time: float) -> None:
    arrays = [data.qpos, data.qvel, data.qacc, data.ctrl, data.qfrc_bias, data.qfrc_passive, data.qfrc_applied]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError(f"NaN or Inf detected at t={sim_time:.4f}s")
    max_qvel = float(np.max(np.abs(data.qvel[arm_dofadr])))
    if max_qvel > MAX_ABS_QVEL:
        raise RuntimeError(f"Simulation instability at t={sim_time:.4f}s: max |qvel|={max_qvel:.3f}")
    max_qacc = float(np.max(np.abs(data.qacc[arm_dofadr])))
    if max_qacc > MAX_ABS_QACC:
        raise RuntimeError(f"Simulation instability at t={sim_time:.4f}s: max |qacc|={max_qacc:.3f}")


def _scenario_name(scenario: str, level: str | None) -> str:
    if scenario in {"inertial", "damping"}:
        if level is None:
            raise ValueError(f"--level is required for scenario {scenario}")
        return f"{scenario}_{level}"
    return scenario


def _recovery_time(time_log: np.ndarray, tracking_error: np.ndarray) -> float | None:
    dt = float(np.median(np.diff(time_log)))
    needed = max(1, int(np.ceil(RECOVERY_HOLD / dt)))
    norm_err = np.linalg.norm(tracking_error, axis=1)
    start_idx = int(np.searchsorted(time_log, DISTURBANCE_END))
    ok = norm_err < RECOVERY_THRESHOLD
    for idx in range(start_idx, len(time_log) - needed + 1):
        if np.all(ok[idx : idx + needed]):
            return float(time_log[idx] - DISTURBANCE_END)
    return None


def compute_metrics(
    scenario_name: str,
    result: dict[str, np.ndarray],
    clip_counts: np.ndarray,
) -> dict[str, object]:
    error = result["position_error"]
    time_log = result["time"]
    motion_mask = ((time_log >= 1.0) & (time_log <= 4.0)) | ((time_log >= 5.0) & (time_log <= 8.0))
    per_joint_rmse = np.sqrt(np.mean(error**2, axis=0))
    per_joint_max_error = np.max(np.abs(error), axis=0)
    torque = result["tau_command"]
    metrics: dict[str, object] = {
        "scenario": scenario_name,
        "q_home": HOME_Q.tolist(),
        "q_target": TARGET_Q.tolist(),
        "total_duration": TOTAL_DURATION,
        "timestep": float(np.median(np.diff(time_log))),
        "torque_limits": ARM_TAU_LIMITS.tolist(),
        "trajectory_schedule": [0.0, 1.0, 4.0, 5.0, 8.0, 9.0],
        "kp_acc": KP_ACC.tolist(),
        "kd_acc": KD_ACC.tolist(),
        "overall_rmse": float(np.sqrt(np.mean(error**2))),
        "per_joint_rmse": per_joint_rmse.tolist(),
        "motion_rmse": float(np.sqrt(np.mean(error[motion_mask] ** 2))),
        "overall_max_error": float(np.max(np.abs(error))),
        "per_joint_max_error": per_joint_max_error.tolist(),
        "torque_rms": float(np.sqrt(np.mean(np.sum(torque**2, axis=1)))),
        "per_joint_torque_rms": np.sqrt(np.mean(torque**2, axis=0)).tolist(),
        "max_abs_tau": float(np.max(np.abs(torque))),
        "clip_counts": clip_counts.astype(int).tolist(),
        "torque_saturation": bool(np.any(clip_counts > 0)),
        "final_home_error": (result["q"][-1] - HOME_Q).tolist(),
        "unstable": False,
    }
    if scenario_name == "disturbance":
        pre_mask = (time_log >= DISTURBANCE_START - 0.5) & (time_log < DISTURBANCE_START)
        during_mask = (time_log >= DISTURBANCE_START) & (time_log < DISTURBANCE_END)
        recovery = _recovery_time(time_log, error)
        metrics.update(
            {
                "pre_disturbance_tracking_error": float(np.max(np.linalg.norm(error[pre_mask], axis=1))),
                "maximum_disturbance_error": float(np.max(np.linalg.norm(error[during_mask], axis=1))),
                "recovery_time": recovery,
                "recovered": recovery is not None,
                "recovery_threshold": RECOVERY_THRESHOLD,
                "recovery_hold": RECOVERY_HOLD,
                "disturbance_window": [DISTURBANCE_START, DISTURBANCE_END],
            }
        )
    return metrics


def run_dual_model_ctc(
    scenario: str,
    level: str | None = None,
    viewer: bool = False,
    save: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    scenario_name = _scenario_name(scenario, level)
    plant_model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    plant_data = mujoco.MjData(plant_model)
    nominal = NominalDynamicsModel.from_xml_path(MODEL_XML, plant_model=plant_model)
    arm_qposadr, arm_dofadr = arm_qpos_dof_addresses(plant_model)

    perturbation_info: dict[str, object] = {}
    if scenario == "inertial":
        assert level is not None
        info = apply_ee_inertial_mismatch(plant_model, plant_data, level)
        plant_data = mujoco.MjData(plant_model)
        perturbation_info = {
            "type": "end_effector_inertial_mismatch",
            "level": level,
            "scale": info.scale,
            "body_name": info.body_name,
            "original_mass": info.original_mass,
            "original_inertia": info.original_inertia.tolist(),
            "modified_mass": info.modified_mass,
            "modified_inertia": info.modified_inertia.tolist(),
        }
    elif scenario == "damping":
        assert level is not None
        info = apply_joint_damping_mismatch(plant_model, level)
        plant_data = mujoco.MjData(plant_model)
        perturbation_info = {
            "type": "joint_damping_mismatch",
            "level": level,
            "scale": info.scale,
            "original_damping": info.original_damping.tolist(),
            "modified_damping": info.modified_damping.tolist(),
        }
    elif scenario == "disturbance":
        perturbation_info = {
            "type": "unmodeled_external_disturbance",
            "force": [10.0, 0.0, 0.0],
            "window": [DISTURBANCE_START, DISTURBANCE_END],
        }
    elif scenario != "nominal":
        raise ValueError(f"Unknown scenario: {scenario}")

    _init_home(plant_model, plant_data)
    _init_home(nominal.model, nominal.data)

    logs: dict[str, list[np.ndarray | float | bool]] = {
        "time": [],
        "q_des": [],
        "qdot_des": [],
        "qddot_des": [],
        "q": [],
        "qdot": [],
        "position_error": [],
        "velocity_error": [],
        "a_cmd": [],
        "tau_base_unclipped": [],
        "tau_command": [],
        "nominal_qfrc_bias": [],
        "nominal_qfrc_passive": [],
        "plant_qfrc_bias": [],
        "plant_qfrc_passive": [],
        "external_force": [],
        "torque_clipping_flag": [],
    }
    clip_counts = np.zeros(7, dtype=int)

    def step_once() -> bool:
        plant_data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(plant_model, plant_data)
        external_force = disturbance_force_at_time(float(plant_data.time)) if scenario == "disturbance" else np.zeros(3)
        if np.any(external_force):
            apply_external_disturbance(plant_model, plant_data, force=external_force)

        q = plant_data.qpos[arm_qposadr].copy()
        qdot = plant_data.qvel[arm_dofadr].copy()
        q_des, qdot_des, qddot_des, _ = desired_state(float(plant_data.time), 1.0)

        nominal.sync_state_from_plant(plant_data)
        nominal.update_dynamics()
        output = computed_torque_control(
            model=nominal.model,
            data=nominal.data,
            arm_dofadr=nominal.arm_dofadr,
            q_des=q_des,
            qdot_des=qdot_des,
            qddot_des=qddot_des,
            kp_acc=KP_ACC,
            kd_acc=KD_ACC,
        )
        tau_unclipped = output.tau_unclipped
        tau = np.clip(tau_unclipped, -ARM_TAU_LIMITS, ARM_TAU_LIMITS)
        clipped = np.abs(tau - tau_unclipped) > 1e-9
        clip_counts[:] += clipped

        plant_data.ctrl[:7] = tau
        plant_data.ctrl[7] = HOME_CTRL[7]
        mujoco.mj_forward(plant_model, plant_data)

        logs["time"].append(float(plant_data.time))
        logs["q_des"].append(q_des.copy())
        logs["qdot_des"].append(qdot_des.copy())
        logs["qddot_des"].append(qddot_des.copy())
        logs["q"].append(q)
        logs["qdot"].append(qdot)
        logs["position_error"].append(q_des - q)
        logs["velocity_error"].append(qdot_des - qdot)
        logs["a_cmd"].append(output.a_cmd.copy())
        logs["tau_base_unclipped"].append(tau_unclipped.copy())
        logs["tau_command"].append(tau.copy())
        logs["nominal_qfrc_bias"].append(output.qfrc_bias.copy())
        logs["nominal_qfrc_passive"].append(output.qfrc_passive.copy())
        logs["plant_qfrc_bias"].append(plant_data.qfrc_bias[arm_dofadr].copy())
        logs["plant_qfrc_passive"].append(plant_data.qfrc_passive[arm_dofadr].copy())
        logs["external_force"].append(external_force.copy())
        logs["torque_clipping_flag"].append(bool(np.any(clipped)))

        _check_stability(plant_data, arm_dofadr, plant_data.time)
        mujoco.mj_step(plant_model, plant_data)
        return plant_data.time < TOTAL_DURATION

    if viewer:
        viewer_module = importlib.import_module("mujoco.viewer")
        with viewer_module.launch_passive(plant_model, plant_data) as handle:
            while handle.is_running() and plant_data.time < TOTAL_DURATION:
                wall_start = time.time()
                keep_running = step_once()
                handle.sync()
                sleep_time = plant_model.opt.timestep - (time.time() - wall_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if not keep_running:
                    break
    else:
        while plant_data.time < TOTAL_DURATION:
            step_once()

    result = {key: np.array(value) for key, value in logs.items()}
    result.update(
        {
            "q_home": HOME_Q,
            "q_target": TARGET_Q,
            "total_duration": np.array(TOTAL_DURATION),
            "timestep": np.array(plant_model.opt.timestep),
            "torque_limits": ARM_TAU_LIMITS,
            "trajectory_schedule": np.array([0.0, 1.0, 4.0, 5.0, 8.0, 9.0]),
            "scenario": np.array(scenario_name),
            "clip_counts": clip_counts,
        }
    )
    metrics = compute_metrics(scenario_name, result, clip_counts)
    metrics["perturbation"] = perturbation_info

    if save:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        npz_path = RESULT_DIR / f"{scenario_name}.npz"
        metrics_path = RESULT_DIR / f"{scenario_name}_metrics.json"
        np.savez(npz_path, **result)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"scenario = {scenario_name}")
        print(f"saved = {npz_path}")
        print(f"metrics = {metrics_path}")
        print(f"overall RMSE = {metrics['overall_rmse']:.8f}")
        print(f"motion RMSE = {metrics['motion_rmse']:.8f}")
        print(f"max error = {metrics['overall_max_error']:.8f}")
        print(f"torque RMS = {metrics['torque_rms']:.8f}")
        print(f"max |tau| = {metrics['max_abs_tau']:.8f}")
        print(f"clipping count = {sum(metrics['clip_counts'])}")
        if scenario_name == "disturbance":
            print(f"maximum disturbance error = {metrics['maximum_disturbance_error']:.8f}")
            print(f"recovery time = {metrics['recovery_time']}")
        print("NaN/Inf/unstable = False")
    return result, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-model CTC robustness benchmark.")
    parser.add_argument("--scenario", choices=["nominal", "inertial", "damping", "disturbance"], required=True)
    parser.add_argument("--level", choices=["mild", "medium", "strong"])
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.scenario in {"inertial", "damping"} and args.level is None:
        parser.error("--level is required for inertial and damping scenarios")
    if args.scenario in {"nominal", "disturbance"} and args.level is not None:
        parser.error("--level is only valid for inertial and damping scenarios")
    run_dual_model_ctc(args.scenario, args.level, viewer=args.viewer, save=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
