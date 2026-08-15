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

from trajectories.quintic import quintic_joint_trajectory


MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
RESULT_DIR = PROJECT_ROOT / "results" / "joint_tracking"

HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])
HOME_CTRL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0])
DELTA_Q = np.array([0.25, -0.20, 0.20, -0.20, 0.15, 0.20, 0.15])
TARGET_Q = HOME_Q + DELTA_Q

KP = np.array([60.0, 80.0, 55.0, 45.0, 18.0, 14.0, 8.0])
KD = np.array([12.0, 16.0, 10.0, 9.0, 4.0, 3.0, 2.0])
ARM_TAU_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
JOINT_LIMIT_MARGIN = 0.05
MAX_ABS_QVEL = 80.0


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    return mujoco.mj_id2name(model, obj_type, obj_id) or ""


def _arm_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = []
    qposadr = []
    dofadr = []
    for index in range(1, 8):
        name = f"joint{index}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Missing joint: {name}")
        joint_ids.append(joint_id)
        qposadr.append(int(model.jnt_qposadr[joint_id]))
        dofadr.append(int(model.jnt_dofadr[joint_id]))
    return np.array(joint_ids, dtype=int), np.array(qposadr, dtype=int), np.array(dofadr, dtype=int)


def _validate_torque_actuators(model: mujoco.MjModel) -> None:
    names = [_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    if names[:7] != [f"torque{i}" for i in range(1, 8)]:
        raise RuntimeError(f"First 7 actuators are not torque1~torque7: {names[:7]}")
    for idx in range(7):
        if int(model.actuator_ctrllimited[idx]) != 1:
            raise RuntimeError(f"{names[idx]} is not control-limited")
        if int(model.actuator_forcelimited[idx]) != 1:
            raise RuntimeError(f"{names[idx]} is not force-limited")
        if not np.allclose(model.actuator_ctrlrange[idx], [-ARM_TAU_LIMITS[idx], ARM_TAU_LIMITS[idx]]):
            raise RuntimeError(f"{names[idx]} ctrlrange mismatch: {model.actuator_ctrlrange[idx]}")
        if not np.allclose(model.actuator_forcerange[idx], [-ARM_TAU_LIMITS[idx], ARM_TAU_LIMITS[idx]]):
            raise RuntimeError(f"{names[idx]} forcerange mismatch: {model.actuator_forcerange[idx]}")
        if not np.allclose(model.actuator_biasprm[idx], 0.0):
            raise RuntimeError(f"{names[idx]} still has nonzero servo bias: {model.actuator_biasprm[idx]}")


def _validate_target_with_limits(model: mujoco.MjModel, joint_ids: np.ndarray) -> None:
    for local_idx, joint_id in enumerate(joint_ids):
        joint_name = f"joint{local_idx + 1}"
        if int(model.jnt_limited[joint_id]) != 1:
            raise RuntimeError(f"{joint_name} has no joint range limit in the model")
        low, high = model.jnt_range[joint_id]
        q_target = TARGET_Q[local_idx]
        if not (low + JOINT_LIMIT_MARGIN <= q_target <= high - JOINT_LIMIT_MARGIN):
            raise RuntimeError(
                f"{joint_name} target {q_target:.6f} violates range "
                f"[{low:.6f}, {high:.6f}] with margin {JOINT_LIMIT_MARGIN:.3f}"
            )


def _initialise_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    data.qvel[:] = 0.0
    data.ctrl[: len(HOME_CTRL)] = HOME_CTRL
    mujoco.mj_forward(model, data)


def desired_state(t: float, duration_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    scale = duration_scale
    t1 = 1.0 * scale
    t2 = 4.0 * scale
    t3 = 5.0 * scale
    t4 = 8.0 * scale
    t5 = 9.0 * scale
    move_duration = 3.0 * scale

    if t <= t1:
        return HOME_Q.copy(), np.zeros(7), np.zeros(7), "hold_home_start"
    if t <= t2:
        q, qdot, qddot = quintic_joint_trajectory(HOME_Q, TARGET_Q, move_duration, t - t1)
        return q, qdot, qddot, "home_to_target"
    if t <= t3:
        return TARGET_Q.copy(), np.zeros(7), np.zeros(7), "hold_target"
    if t <= t4:
        q, qdot, qddot = quintic_joint_trajectory(TARGET_Q, HOME_Q, move_duration, t - t3)
        return q, qdot, qddot, "target_to_home"
    if t <= t5:
        return HOME_Q.copy(), np.zeros(7), np.zeros(7), "hold_home_end"
    return HOME_Q.copy(), np.zeros(7), np.zeros(7), "done"


def _compute_tau(
    controller: str,
    q_des: np.ndarray,
    qdot_des: np.ndarray,
    q: np.ndarray,
    qdot: np.ndarray,
    qfrc_bias_arm: np.ndarray,
) -> np.ndarray:
    tau = KP * (q_des - q) + KD * (qdot_des - qdot)
    if controller == "pd_gc":
        tau = tau + qfrc_bias_arm
    elif controller != "pd":
        raise ValueError(f"Unsupported controller: {controller}")
    return tau


def _check_stability(data: mujoco.MjData, qdot: np.ndarray, sim_time: float) -> None:
    arrays = [data.qpos, data.qvel, data.ctrl, data.qfrc_bias]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError(f"NaN or Inf detected at t={sim_time:.4f}s")
    max_abs_qvel = float(np.max(np.abs(qdot)))
    if max_abs_qvel > MAX_ABS_QVEL:
        raise RuntimeError(
            f"Simulation instability at t={sim_time:.4f}s: max |arm qvel|={max_abs_qvel:.3f}"
        )


def _metrics(
    controller: str,
    q: np.ndarray,
    tracking_error: np.ndarray,
    qdot: np.ndarray,
    tau: np.ndarray,
    clip_counts: np.ndarray,
    unstable: bool,
    duration_scale: float,
    total_duration: float,
    timestep: float,
) -> dict[str, object]:
    per_joint_rmse = np.sqrt(np.mean(tracking_error**2, axis=0))
    overall_rmse = float(np.sqrt(np.mean(tracking_error**2)))
    per_joint_max_abs_error = np.max(np.abs(tracking_error), axis=0)
    overall_max_abs_error = float(np.max(np.abs(tracking_error)))
    final_home_error = q[-1] - HOME_Q
    return {
        "controller": controller,
        "q_home": HOME_Q.tolist(),
        "q_target": TARGET_Q.tolist(),
        "duration_scale": float(duration_scale),
        "total_duration": float(total_duration),
        "timestep": float(timestep),
        "torque_limits": ARM_TAU_LIMITS.tolist(),
        "trajectory_schedule": {
            "hold_home_start": [0.0, 1.0 * duration_scale],
            "home_to_target": [1.0 * duration_scale, 4.0 * duration_scale],
            "hold_target": [4.0 * duration_scale, 5.0 * duration_scale],
            "target_to_home": [5.0 * duration_scale, 8.0 * duration_scale],
            "hold_home_end": [8.0 * duration_scale, 9.0 * duration_scale],
        },
        "kp": KP.tolist(),
        "kd": KD.tolist(),
        "per_joint_rmse": per_joint_rmse.tolist(),
        "overall_rmse": overall_rmse,
        "per_joint_max_abs_error": per_joint_max_abs_error.tolist(),
        "overall_max_abs_error": overall_max_abs_error,
        "final_home_error": final_home_error.tolist(),
        "max_abs_qvel": float(np.max(np.abs(qdot))),
        "max_abs_tau": float(np.max(np.abs(tau))),
        "clip_counts": clip_counts.astype(int).tolist(),
        "torque_saturation": bool(np.any(clip_counts > 0)),
        "unstable": bool(unstable),
    }


def run(controller: str, duration_scale: float, viewer: bool) -> dict[str, object]:
    if duration_scale <= 0.0 or not np.isfinite(duration_scale):
        raise ValueError("--duration-scale must be positive and finite")
    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Torque model XML not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    _validate_torque_actuators(model)
    joint_ids, arm_qposadr, arm_dofadr = _arm_addresses(model)
    _validate_target_with_limits(model, joint_ids)
    _initialise_home(model, data)

    total_duration = 9.0 * duration_scale
    times = []
    q_des_log = []
    qdot_des_log = []
    qddot_des_log = []
    q_log = []
    qdot_log = []
    err_log = []
    tau_log = []
    bias_log = []
    phase_log = []
    clip_counts = np.zeros(7, dtype=int)

    def step_once() -> bool:
        mujoco.mj_forward(model, data)
        q = data.qpos[arm_qposadr].copy()
        qdot = data.qvel[arm_dofadr].copy()
        qfrc_bias_arm = data.qfrc_bias[arm_dofadr].copy()
        q_des, qdot_des, qddot_des, phase = desired_state(float(data.time), duration_scale)

        tracking_error = q_des - q
        tau_raw = _compute_tau(controller, q_des, qdot_des, q, qdot, qfrc_bias_arm)
        tau = np.clip(tau_raw, -ARM_TAU_LIMITS, ARM_TAU_LIMITS)
        clip_counts[:] += np.abs(tau - tau_raw) > 1e-9

        data.ctrl[:7] = tau
        data.ctrl[7] = HOME_CTRL[7]

        times.append(float(data.time))
        q_des_log.append(q_des)
        qdot_des_log.append(qdot_des)
        qddot_des_log.append(qddot_des)
        q_log.append(q)
        qdot_log.append(qdot)
        err_log.append(tracking_error)
        tau_log.append(tau.copy())
        bias_log.append(qfrc_bias_arm)
        phase_log.append(phase)

        _check_stability(data, qdot, data.time)
        mujoco.mj_step(model, data)
        return data.time < total_duration

    if viewer:
        viewer_module = importlib.import_module("mujoco.viewer")
        with viewer_module.launch_passive(model, data) as handle:
            while handle.is_running() and data.time < total_duration:
                wall_start = time.time()
                keep_running = step_once()
                handle.sync()
                elapsed = time.time() - wall_start
                sleep_time = model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if not keep_running:
                    break
    else:
        while data.time < total_duration:
            step_once()

    result = {
        "time": np.array(times),
        "q_des": np.vstack(q_des_log),
        "qdot_des": np.vstack(qdot_des_log),
        "qddot_des": np.vstack(qddot_des_log),
        "q": np.vstack(q_log),
        "qdot": np.vstack(qdot_log),
        "tracking_error": np.vstack(err_log),
        "tau_command": np.vstack(tau_log),
        "qfrc_bias": np.vstack(bias_log),
        "trajectory_phase": np.array(phase_log),
        "q_target": TARGET_Q,
        "q_home": HOME_Q,
        "duration_scale": np.array(duration_scale),
        "total_duration": np.array(total_duration),
        "timestep": np.array(model.opt.timestep),
        "torque_limits": ARM_TAU_LIMITS,
        "trajectory_schedule": np.array([0.0, 1.0, 4.0, 5.0, 8.0, 9.0]) * duration_scale,
        "kp": KP,
        "kd": KD,
        "clip_counts": clip_counts,
    }
    metrics = _metrics(
        controller,
        result["q"],
        result["tracking_error"],
        result["qdot"],
        result["tau_command"],
        clip_counts,
        unstable=False,
        duration_scale=duration_scale,
        total_duration=total_duration,
        timestep=float(model.opt.timestep),
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = RESULT_DIR / f"{controller}.npz"
    metrics_path = RESULT_DIR / f"{controller}_metrics.json"
    np.savez(npz_path, **result)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"controller = {controller}")
    print(f"saved = {npz_path}")
    print(f"metrics = {metrics_path}")
    print(f"q_target = {np.array2string(TARGET_Q, precision=6)}")
    print(f"overall RMSE = {metrics['overall_rmse']:.8f}")
    print(f"per-joint RMSE = {np.array2string(np.array(metrics['per_joint_rmse']), precision=8)}")
    print(f"overall max abs error = {metrics['overall_max_abs_error']:.8f}")
    print(f"final home error = {np.array2string(np.array(metrics['final_home_error']), precision=8)}")
    print(f"max |qvel| = {metrics['max_abs_qvel']:.8f}")
    print(f"max |tau| = {metrics['max_abs_tau']:.8f}")
    print(f"torque clipping counts = {metrics['clip_counts']}")
    print("NaN/Inf/unstable = False")
    if metrics["torque_saturation"]:
        print("[WARNING] Torque clipping occurred. Do not increase torque limits to hide this.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint-space trajectory tracking on Panda torque model.")
    parser.add_argument("--controller", choices=["pd", "pd_gc"], required=True)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration-scale", type=float, default=1.0)
    args = parser.parse_args()
    run(args.controller, args.duration_scale, args.viewer)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
