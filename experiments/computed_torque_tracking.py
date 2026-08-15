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
from experiments.joint_trajectory_tracking import (
    ARM_TAU_LIMITS,
    DELTA_Q,
    HOME_CTRL,
    HOME_Q,
    HOME_QPOS,
    JOINT_LIMIT_MARGIN,
    TARGET_Q,
    desired_state,
)


MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
RESULT_DIR = PROJECT_ROOT / "results" / "computed_torque"

ZETA = 1.0
WN = np.array([5.5, 5.5, 5.0, 5.0, 4.5, 4.5, 4.0])
KP_ACC = WN**2
KD_ACC = 2.0 * ZETA * WN
MAX_ABS_QVEL = 80.0
MAX_ABS_QACC = 300.0


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
        expected = [-ARM_TAU_LIMITS[idx], ARM_TAU_LIMITS[idx]]
        if int(model.actuator_ctrllimited[idx]) != 1:
            raise RuntimeError(f"{names[idx]} is not control-limited")
        if int(model.actuator_forcelimited[idx]) != 1:
            raise RuntimeError(f"{names[idx]} is not force-limited")
        if not np.allclose(model.actuator_ctrlrange[idx], expected):
            raise RuntimeError(f"{names[idx]} ctrlrange mismatch: {model.actuator_ctrlrange[idx]}")
        if not np.allclose(model.actuator_forcerange[idx], expected):
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


def _check_stability(data: mujoco.MjData, qdot: np.ndarray, qacc: np.ndarray, sim_time: float) -> None:
    arrays = [data.qpos, data.qvel, data.qacc, data.ctrl, data.qfrc_bias, data.qfrc_passive]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError(f"NaN or Inf detected at t={sim_time:.4f}s")
    max_abs_qvel = float(np.max(np.abs(qdot)))
    if max_abs_qvel > MAX_ABS_QVEL:
        raise RuntimeError(
            f"Simulation instability at t={sim_time:.4f}s: max |arm qvel|={max_abs_qvel:.3f}"
        )
    max_abs_qacc = float(np.max(np.abs(qacc)))
    if max_abs_qacc > MAX_ABS_QACC:
        raise RuntimeError(
            f"Simulation instability at t={sim_time:.4f}s: max |arm qacc|={max_abs_qacc:.3f}"
        )


def _metrics(
    q: np.ndarray,
    tracking_error: np.ndarray,
    velocity_error: np.ndarray,
    qdot: np.ndarray,
    qacc: np.ndarray,
    a_cmd: np.ndarray,
    tau: np.ndarray,
    tau_unclipped: np.ndarray,
    clip_counts: np.ndarray,
    no_clip_tau_model_diff: float,
) -> dict[str, object]:
    per_joint_rmse = np.sqrt(np.mean(tracking_error**2, axis=0))
    overall_rmse = float(np.sqrt(np.mean(tracking_error**2)))
    per_joint_max_abs_error = np.max(np.abs(tracking_error), axis=0)
    overall_max_abs_error = float(np.max(np.abs(tracking_error)))
    acc_error = qacc - a_cmd
    return {
        "controller": "ctc",
        "q_home": HOME_Q.tolist(),
        "q_target": TARGET_Q.tolist(),
        "delta_q": DELTA_Q.tolist(),
        "duration_scale": 1.0,
        "total_duration": 9.0,
        "timestep": None,
        "torque_limits": ARM_TAU_LIMITS.tolist(),
        "trajectory_schedule": {
            "hold_home_start": [0.0, 1.0],
            "home_to_target": [1.0, 4.0],
            "hold_target": [4.0, 5.0],
            "target_to_home": [5.0, 8.0],
            "hold_home_end": [8.0, 9.0],
        },
        "zeta": ZETA,
        "wn": WN.tolist(),
        "kp_acc": KP_ACC.tolist(),
        "kd_acc": KD_ACC.tolist(),
        "per_joint_rmse": per_joint_rmse.tolist(),
        "overall_rmse": overall_rmse,
        "per_joint_max_abs_error": per_joint_max_abs_error.tolist(),
        "overall_max_abs_error": overall_max_abs_error,
        "final_home_error": (q[-1] - HOME_Q).tolist(),
        "max_abs_qvel": float(np.max(np.abs(qdot))),
        "max_abs_qacc": float(np.max(np.abs(qacc))),
        "max_abs_tau": float(np.max(np.abs(tau))),
        "max_abs_tau_unclipped": float(np.max(np.abs(tau_unclipped))),
        "clip_counts": clip_counts.astype(int).tolist(),
        "torque_saturation": bool(np.any(clip_counts > 0)),
        "overall_velocity_rmse": float(np.sqrt(np.mean(velocity_error**2))),
        "overall_acceleration_tracking_rmse": float(np.sqrt(np.mean(acc_error**2))),
        "max_abs_acceleration_tracking_error": float(np.max(np.abs(acc_error))),
        "no_clip_tau_model_command_max_abs_diff": float(no_clip_tau_model_diff),
        "unstable": False,
    }


def run(viewer: bool) -> dict[str, object]:
    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Torque model XML not found: {MODEL_XML}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    _validate_torque_actuators(model)
    joint_ids, arm_qposadr, arm_dofadr = _arm_addresses(model)
    _validate_target_with_limits(model, joint_ids)
    _initialise_home(model, data)

    total_duration = 9.0
    times = []
    q_des_log = []
    qdot_des_log = []
    qddot_des_log = []
    q_log = []
    qdot_log = []
    qacc_log = []
    err_log = []
    vel_err_log = []
    a_cmd_log = []
    inertial_log = []
    bias_log = []
    passive_log = []
    tau_unclipped_log = []
    tau_log = []
    phase_log = []
    clip_counts = np.zeros(7, dtype=int)
    no_clip_tau_model_diff = 0.0

    def step_once() -> bool:
        nonlocal no_clip_tau_model_diff

        mujoco.mj_forward(model, data)
        q = data.qpos[arm_qposadr].copy()
        qdot = data.qvel[arm_dofadr].copy()
        q_des, qdot_des, qddot_des, phase = desired_state(float(data.time), 1.0)
        output = computed_torque_control(
            model=model,
            data=data,
            arm_dofadr=arm_dofadr,
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
        if not np.any(clipped):
            no_clip_tau_model_diff = max(no_clip_tau_model_diff, float(np.max(np.abs(tau - tau_unclipped))))

        data.ctrl[:7] = tau
        data.ctrl[7] = HOME_CTRL[7]
        mujoco.mj_forward(model, data)
        qacc = data.qacc[arm_dofadr].copy()

        tracking_error = q_des - q
        velocity_error = qdot_des - qdot

        times.append(float(data.time))
        q_des_log.append(q_des.copy())
        qdot_des_log.append(qdot_des.copy())
        qddot_des_log.append(qddot_des.copy())
        q_log.append(q)
        qdot_log.append(qdot)
        qacc_log.append(qacc)
        err_log.append(tracking_error)
        vel_err_log.append(velocity_error)
        a_cmd_log.append(output.a_cmd.copy())
        inertial_log.append(output.tau_inertial.copy())
        bias_log.append(output.qfrc_bias.copy())
        passive_log.append(output.qfrc_passive.copy())
        tau_unclipped_log.append(tau_unclipped.copy())
        tau_log.append(tau.copy())
        phase_log.append(phase)

        _check_stability(data, qdot, qacc, data.time)
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
        "qacc": np.vstack(qacc_log),
        "tracking_error": np.vstack(err_log),
        "velocity_error": np.vstack(vel_err_log),
        "a_cmd": np.vstack(a_cmd_log),
        "tau_inertial": np.vstack(inertial_log),
        "qfrc_bias": np.vstack(bias_log),
        "qfrc_passive": np.vstack(passive_log),
        "tau_unclipped": np.vstack(tau_unclipped_log),
        "tau_command": np.vstack(tau_log),
        "trajectory_phase": np.array(phase_log),
        "q_home": HOME_Q,
        "q_target": TARGET_Q,
        "duration_scale": np.array(1.0),
        "total_duration": np.array(total_duration),
        "timestep": np.array(model.opt.timestep),
        "torque_limits": ARM_TAU_LIMITS,
        "trajectory_schedule": np.array([0.0, 1.0, 4.0, 5.0, 8.0, 9.0]),
        "wn": WN,
        "zeta": np.array(ZETA),
        "kp_acc": KP_ACC,
        "kd_acc": KD_ACC,
        "clip_counts": clip_counts,
    }
    metrics = _metrics(
        q=result["q"],
        tracking_error=result["tracking_error"],
        velocity_error=result["velocity_error"],
        qdot=result["qdot"],
        qacc=result["qacc"],
        a_cmd=result["a_cmd"],
        tau=result["tau_command"],
        tau_unclipped=result["tau_unclipped"],
        clip_counts=clip_counts,
        no_clip_tau_model_diff=no_clip_tau_model_diff,
    )
    metrics["timestep"] = float(model.opt.timestep)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = RESULT_DIR / "ctc.npz"
    metrics_path = RESULT_DIR / "ctc_metrics.json"
    np.savez(npz_path, **result)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("controller = computed_torque")
    print(f"saved = {npz_path}")
    print(f"metrics = {metrics_path}")
    print(f"wn = {np.array2string(WN, precision=4)}")
    print(f"zeta = {ZETA:.4f}")
    print(f"Kp_acc = {np.array2string(KP_ACC, precision=4)}")
    print(f"Kd_acc = {np.array2string(KD_ACC, precision=4)}")
    print(f"overall RMSE = {metrics['overall_rmse']:.8f}")
    print(f"per-joint RMSE = {np.array2string(np.array(metrics['per_joint_rmse']), precision=8)}")
    print(f"overall max abs error = {metrics['overall_max_abs_error']:.8f}")
    print(f"final home error = {np.array2string(np.array(metrics['final_home_error']), precision=8)}")
    print(f"max |qvel| = {metrics['max_abs_qvel']:.8f}")
    print(f"max |qacc| = {metrics['max_abs_qacc']:.8f}")
    print(f"max |tau| = {metrics['max_abs_tau']:.8f}")
    print(f"torque clipping counts = {metrics['clip_counts']}")
    print(f"no-clip tau model/command max abs diff = {no_clip_tau_model_diff:.12e}")
    print("NaN/Inf/unstable = False")
    if metrics["torque_saturation"]:
        print("[WARNING] Torque clipping occurred. Do not increase torque limits to hide this.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Computed torque tracking on Panda torque model.")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    run(viewer=args.viewer)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
