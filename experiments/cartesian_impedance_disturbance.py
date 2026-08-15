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

from controllers.cartesian_impedance import cartesian_impedance_control
from experiments.joint_trajectory_tracking import ARM_TAU_LIMITS, HOME_CTRL, HOME_QPOS


MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
RESULT_DIR = PROJECT_ROOT / "results" / "cartesian_impedance"
EE_SITE_NAME = "attachment_site"
STIFFNESS_CONFIGS = {
    "low": {
        "k_pos": np.array([100.0, 100.0, 100.0]),
        "d_pos": np.array([35.0, 35.0, 35.0]),
    },
    "medium": {
        "k_pos": np.array([250.0, 250.0, 250.0]),
        "d_pos": np.array([55.0, 55.0, 55.0]),
    },
    "high": {
        "k_pos": np.array([500.0, 500.0, 500.0]),
        "d_pos": np.array([80.0, 80.0, 80.0]),
    },
}
K_ROT = np.array([30.0, 30.0, 30.0])
D_ROT = np.array([8.0, 8.0, 8.0])
DISTURBANCE_FORCE = np.array([10.0, 0.0, 0.0])
FORCE_START = 2.0
FORCE_END = 4.0
TOTAL_DURATION = 8.0
MAX_ABS_QVEL = 80.0


def _arm_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qposadr = []
    dofadr = []
    for index in range(1, 8):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}")
        if joint_id < 0:
            raise RuntimeError(f"Missing joint{index}")
        qposadr.append(int(model.jnt_qposadr[joint_id]))
        dofadr.append(int(model.jnt_dofadr[joint_id]))
    return np.array(qposadr, dtype=int), np.array(dofadr, dtype=int)


def _check_stability(data: mujoco.MjData, arm_dofadr: np.ndarray, sim_time: float) -> None:
    arrays = [
        data.qpos,
        data.qvel,
        data.ctrl,
        data.qfrc_bias,
        data.qfrc_passive,
        data.qfrc_applied,
    ]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError(f"NaN or Inf detected at t={sim_time:.4f}s")
    max_qvel = float(np.max(np.abs(data.qvel[arm_dofadr])))
    if max_qvel > MAX_ABS_QVEL:
        raise RuntimeError(f"Simulation instability at t={sim_time:.4f}s: max |qvel|={max_qvel:.3f}")


def _recovery_time(time_log: np.ndarray, pos_err: np.ndarray) -> float | None:
    threshold = 0.002
    hold_time = 0.2
    dt = float(np.median(np.diff(time_log)))
    needed = max(1, int(np.ceil(hold_time / dt)))
    norm_err = np.linalg.norm(pos_err, axis=1)
    start_idx = int(np.searchsorted(time_log, FORCE_END))
    ok = norm_err < threshold
    for idx in range(start_idx, len(time_log) - needed + 1):
        if np.all(ok[idx : idx + needed]):
            return float(time_log[idx] - FORCE_END)
    return None


def _metrics(
    stiffness: str,
    result: dict[str, np.ndarray],
    clip_counts: np.ndarray,
) -> dict[str, object]:
    time_log = result["time"]
    pos_err = result["position_error"]
    steady_mask = (time_log >= 3.5) & (time_log <= 4.0)
    initial_mask = time_log < FORCE_START
    x_error = pos_err[:, 0]
    steady_abs_dx = float(np.mean(np.abs(x_error[steady_mask])))
    max_displacement = float(np.max(np.linalg.norm(pos_err, axis=1)))
    recovery = _recovery_time(time_log, pos_err)
    kx = float(result["k_pos"][0])
    theoretical = float(np.linalg.norm(DISTURBANCE_FORCE) / kx)
    return {
        "stiffness": stiffness,
        "k_pos": result["k_pos"].tolist(),
        "d_pos": result["d_pos"].tolist(),
        "k_rot": result["k_rot"].tolist(),
        "d_rot": result["d_rot"].tolist(),
        "disturbance_force": DISTURBANCE_FORCE.tolist(),
        "initial_hold_position_error": float(np.max(np.linalg.norm(pos_err[initial_mask], axis=1))),
        "steady_x_displacement_abs": steady_abs_dx,
        "steady_x_displacement_signed_mean": float(np.mean(x_error[steady_mask])),
        "theoretical_x_displacement": theoretical,
        "maximum_displacement": max_displacement,
        "maximum_abs_x_displacement": float(np.max(np.abs(x_error))),
        "recovery_time": recovery,
        "recovered": recovery is not None,
        "recovery_threshold_m": 0.002,
        "recovery_hold_s": 0.2,
        "max_cartesian_velocity": float(np.max(np.linalg.norm(result["linear_velocity"], axis=1))),
        "max_joint_torque": float(np.max(np.abs(result["tau_command"]))),
        "clip_counts": clip_counts.astype(int).tolist(),
        "torque_saturation": bool(np.any(clip_counts > 0)),
        "orientation_max_error": float(np.max(np.linalg.norm(result["orientation_error"], axis=1))),
        "unstable": False,
    }


def run(stiffness: str, viewer: bool) -> dict[str, object]:
    config = STIFFNESS_CONFIGS[stiffness]
    k_pos = config["k_pos"]
    d_pos = config["d_pos"]

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    if site_id < 0:
        raise RuntimeError(f"Required EE site '{EE_SITE_NAME}' not found")
    body_id = int(model.site_bodyid[site_id])
    arm_qposadr, arm_dofadr = _arm_addresses(model)

    data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    data.qvel[:] = 0.0
    data.ctrl[: len(HOME_CTRL)] = HOME_CTRL
    mujoco.mj_forward(model, data)
    p_des = data.site_xpos[site_id].copy()
    r_des = data.site_xmat[site_id].reshape(3, 3).copy()

    logs: dict[str, list[np.ndarray | float]] = {
        "time": [],
        "q": [],
        "qdot": [],
        "ee_position": [],
        "ee_orientation": [],
        "position_error": [],
        "orientation_error": [],
        "linear_velocity": [],
        "angular_velocity": [],
        "external_force": [],
        "cartesian_force": [],
        "cartesian_moment": [],
        "tau_task": [],
        "qfrc_bias": [],
        "qfrc_passive": [],
        "tau_unclipped": [],
        "tau_command": [],
    }
    clip_counts = np.zeros(7, dtype=int)

    def step_once() -> bool:
        data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        external_force = (
            DISTURBANCE_FORCE.copy()
            if FORCE_START <= float(data.time) < FORCE_END
            else np.zeros(3)
        )
        if np.any(external_force):
            mujoco.mj_applyFT(
                model,
                data,
                external_force,
                np.zeros(3),
                data.site_xpos[site_id].copy(),
                body_id,
                data.qfrc_applied,
            )

        output = cartesian_impedance_control(
            model, data, site_id, arm_dofadr, p_des, r_des, k_pos, d_pos, K_ROT, D_ROT
        )
        tau = np.clip(output.tau_unclipped, -ARM_TAU_LIMITS, ARM_TAU_LIMITS)
        clip_counts[:] += np.abs(tau - output.tau_unclipped) > 1e-9
        data.ctrl[:7] = tau
        data.ctrl[7] = HOME_CTRL[7]
        mujoco.mj_forward(model, data)

        logs["time"].append(float(data.time))
        logs["q"].append(data.qpos[arm_qposadr].copy())
        logs["qdot"].append(data.qvel[arm_dofadr].copy())
        logs["ee_position"].append(data.site_xpos[site_id].copy())
        logs["ee_orientation"].append(data.site_xmat[site_id].reshape(3, 3).copy())
        logs["position_error"].append(output.position_error.copy())
        logs["orientation_error"].append(output.orientation_error.copy())
        logs["linear_velocity"].append(output.linear_velocity.copy())
        logs["angular_velocity"].append(output.angular_velocity.copy())
        logs["external_force"].append(external_force.copy())
        logs["cartesian_force"].append(output.cartesian_force.copy())
        logs["cartesian_moment"].append(output.cartesian_moment.copy())
        logs["tau_task"].append(output.tau_task.copy())
        logs["qfrc_bias"].append(output.qfrc_bias.copy())
        logs["qfrc_passive"].append(output.qfrc_passive.copy())
        logs["tau_unclipped"].append(output.tau_unclipped.copy())
        logs["tau_command"].append(tau.copy())

        _check_stability(data, arm_dofadr, data.time)
        mujoco.mj_step(model, data)
        return data.time < TOTAL_DURATION

    if viewer:
        viewer_module = importlib.import_module("mujoco.viewer")
        with viewer_module.launch_passive(model, data) as handle:
            while handle.is_running() and data.time < TOTAL_DURATION:
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
        while data.time < TOTAL_DURATION:
            step_once()

    result = {key: np.array(value) for key, value in logs.items()}
    result.update(
        {
            "p_des": p_des,
            "r_des": r_des,
            "k_pos": k_pos,
            "d_pos": d_pos,
            "k_rot": K_ROT,
            "d_rot": D_ROT,
            "disturbance_force": DISTURBANCE_FORCE,
            "force_window": np.array([FORCE_START, FORCE_END]),
            "torque_limits": ARM_TAU_LIMITS,
            "clip_counts": clip_counts,
        }
    )
    metrics = _metrics(stiffness, result, clip_counts)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = RESULT_DIR / f"disturbance_{stiffness}.npz"
    metrics_path = RESULT_DIR / f"disturbance_{stiffness}_metrics.json"
    np.savez(npz_path, **result)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"stiffness = {stiffness}")
    print(f"saved = {npz_path}")
    print(f"metrics = {metrics_path}")
    print(f"K_pos = {np.array2string(k_pos, precision=4)}")
    print(f"D_pos = {np.array2string(d_pos, precision=4)}")
    print(f"steady |dx| = {metrics['steady_x_displacement_abs']:.8f} m")
    print(f"F/K theoretical dx = {metrics['theoretical_x_displacement']:.8f} m")
    print(f"max displacement = {metrics['maximum_displacement']:.8f} m")
    print(f"recovery time = {metrics['recovery_time']}")
    print(f"max torque = {metrics['max_joint_torque']:.8f} Nm")
    print(f"clipping counts = {metrics['clip_counts']}")
    print(f"orientation max error = {metrics['orientation_max_error']:.8f} rad")
    print("NaN/Inf/unstable = False")
    if metrics["torque_saturation"]:
        print("[WARNING] Torque clipping occurred. Do not increase torque limits to hide this.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Cartesian impedance disturbance experiment.")
    parser.add_argument("--stiffness", choices=["low", "medium", "high"], required=True)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    run(args.stiffness, args.viewer)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
