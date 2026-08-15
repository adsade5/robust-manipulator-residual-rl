from __future__ import annotations

import json
import sys
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
K_POS = np.array([250.0, 250.0, 250.0])
D_POS = np.array([55.0, 55.0, 55.0])
K_ROT = np.array([30.0, 30.0, 30.0])
D_ROT = np.array([8.0, 8.0, 8.0])
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
    arrays = [data.qpos, data.qvel, data.ctrl, data.qfrc_bias, data.qfrc_passive]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError(f"NaN or Inf detected at t={sim_time:.4f}s")
    max_qvel = float(np.max(np.abs(data.qvel[arm_dofadr])))
    if max_qvel > MAX_ABS_QVEL:
        raise RuntimeError(f"Simulation instability at t={sim_time:.4f}s: max |qvel|={max_qvel:.3f}")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    if site_id < 0:
        raise RuntimeError(f"Required EE site '{EE_SITE_NAME}' not found")
    _, arm_dofadr = _arm_addresses(model)

    data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    data.qvel[:] = 0.0
    data.ctrl[: len(HOME_CTRL)] = HOME_CTRL
    mujoco.mj_forward(model, data)
    p_des = data.site_xpos[site_id].copy()
    r_des = data.site_xmat[site_id].reshape(3, 3).copy()

    logs: dict[str, list[np.ndarray | float]] = {
        "time": [],
        "position_error": [],
        "orientation_error": [],
        "linear_velocity": [],
        "angular_velocity": [],
        "tau_command": [],
        "tau_unclipped": [],
    }
    clip_counts = np.zeros(7, dtype=int)
    duration = 5.0

    while data.time < duration:
        data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        output = cartesian_impedance_control(
            model, data, site_id, arm_dofadr, p_des, r_des, K_POS, D_POS, K_ROT, D_ROT
        )
        tau = np.clip(output.tau_unclipped, -ARM_TAU_LIMITS, ARM_TAU_LIMITS)
        clip_counts += np.abs(tau - output.tau_unclipped) > 1e-9
        data.ctrl[:7] = tau
        data.ctrl[7] = HOME_CTRL[7]

        logs["time"].append(float(data.time))
        logs["position_error"].append(output.position_error.copy())
        logs["orientation_error"].append(output.orientation_error.copy())
        logs["linear_velocity"].append(output.linear_velocity.copy())
        logs["angular_velocity"].append(output.angular_velocity.copy())
        logs["tau_command"].append(tau.copy())
        logs["tau_unclipped"].append(output.tau_unclipped.copy())

        _check_stability(data, arm_dofadr, data.time)
        mujoco.mj_step(model, data)

    result = {key: np.array(value) for key, value in logs.items()}
    result.update(
        {
            "p_des": p_des,
            "r_des": r_des,
            "k_pos": K_POS,
            "d_pos": D_POS,
            "k_rot": K_ROT,
            "d_rot": D_ROT,
            "clip_counts": clip_counts,
        }
    )
    max_position_drift = float(np.max(np.linalg.norm(result["position_error"], axis=1)))
    max_orientation_drift = float(np.max(np.linalg.norm(result["orientation_error"], axis=1)))
    metrics = {
        "max_position_drift": max_position_drift,
        "max_orientation_drift": max_orientation_drift,
        "max_abs_tau": float(np.max(np.abs(result["tau_command"]))),
        "clip_counts": clip_counts.astype(int).tolist(),
        "torque_saturation": bool(np.any(clip_counts > 0)),
        "unstable": False,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = RESULT_DIR / "hold.npz"
    metrics_path = RESULT_DIR / "hold_metrics.json"
    np.savez(npz_path, **result)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"saved = {npz_path}")
    print(f"metrics = {metrics_path}")
    print(f"max position drift = {max_position_drift:.12e} m")
    print(f"max orientation drift = {max_orientation_drift:.12e} rad")
    print(f"max |tau| = {metrics['max_abs_tau']:.8f}")
    print(f"clipping counts = {metrics['clip_counts']}")
    print("NaN/Inf/unstable = False")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
