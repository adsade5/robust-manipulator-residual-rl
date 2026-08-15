from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
RESULT_DIR = PROJECT_ROOT / "results" / "minimal_torque"

HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])
HOME_CTRL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0])
ARM_TAU_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

KP = np.array([60.0, 80.0, 55.0, 45.0, 18.0, 14.0, 8.0])
KD = np.array([12.0, 16.0, 10.0, 9.0, 4.0, 3.0, 2.0])
MAX_ABS_QVEL = 80.0


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    return mujoco.mj_id2name(model, obj_type, obj_id) or ""


def _arm_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qposadr = []
    dofadr = []
    for index in range(1, 8):
        joint_name = f"joint{index}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Missing joint: {joint_name}")
        qposadr.append(int(model.jnt_qposadr[joint_id]))
        dofadr.append(int(model.jnt_dofadr[joint_id]))
    return np.array(qposadr, dtype=int), np.array(dofadr, dtype=int)


def _validate_torque_actuators(model: mujoco.MjModel) -> None:
    names = [_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    if model.nu < 8:
        raise RuntimeError(f"Expected at least 8 actuators, got {model.nu}")
    if names[:7] != [f"torque{i}" for i in range(1, 8)]:
        raise RuntimeError(f"First 7 actuators are not torque motors: {names[:7]}")
    for idx in range(7):
        if not np.allclose(model.actuator_biasprm[idx], 0.0):
            raise RuntimeError(f"{names[idx]} has nonzero biasprm: {model.actuator_biasprm[idx]}")
        if not np.allclose(model.actuator_gainprm[idx, 0], 1.0):
            raise RuntimeError(f"{names[idx]} does not have fixed gain 1")


def _initialise_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if model.nq < len(HOME_QPOS) or model.nu < len(HOME_CTRL):
        raise RuntimeError("Model dimensions are smaller than the expected Panda home state")
    data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    data.qvel[:] = 0.0
    data.ctrl[: len(HOME_CTRL)] = HOME_CTRL
    mujoco.mj_forward(model, data)


def _compute_tau(
    mode: str,
    q: np.ndarray,
    qdot: np.ndarray,
    qfrc_bias_arm: np.ndarray,
) -> np.ndarray:
    if mode == "none":
        return np.zeros(7)
    if mode == "gravity":
        return qfrc_bias_arm.copy()
    if mode == "pd":
        return KP * (HOME_QPOS[:7] - q) - KD * qdot
    if mode == "pd_gc":
        return KP * (HOME_QPOS[:7] - q) - KD * qdot + qfrc_bias_arm
    raise ValueError(f"Unsupported mode: {mode}")


def _check_stability(data: mujoco.MjData, qdot: np.ndarray, sim_time: float) -> None:
    arrays = [data.qpos, data.qvel, data.ctrl, data.qfrc_bias]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError(f"NaN or Inf detected at t={sim_time:.4f}s")
    max_abs_qvel = float(np.max(np.abs(qdot)))
    if max_abs_qvel > MAX_ABS_QVEL:
        raise RuntimeError(
            f"Simulation instability at t={sim_time:.4f}s: max |arm qvel|={max_abs_qvel:.3f}"
        )


def run(mode: str, duration: float, viewer: bool) -> dict[str, np.ndarray | float | bool]:
    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Torque model XML not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    _validate_torque_actuators(model)
    arm_qposadr, arm_dofadr = _arm_addresses(model)
    _initialise_home(model, data)

    times = []
    qs = []
    qdots = []
    taus = []
    biases = []
    errors = []
    clip_counts = np.zeros(7, dtype=int)

    initial_q = data.qpos[arm_qposadr].copy()
    max_abs_qdot = 0.0
    max_abs_tau = 0.0

    def step_once() -> bool:
        nonlocal max_abs_qdot, max_abs_tau

        mujoco.mj_forward(model, data)
        q = data.qpos[arm_qposadr].copy()
        qdot = data.qvel[arm_dofadr].copy()
        qfrc_bias_arm = data.qfrc_bias[arm_dofadr].copy()

        tau_raw = _compute_tau(mode, q, qdot, qfrc_bias_arm)
        tau = np.clip(tau_raw, -ARM_TAU_LIMITS, ARM_TAU_LIMITS)
        clip_counts[:] += np.abs(tau - tau_raw) > 1e-9

        data.ctrl[:7] = tau
        data.ctrl[7] = HOME_CTRL[7]

        times.append(float(data.time))
        qs.append(q)
        qdots.append(qdot)
        taus.append(tau.copy())
        biases.append(qfrc_bias_arm)
        if mode in {"pd", "pd_gc"}:
            errors.append(HOME_QPOS[:7] - q)

        max_abs_qdot = max(max_abs_qdot, float(np.max(np.abs(qdot))))
        max_abs_tau = max(max_abs_tau, float(np.max(np.abs(tau))))
        _check_stability(data, qdot, data.time)

        mujoco.mj_step(model, data)
        return data.time < duration

    if viewer:
        viewer_module = importlib.import_module("mujoco.viewer")

        with viewer_module.launch_passive(model, data) as handle:
            while handle.is_running() and data.time < duration:
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
        while data.time < duration:
            step_once()

    final_q = data.qpos[arm_qposadr].copy()
    result: dict[str, np.ndarray | float | bool] = {
        "time": np.array(times),
        "q": np.vstack(qs),
        "qdot": np.vstack(qdots),
        "tau_command": np.vstack(taus),
        "qfrc_bias": np.vstack(biases),
        "initial_q": initial_q,
        "final_q": final_q,
        "max_abs_qdot": max_abs_qdot,
        "max_abs_tau": max_abs_tau,
        "clip_counts": clip_counts,
        "torque_clipping": bool(np.any(clip_counts > 0)),
        "unstable": False,
    }
    if errors:
        error_array = np.vstack(errors)
        result["q_error"] = error_array
        result["final_joint_error"] = HOME_QPOS[:7] - final_q
        result["rmse"] = np.sqrt(np.mean(error_array**2, axis=0))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / f"{mode}.npz"
    np.savez(output_path, **result)

    print(f"mode = {mode}")
    print(f"saved = {output_path}")
    print(f"initial q = {np.array2string(initial_q, precision=6)}")
    print(f"final q = {np.array2string(final_q, precision=6)}")
    print(f"max |qdot| = {max_abs_qdot:.6f}")
    print(f"max |tau| = {max_abs_tau:.6f}")
    print(f"torque clipping = {bool(np.any(clip_counts > 0))}, counts = {clip_counts.tolist()}")
    print("NaN/unstable = False")
    if mode in {"pd", "pd_gc"}:
        print(f"final joint error = {np.array2string(result['final_joint_error'], precision=6)}")
        print(f"RMSE = {np.array2string(result['rmse'], precision=6)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal Panda torque-control experiments.")
    parser.add_argument("--mode", choices=["none", "gravity", "pd", "pd_gc"], required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    run(args.mode, args.duration, args.viewer)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
