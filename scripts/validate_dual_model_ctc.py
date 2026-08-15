from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.computed_torque import computed_torque_control
from experiments.computed_torque_tracking import KD_ACC, KP_ACC
from experiments.joint_trajectory_tracking import ARM_TAU_LIMITS, HOME_CTRL, HOME_QPOS, desired_state
from experiments.model_mismatch_benchmark import TOTAL_DURATION, compute_metrics, run_dual_model_ctc
from robustness.dual_model import NominalDynamicsModel, arm_qpos_dof_addresses


MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
RESULT_DIR = PROJECT_ROOT / "results" / "model_mismatch"


def _init_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    data.qvel[:] = 0.0
    data.ctrl[: len(HOME_CTRL)] = HOME_CTRL
    mujoco.mj_forward(model, data)


def _simulate_original() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    arm_qposadr, arm_dofadr = arm_qpos_dof_addresses(model)
    _init_home(model, data)

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

    while data.time < TOTAL_DURATION:
        data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        q = data.qpos[arm_qposadr].copy()
        qdot = data.qvel[arm_dofadr].copy()
        q_des, qdot_des, qddot_des, _ = desired_state(float(data.time), 1.0)
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
        tau = np.clip(output.tau_unclipped, -ARM_TAU_LIMITS, ARM_TAU_LIMITS)
        clipped = np.abs(tau - output.tau_unclipped) > 1e-9
        clip_counts += clipped
        data.ctrl[:7] = tau
        data.ctrl[7] = HOME_CTRL[7]

        logs["time"].append(float(data.time))
        logs["q_des"].append(q_des.copy())
        logs["qdot_des"].append(qdot_des.copy())
        logs["qddot_des"].append(qddot_des.copy())
        logs["q"].append(q)
        logs["qdot"].append(qdot)
        logs["position_error"].append(q_des - q)
        logs["velocity_error"].append(qdot_des - qdot)
        logs["a_cmd"].append(output.a_cmd.copy())
        logs["tau_base_unclipped"].append(output.tau_unclipped.copy())
        logs["tau_command"].append(tau.copy())
        logs["nominal_qfrc_bias"].append(output.qfrc_bias.copy())
        logs["nominal_qfrc_passive"].append(output.qfrc_passive.copy())
        logs["plant_qfrc_bias"].append(output.qfrc_bias.copy())
        logs["plant_qfrc_passive"].append(output.qfrc_passive.copy())
        logs["external_force"].append(np.zeros(3))
        logs["torque_clipping_flag"].append(bool(np.any(clipped)))
        mujoco.mj_step(model, data)

    result = {key: np.array(value) for key, value in logs.items()}
    result.update({"clip_counts": clip_counts})
    metrics = compute_metrics("original_ctc", result, clip_counts)
    return result, metrics


def _single_timestep_diagnostic() -> float:
    plant_model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    plant_data = mujoco.MjData(plant_model)
    nominal = NominalDynamicsModel.from_xml_path(MODEL_XML, plant_model=plant_model)
    _, arm_dofadr = arm_qpos_dof_addresses(plant_model)
    _init_home(plant_model, plant_data)
    plant_data.time = 2.25
    mujoco.mj_forward(plant_model, plant_data)
    q_des, qdot_des, qddot_des, _ = desired_state(float(plant_data.time), 1.0)

    original = computed_torque_control(
        plant_model, plant_data, arm_dofadr, q_des, qdot_des, qddot_des, KP_ACC, KD_ACC
    ).tau_unclipped

    nominal.sync_state_from_plant(plant_data)
    nominal.update_dynamics()
    dual = computed_torque_control(
        nominal.model,
        nominal.data,
        nominal.arm_dofadr,
        q_des,
        qdot_des,
        qddot_des,
        KP_ACC,
        KD_ACC,
    ).tau_unclipped
    return float(np.max(np.abs(original - dual)))


def main() -> None:
    original, original_metrics = _simulate_original()
    dual, dual_metrics = run_dual_model_ctc("nominal", save=False)

    max_q_diff = float(np.max(np.abs(original["q"] - dual["q"])))
    max_qdot_diff = float(np.max(np.abs(original["qdot"] - dual["qdot"])))
    max_tau_diff = float(np.max(np.abs(original["tau_command"] - dual["tau_command"])))
    rmse_original = float(original_metrics["overall_rmse"])
    rmse_dual = float(dual_metrics["overall_rmse"])
    rmse_rel_diff = 100.0 * abs(rmse_dual - rmse_original) / rmse_original if rmse_original > 0 else 0.0
    single_step_tau_diff = _single_timestep_diagnostic()

    validation = {
        "original_rmse": rmse_original,
        "dual_nominal_rmse": rmse_dual,
        "rmse_relative_difference_percent": rmse_rel_diff,
        "max_q_difference": max_q_diff,
        "max_qdot_difference": max_qdot_diff,
        "max_tau_difference": max_tau_diff,
        "single_timestep_tau_difference": single_step_tau_diff,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / "dual_model_validation.json"
    output_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    print(f"original CTC RMSE = {rmse_original:.12e}")
    print(f"dual-model nominal CTC RMSE = {rmse_dual:.12e}")
    print(f"RMSE relative difference = {rmse_rel_diff:.12e}%")
    print(f"max |q_original - q_dual| = {max_q_diff:.12e}")
    print(f"max |qdot_original - qdot_dual| = {max_qdot_diff:.12e}")
    print(f"max |tau_original - tau_dual| = {max_tau_diff:.12e}")
    print(f"single timestep torque diff = {single_step_tau_diff:.12e}")
    print(f"saved = {output_path}")

    if max_q_diff > 1e-10 or max_tau_diff > 1e-10 or single_step_tau_diff > 1e-10:
        raise RuntimeError("Dual-model nominal CTC does not reproduce original CTC closely enough")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
