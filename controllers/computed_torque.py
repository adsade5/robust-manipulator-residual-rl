from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ComputedTorqueOutput:
    tau_unclipped: np.ndarray
    a_cmd: np.ndarray
    tau_inertial: np.ndarray
    qfrc_bias: np.ndarray
    qfrc_passive: np.ndarray


def computed_torque_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_dofadr: np.ndarray,
    q_des: np.ndarray,
    qdot_des: np.ndarray,
    qddot_des: np.ndarray,
    kp_acc: np.ndarray,
    kd_acc: np.ndarray,
) -> ComputedTorqueOutput:
    arm_dofadr = np.asarray(arm_dofadr, dtype=int)
    q_des = np.asarray(q_des, dtype=float)
    qdot_des = np.asarray(qdot_des, dtype=float)
    qddot_des = np.asarray(qddot_des, dtype=float)
    kp_acc = np.asarray(kp_acc, dtype=float)
    kd_acc = np.asarray(kd_acc, dtype=float)

    for name, value in [
        ("arm_dofadr", arm_dofadr),
        ("q_des", q_des),
        ("qdot_des", qdot_des),
        ("qddot_des", qddot_des),
        ("kp_acc", kp_acc),
        ("kd_acc", kd_acc),
    ]:
        if value.shape != (7,):
            raise ValueError(f"{name} must have shape (7,), got {value.shape}")
    for name, value in [
        ("q_des", q_des),
        ("qdot_des", qdot_des),
        ("qddot_des", qddot_des),
        ("kp_acc", kp_acc),
        ("kd_acc", kd_acc),
    ]:
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or Inf")

    q = data.qpos[arm_dofadr]
    qdot = data.qvel[arm_dofadr]
    e = q_des - q
    edot = qdot_des - qdot
    a_cmd = qddot_des + kp_acc * e + kd_acc * edot

    a_full = np.zeros(model.nv)
    a_full[arm_dofadr] = a_cmd
    inertial_term_full = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, inertial_term_full, a_full)

    tau_inertial = inertial_term_full[arm_dofadr].copy()
    qfrc_bias = data.qfrc_bias[arm_dofadr].copy()
    qfrc_passive = data.qfrc_passive[arm_dofadr].copy()
    tau_unclipped = tau_inertial + qfrc_bias - qfrc_passive

    return ComputedTorqueOutput(
        tau_unclipped=tau_unclipped.copy(),
        a_cmd=a_cmd.copy(),
        tau_inertial=tau_inertial,
        qfrc_bias=qfrc_bias,
        qfrc_passive=qfrc_passive,
    )
