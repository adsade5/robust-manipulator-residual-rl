from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from controllers.cartesian_utils import orientation_error, position_error


@dataclass(frozen=True)
class CartesianImpedanceOutput:
    tau_unclipped: np.ndarray
    tau_task: np.ndarray
    position_error: np.ndarray
    orientation_error: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    cartesian_force: np.ndarray
    cartesian_moment: np.ndarray
    qfrc_bias: np.ndarray
    qfrc_passive: np.ndarray
    jacobian_arm: np.ndarray


def cartesian_impedance_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    arm_dofadr: np.ndarray,
    p_des: np.ndarray,
    r_des: np.ndarray,
    k_pos: np.ndarray,
    d_pos: np.ndarray,
    k_rot: np.ndarray,
    d_rot: np.ndarray,
) -> CartesianImpedanceOutput:
    arm_dofadr = np.asarray(arm_dofadr, dtype=int)
    p_des = np.asarray(p_des, dtype=float)
    r_des = np.asarray(r_des, dtype=float)
    k_pos = np.asarray(k_pos, dtype=float)
    d_pos = np.asarray(d_pos, dtype=float)
    k_rot = np.asarray(k_rot, dtype=float)
    d_rot = np.asarray(d_rot, dtype=float)

    for name, value, shape in [
        ("arm_dofadr", arm_dofadr, (7,)),
        ("p_des", p_des, (3,)),
        ("r_des", r_des, (3, 3)),
        ("k_pos", k_pos, (3,)),
        ("d_pos", d_pos, (3,)),
        ("k_rot", k_rot, (3,)),
        ("d_rot", d_rot, (3,)),
    ]:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    for name, value in [
        ("p_des", p_des),
        ("r_des", r_des),
        ("k_pos", k_pos),
        ("d_pos", d_pos),
        ("k_rot", k_rot),
        ("d_rot", d_rot),
    ]:
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or Inf")

    p = data.site_xpos[site_id].copy()
    r = data.site_xmat[site_id].reshape(3, 3).copy()
    e_pos = position_error(p_des, p)
    e_rot = orientation_error(r_des, r)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jacp_arm = jacp[:, arm_dofadr]
    jacr_arm = jacr[:, arm_dofadr]
    jac_arm = np.vstack([jacp_arm, jacr_arm])

    qdot_arm = data.qvel[arm_dofadr].copy()
    linear_velocity = jacp_arm @ qdot_arm
    angular_velocity = jacr_arm @ qdot_arm

    force = k_pos * e_pos + d_pos * (-linear_velocity)
    moment = k_rot * e_rot + d_rot * (-angular_velocity)
    wrench = np.concatenate([force, moment])
    tau_task = jac_arm.T @ wrench
    qfrc_bias = data.qfrc_bias[arm_dofadr].copy()
    qfrc_passive = data.qfrc_passive[arm_dofadr].copy()
    tau_unclipped = tau_task + qfrc_bias - qfrc_passive

    return CartesianImpedanceOutput(
        tau_unclipped=tau_unclipped.copy(),
        tau_task=tau_task.copy(),
        position_error=e_pos,
        orientation_error=e_rot,
        linear_velocity=linear_velocity.copy(),
        angular_velocity=angular_velocity.copy(),
        cartesian_force=force.copy(),
        cartesian_moment=moment.copy(),
        qfrc_bias=qfrc_bias,
        qfrc_passive=qfrc_passive,
        jacobian_arm=jac_arm.copy(),
    )
