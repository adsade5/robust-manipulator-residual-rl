from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from robustness.dual_model import arm_qpos_dof_addresses


EE_SITE_NAME = "attachment_site"
INERTIAL_SCALES = {"mild": 1.25, "medium": 1.50, "strong": 2.00}
DAMPING_SCALES = {"mild": 1.50, "medium": 2.00, "strong": 3.00}
DISTURBANCE_FORCE = np.array([10.0, 0.0, 0.0])
DISTURBANCE_START = 2.5
DISTURBANCE_END = 3.0


@dataclass(frozen=True)
class InertialPerturbationInfo:
    body_id: int
    body_name: str
    scale: float
    original_mass: float
    original_inertia: np.ndarray
    modified_mass: float
    modified_inertia: np.ndarray


@dataclass(frozen=True)
class DampingPerturbationInfo:
    scale: float
    arm_dofadr: np.ndarray
    original_damping: np.ndarray
    modified_damping: np.ndarray


def ee_body_id(model: mujoco.MjModel, site_name: str = EE_SITE_NAME) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError(f"Required EE site '{site_name}' not found")
    return int(model.site_bodyid[site_id])


def ee_body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def apply_ee_inertial_mismatch(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    level: str,
) -> InertialPerturbationInfo:
    if level not in INERTIAL_SCALES:
        raise ValueError(f"Unknown inertial mismatch level: {level}")
    body_id = ee_body_id(model)
    body_name = ee_body_name(model, body_id)
    scale = INERTIAL_SCALES[level]
    original_mass = float(model.body_mass[body_id])
    original_inertia = model.body_inertia[body_id].copy()

    model.body_mass[body_id] = original_mass * scale
    model.body_inertia[body_id] = original_inertia * scale
    mujoco.mj_setConst(model, data)
    data = data  # explicit: caller should reset/forward its MjData after model constants update.

    return InertialPerturbationInfo(
        body_id=body_id,
        body_name=body_name,
        scale=scale,
        original_mass=original_mass,
        original_inertia=original_inertia,
        modified_mass=float(model.body_mass[body_id]),
        modified_inertia=model.body_inertia[body_id].copy(),
    )


def set_ee_inertial_scale_from_nominal(
    plant_model: mujoco.MjModel,
    plant_data: mujoco.MjData,
    nominal_model: mujoco.MjModel,
    scale: float,
) -> InertialPerturbationInfo:
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive and finite, got {scale}")
    body_id = ee_body_id(plant_model)
    body_name = ee_body_name(plant_model, body_id)
    nominal_body_id = ee_body_id(nominal_model)
    if body_name != ee_body_name(nominal_model, nominal_body_id):
        raise RuntimeError("Plant and nominal EE body names do not match")

    original_mass = float(nominal_model.body_mass[nominal_body_id])
    original_inertia = nominal_model.body_inertia[nominal_body_id].copy()
    plant_model.body_mass[body_id] = original_mass * scale
    plant_model.body_inertia[body_id] = original_inertia * scale
    mujoco.mj_setConst(plant_model, plant_data)
    return InertialPerturbationInfo(
        body_id=body_id,
        body_name=body_name,
        scale=float(scale),
        original_mass=original_mass,
        original_inertia=original_inertia,
        modified_mass=float(plant_model.body_mass[body_id]),
        modified_inertia=plant_model.body_inertia[body_id].copy(),
    )


def apply_joint_damping_mismatch(model: mujoco.MjModel, level: str) -> DampingPerturbationInfo:
    if level not in DAMPING_SCALES:
        raise ValueError(f"Unknown damping mismatch level: {level}")
    _, arm_dofadr = arm_qpos_dof_addresses(model)
    scale = DAMPING_SCALES[level]
    original = model.dof_damping[arm_dofadr].copy()
    model.dof_damping[arm_dofadr] = original * scale
    return DampingPerturbationInfo(
        scale=scale,
        arm_dofadr=arm_dofadr,
        original_damping=original,
        modified_damping=model.dof_damping[arm_dofadr].copy(),
    )


def disturbance_force_at_time(t: float) -> np.ndarray:
    if DISTURBANCE_START <= t < DISTURBANCE_END:
        return DISTURBANCE_FORCE.copy()
    return np.zeros(3)


def apply_external_disturbance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str = EE_SITE_NAME,
    force: np.ndarray | None = None,
) -> np.ndarray:
    force = DISTURBANCE_FORCE.copy() if force is None else np.asarray(force, dtype=float)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError(f"Required EE site '{site_name}' not found")
    body_id = int(model.site_bodyid[site_id])
    mujoco.mj_applyFT(
        model,
        data,
        force,
        np.zeros(3),
        data.site_xpos[site_id].copy(),
        body_id,
        data.qfrc_applied,
    )
    return force.copy()
