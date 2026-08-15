from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.joint_trajectory_tracking import HOME_CTRL, HOME_QPOS
from robustness.dual_model import NominalDynamicsModel, arm_qpos_dof_addresses, full_mass_matrix
from robustness.perturbations import apply_ee_inertial_mismatch, apply_joint_damping_mismatch, ee_body_id, ee_body_name


MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
RESULT_DIR = PROJECT_ROOT / "results" / "model_mismatch"


def _init_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    data.qvel[:] = 0.0
    data.ctrl[: len(HOME_CTRL)] = HOME_CTRL
    mujoco.mj_forward(model, data)


def _arm_mass(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    _, dofs = arm_qpos_dof_addresses(model)
    return full_mass_matrix(model, data)[np.ix_(dofs, dofs)]


def main() -> None:
    plant_model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    plant_data = mujoco.MjData(plant_model)
    nominal = NominalDynamicsModel.from_xml_path(MODEL_XML, plant_model=plant_model)
    _init_state(plant_model, plant_data)
    _init_state(nominal.model, nominal.data)
    body_id = ee_body_id(plant_model)
    body_name = ee_body_name(plant_model, body_id)

    nominal_mass_before = float(nominal.model.body_mass[body_id])
    nominal_inertia_before = nominal.model.body_inertia[body_id].copy()
    nominal_m_before = _arm_mass(nominal.model, nominal.data)
    plant_m_nominal = _arm_mass(plant_model, plant_data)
    nominal_condition_diff = float(np.max(np.abs(plant_m_nominal - nominal_m_before)))

    info = apply_ee_inertial_mismatch(plant_model, plant_data, "strong")
    plant_data = mujoco.MjData(plant_model)
    _init_state(plant_model, plant_data)
    nominal.update_dynamics()
    plant_m_perturbed = _arm_mass(plant_model, plant_data)
    nominal_m_after = _arm_mass(nominal.model, nominal.data)
    inertial_m_diff = float(np.max(np.abs(plant_m_perturbed - nominal_m_after)))
    nominal_m_change = float(np.max(np.abs(nominal_m_after - nominal_m_before)))
    nominal_mass_after = float(nominal.model.body_mass[body_id])
    nominal_inertia_after = nominal.model.body_inertia[body_id].copy()

    if nominal_condition_diff > 1e-12:
        raise RuntimeError(f"Nominal condition M mismatch: {nominal_condition_diff:.12e}")
    if inertial_m_diff <= 1e-8:
        raise RuntimeError("Inertial perturbation did not change plant M relative to nominal")
    if nominal_m_change > 1e-12:
        raise RuntimeError("Nominal M changed after plant inertial perturbation")
    if not np.isclose(nominal_mass_before, nominal_mass_after) or not np.allclose(
        nominal_inertia_before, nominal_inertia_after
    ):
        raise RuntimeError("Nominal body mass/inertia changed after plant perturbation")

    damping_plant = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    damping_data = mujoco.MjData(damping_plant)
    damping_nominal = NominalDynamicsModel.from_xml_path(MODEL_XML, plant_model=damping_plant)
    _init_state(damping_plant, damping_data)
    _init_state(damping_nominal.model, damping_nominal.data)
    _, arm_dofadr = arm_qpos_dof_addresses(damping_plant)
    nominal_damping_before = damping_nominal.model.dof_damping[arm_dofadr].copy()
    damping_info = apply_joint_damping_mismatch(damping_plant, "strong")
    damping_data = mujoco.MjData(damping_plant)
    _init_state(damping_plant, damping_data)
    qvel_test = np.array([0.25, -0.2, 0.15, -0.1, 0.08, -0.06, 0.04])
    damping_data.qvel[arm_dofadr] = qvel_test
    damping_nominal.data.qpos[:] = damping_data.qpos
    damping_nominal.data.qvel[:] = damping_data.qvel
    mujoco.mj_forward(damping_plant, damping_data)
    damping_nominal.update_dynamics()
    passive_diff = float(
        np.max(np.abs(damping_data.qfrc_passive[arm_dofadr] - damping_nominal.data.qfrc_passive[arm_dofadr]))
    )
    nominal_damping_after = damping_nominal.model.dof_damping[arm_dofadr].copy()
    if passive_diff <= 1e-8:
        raise RuntimeError("Damping perturbation did not produce passive-force difference")
    if not np.allclose(nominal_damping_before, nominal_damping_after):
        raise RuntimeError("Nominal damping changed after plant damping perturbation")

    validation = {
        "ee_body_name": body_name,
        "nominal_body_mass": nominal_mass_before,
        "nominal_body_inertia": nominal_inertia_before.tolist(),
        "perturbed_plant_body_mass": info.modified_mass,
        "perturbed_plant_body_inertia": info.modified_inertia.tolist(),
        "nominal_condition_m_max_abs_diff": nominal_condition_diff,
        "inertial_mismatch_m_max_abs_diff": inertial_m_diff,
        "nominal_m_change_after_plant_perturbation": nominal_m_change,
        "nominal_damping": nominal_damping_before.tolist(),
        "plant_damping_strong": damping_info.modified_damping.tolist(),
        "passive_force_max_abs_diff": passive_diff,
        "nominal_model_unchanged": True,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / "model_separation_validation.json"
    output_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    print(f"EE body = {body_name}")
    print(f"Nominal body mass = {nominal_mass_before:.12e}")
    print(f"Nominal body inertia = {np.array2string(nominal_inertia_before, precision=12)}")
    print(f"Perturbed plant body mass = {info.modified_mass:.12e}")
    print(f"Perturbed plant body inertia = {np.array2string(info.modified_inertia, precision=12)}")
    print(f"Nominal condition max |M_plant - M_nominal| = {nominal_condition_diff:.12e}")
    print(f"Inertial mismatch max |M_plant - M_nominal| = {inertial_m_diff:.12e}")
    print(f"Nominal M change after plant perturbation = {nominal_m_change:.12e}")
    print(f"Nominal damping = {np.array2string(nominal_damping_before, precision=8)}")
    print(f"Plant damping strong = {np.array2string(damping_info.modified_damping, precision=8)}")
    print(f"Passive force max abs diff = {passive_diff:.12e}")
    print("Nominal model remained unchanged = True")
    print(f"saved = {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
