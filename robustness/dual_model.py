from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


def arm_joint_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = []
    for name in ARM_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Missing arm joint: {name}")
        ids.append(joint_id)
    return np.array(ids, dtype=int)


def arm_qpos_dof_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = arm_joint_ids(model)
    qposadr = np.array([int(model.jnt_qposadr[jid]) for jid in joint_ids], dtype=int)
    dofadr = np.array([int(model.jnt_dofadr[jid]) for jid in joint_ids], dtype=int)
    return qposadr, dofadr


def full_mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    matrix = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, matrix)
    return matrix


@dataclass
class NominalDynamicsModel:
    model: mujoco.MjModel
    data: mujoco.MjData
    arm_qposadr: np.ndarray
    arm_dofadr: np.ndarray

    @classmethod
    def from_xml_path(cls, xml_path: str | Path, plant_model: mujoco.MjModel | None = None) -> "NominalDynamicsModel":
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        arm_qposadr, arm_dofadr = arm_qpos_dof_addresses(model)
        instance = cls(model=model, data=data, arm_qposadr=arm_qposadr, arm_dofadr=arm_dofadr)
        if plant_model is not None:
            instance.validate_against_plant(plant_model)
        return instance

    def validate_against_plant(self, plant_model: mujoco.MjModel) -> None:
        if plant_model.nq != self.model.nq:
            raise RuntimeError(f"plant nq={plant_model.nq} != nominal nq={self.model.nq}")
        if plant_model.nv != self.model.nv:
            raise RuntimeError(f"plant nv={plant_model.nv} != nominal nv={self.model.nv}")

        plant_qposadr, plant_dofadr = arm_qpos_dof_addresses(plant_model)
        if not np.array_equal(plant_qposadr, self.arm_qposadr):
            raise RuntimeError(f"Arm qpos addresses differ: plant={plant_qposadr}, nominal={self.arm_qposadr}")
        if not np.array_equal(plant_dofadr, self.arm_dofadr):
            raise RuntimeError(f"Arm dof addresses differ: plant={plant_dofadr}, nominal={self.arm_dofadr}")

        for name in ARM_JOINTS:
            plant_id = mujoco.mj_name2id(plant_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            nominal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if plant_id < 0 or nominal_id < 0:
                raise RuntimeError(f"Missing joint while validating {name}")

    def sync_state_from_plant(self, plant_data: mujoco.MjData) -> None:
        if plant_data.qpos.shape != self.data.qpos.shape or plant_data.qvel.shape != self.data.qvel.shape:
            raise RuntimeError("Plant and nominal state vector shapes differ")
        self.data.qpos[:] = plant_data.qpos
        self.data.qvel[:] = plant_data.qvel
        if plant_data.ctrl.shape == self.data.ctrl.shape:
            self.data.ctrl[:] = plant_data.ctrl

    def update_dynamics(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def get_mass_matrix(self) -> np.ndarray:
        return full_mass_matrix(self.model, self.data)

    def mass_vector_product(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.model.nv,):
            raise ValueError(f"vector must have shape ({self.model.nv},), got {vector.shape}")
        result = np.zeros(self.model.nv)
        mujoco.mj_mulM(self.model, self.data, result, vector)
        return result

    def get_bias_force(self) -> np.ndarray:
        return self.data.qfrc_bias.copy()

    def get_passive_force(self) -> np.ndarray:
        return self.data.qfrc_passive.copy()
