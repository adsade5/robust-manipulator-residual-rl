from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
EXPECTED_LIMITS = np.array(
    [[-87.0, 87.0]] * 4 + [[-12.0, 12.0]] * 3,
    dtype=float,
)


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    return mujoco.mj_id2name(model, obj_type, obj_id) or ""


def _actuator_names(model: mujoco.MjModel) -> list[str]:
    return [_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]


def _joint_names(model: mujoco.MjModel) -> list[str]:
    return [_name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]


def inspect_model() -> None:
    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Torque model XML not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))

    print(f"model.nq = {model.nq}")
    print(f"model.nv = {model.nv}")
    print(f"model.nu = {model.nu}")

    print("\nJoints:")
    for idx, name in enumerate(_joint_names(model)):
        print(
            f"  {idx}: {name} "
            f"qposadr={model.jnt_qposadr[idx]} dofadr={model.jnt_dofadr[idx]}"
        )

    print("\njoint1~joint7 addresses:")
    for name in ARM_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Missing joint: {name}")
        print(
            f"  {name}: qposadr={model.jnt_qposadr[joint_id]}, "
            f"dofadr={model.jnt_dofadr[joint_id]}"
        )

    print("\nActuators:")
    for idx, name in enumerate(_actuator_names(model)):
        print(
            f"  {idx}: {name} "
            f"trntype={int(model.actuator_trntype[idx])} "
            f"trnid={model.actuator_trnid[idx].tolist()} "
            f"gear={model.actuator_gear[idx].tolist()} "
            f"ctrlrange={model.actuator_ctrlrange[idx].tolist()} "
            f"forcerange={model.actuator_forcerange[idx].tolist()} "
            f"gainprm={model.actuator_gainprm[idx].tolist()} "
            f"biasprm={model.actuator_biasprm[idx].tolist()}"
        )

    actuator_names = _actuator_names(model)
    if actuator_names[:7] != [f"torque{i}" for i in range(1, 8)]:
        raise RuntimeError(f"First 7 actuators are not torque1~torque7: {actuator_names[:7]}")

    for idx, joint_name in enumerate(ARM_JOINTS):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if int(model.actuator_trntype[idx]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise RuntimeError(f"{actuator_names[idx]} is not a joint actuator")
        if int(model.actuator_trnid[idx, 0]) != joint_id:
            raise RuntimeError(f"{actuator_names[idx]} does not target {joint_name}")
        if not np.allclose(model.actuator_gear[idx, 0], 1.0):
            raise RuntimeError(f"{actuator_names[idx]} does not have gear=1")
        if not np.allclose(model.actuator_ctrlrange[idx], EXPECTED_LIMITS[idx]):
            raise RuntimeError(
                f"{actuator_names[idx]} ctrlrange mismatch: {model.actuator_ctrlrange[idx]}"
            )
        if not np.allclose(model.actuator_forcerange[idx], EXPECTED_LIMITS[idx]):
            raise RuntimeError(
                f"{actuator_names[idx]} forcerange mismatch: {model.actuator_forcerange[idx]}"
            )
        if int(model.actuator_ctrllimited[idx]) != 1:
            raise RuntimeError(f"{actuator_names[idx]} ctrl limit is not enabled")
        if int(model.actuator_forcelimited[idx]) != 1:
            raise RuntimeError(f"{actuator_names[idx]} force limit is not enabled")
        if not np.allclose(model.actuator_gainprm[idx, 0], 1.0):
            raise RuntimeError(f"{actuator_names[idx]} fixed gain is not 1")
        if not np.allclose(model.actuator_biasprm[idx], 0.0):
            raise RuntimeError(
                f"{actuator_names[idx]} still has nonzero servo biasprm: "
                f"{model.actuator_biasprm[idx]}"
            )

    print("\n[CHECK] joint1~joint7 are torque motors")


if __name__ == "__main__":
    try:
        inspect_model()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
