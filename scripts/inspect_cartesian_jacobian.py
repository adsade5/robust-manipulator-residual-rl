from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])
EE_SITE_NAME = "attachment_site"


def _arm_dofs(model: mujoco.MjModel) -> np.ndarray:
    dofs = []
    for index in range(1, 8):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}")
        if joint_id < 0:
            raise RuntimeError(f"Missing joint{index}")
        dofs.append(int(model.jnt_dofadr[joint_id]))
    return np.array(dofs, dtype=int)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    if site_id < 0:
        raise RuntimeError(f"Required EE site '{EE_SITE_NAME}' not found")
    arm_dofadr = _arm_dofs(model)

    data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    j6 = np.vstack([jacp, jacr])
    j_arm = j6[:, arm_dofadr]
    if not np.all(np.isfinite(j_arm)):
        raise FloatingPointError("J_arm contains NaN or Inf")

    singular_values = np.linalg.svd(j_arm, compute_uv=False)
    rank = int(np.linalg.matrix_rank(j_arm, tol=1e-9))
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 1e-12 else float("inf")

    print(f"J_pos shape = {jacp.shape}")
    print(f"J_rot shape = {jacr.shape}")
    print(f"J6 shape = {j6.shape}")
    print(f"J_arm shape = {j_arm.shape}")
    print("J_arm =")
    print(np.array2string(j_arm, precision=8, suppress_small=False))
    print(f"rank = {rank}")
    print(f"singular values = {np.array2string(singular_values, precision=12)}")
    print(f"condition information = {condition:.12e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
