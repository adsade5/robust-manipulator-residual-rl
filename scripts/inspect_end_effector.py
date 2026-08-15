from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])
EE_SITE_NAME = "attachment_site"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    site_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or "" for i in range(model.nsite)]
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]

    print("All site names:")
    for index, name in enumerate(site_names):
        print(f"  {index}: {name}")
    print("\nAll body names:")
    for index, name in enumerate(body_names):
        print(f"  {index}: {name}")

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    if site_id < 0:
        raise RuntimeError(f"Required EE site '{EE_SITE_NAME}' not found. Available sites: {site_names}")

    data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    body_id = int(model.site_bodyid[site_id])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])

    print(f"\nEE site name = {EE_SITE_NAME}")
    print(f"site id = {site_id}")
    print(f"body id = {body_id} ({body_names[body_id]})")
    print(f"home EE position = {np.array2string(data.site_xpos[site_id], precision=8)}")
    print("home EE rotation matrix =")
    print(np.array2string(data.site_xmat[site_id].reshape(3, 3), precision=8))
    print(f"home EE quaternion = {np.array2string(quat, precision=8)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
