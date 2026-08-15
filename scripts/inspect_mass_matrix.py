from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
RESULT_DIR = PROJECT_ROOT / "results" / "computed_torque"
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])
HOME_CTRL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0])


def _arm_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qposadr = []
    dofadr = []
    for index in range(1, 8):
        name = f"joint{index}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Missing joint: {name}")
        qposadr.append(int(model.jnt_qposadr[joint_id]))
        dofadr.append(int(model.jnt_dofadr[joint_id]))
    return np.array(qposadr, dtype=int), np.array(dofadr, dtype=int)


def inspect_mass_matrix() -> dict[str, object]:
    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Torque model XML not found: {MODEL_XML}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    _, arm_dofadr = _arm_addresses(model)

    data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    data.qvel[:] = 0.0
    data.ctrl[: len(HOME_CTRL)] = HOME_CTRL
    mujoco.mj_forward(model, data)

    full_m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full_m)
    m_arm = full_m[np.ix_(arm_dofadr, arm_dofadr)]

    symmetry_error = float(np.linalg.norm(m_arm - m_arm.T))
    eigenvalues = np.linalg.eigvalsh(0.5 * (m_arm + m_arm.T))
    min_eigenvalue = float(np.min(eigenvalues))
    max_eigenvalue = float(np.max(eigenvalues))
    condition_number = float(np.linalg.cond(m_arm))

    if symmetry_error > 1e-9:
        raise RuntimeError(f"M_arm is not approximately symmetric: ||M-M.T||={symmetry_error}")
    if min_eigenvalue <= 0.0:
        raise RuntimeError(f"M_arm is not positive definite: min eigenvalue={min_eigenvalue}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = RESULT_DIR / "mass_matrix_home.npy"
    np.save(npy_path, m_arm)

    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(m_arm, cmap="viridis")
    axis.set_title("Panda home arm inertia matrix M(q)")
    axis.set_xlabel("joint dof")
    axis.set_ylabel("joint dof")
    axis.set_xticks(np.arange(7))
    axis.set_yticks(np.arange(7))
    axis.set_xticklabels([str(i) for i in range(1, 8)])
    axis.set_yticklabels([str(i) for i in range(1, 8)])
    fig.colorbar(image, ax=axis, label="inertia")
    fig.tight_layout()
    png_path = RESULT_DIR / "mass_matrix_home.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"full M shape = {full_m.shape}")
    print("M_arm =")
    print(np.array2string(m_arm, precision=8, suppress_small=False))
    print(f"symmetry error ||M-M.T|| = {symmetry_error:.12e}")
    print(f"eigenvalues = {np.array2string(eigenvalues, precision=12)}")
    print(f"minimum eigenvalue = {min_eigenvalue:.12e}")
    print(f"maximum eigenvalue = {max_eigenvalue:.12e}")
    print(f"condition number = {condition_number:.12e}")
    print(f"saved = {npy_path}")
    print(f"saved = {png_path}")

    return {
        "symmetry_error": symmetry_error,
        "eigenvalues": eigenvalues.tolist(),
        "min_eigenvalue": min_eigenvalue,
        "max_eigenvalue": max_eigenvalue,
        "condition_number": condition_number,
    }


if __name__ == "__main__":
    try:
        inspect_mass_matrix()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
