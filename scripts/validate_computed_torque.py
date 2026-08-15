from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.computed_torque import computed_torque_control
from experiments.computed_torque_tracking import KD_ACC, KP_ACC
from experiments.joint_trajectory_tracking import HOME_CTRL, HOME_QPOS, desired_state


MODEL_XML = PROJECT_ROOT / "models" / "franka_emika_panda_torque" / "panda.xml"
RESULT_DIR = PROJECT_ROOT / "results" / "computed_torque"


def _arm_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qposadr = []
    dofadr = []
    for index in range(1, 8):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}")
        if joint_id < 0:
            raise RuntimeError(f"Missing joint{index}")
        qposadr.append(int(model.jnt_qposadr[joint_id]))
        dofadr.append(int(model.jnt_dofadr[joint_id]))
    return np.array(qposadr, dtype=int), np.array(dofadr, dtype=int)


def main() -> None:
    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Torque model XML not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    _, arm_dofadr = _arm_addresses(model)

    data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    data.qvel[:] = 0.0
    data.ctrl[: len(HOME_CTRL)] = HOME_CTRL

    # Validate away from the trivial home state so M(q), bias, and passive terms are nonzero.
    q_des, qdot_des, qddot_des, _ = desired_state(2.35, 1.0)
    data.qpos[arm_dofadr] = q_des - np.array([0.01, -0.015, 0.008, -0.006, 0.004, -0.005, 0.003])
    data.qvel[arm_dofadr] = qdot_des - np.array([0.03, -0.02, 0.015, -0.01, 0.008, -0.006, 0.004])
    mujoco.mj_forward(model, data)

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

    a_full = np.zeros(model.nv)
    a_full[arm_dofadr] = output.a_cmd
    inertial_full = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, inertial_full, a_full)
    tau_model_arm = (
        inertial_full[arm_dofadr]
        + data.qfrc_bias[arm_dofadr]
        - data.qfrc_passive[arm_dofadr]
    )
    manual_diff = float(np.max(np.abs(tau_model_arm - output.tau_unclipped)))

    inverse_data = mujoco.MjData(model)
    inverse_data.qpos[:] = data.qpos
    inverse_data.qvel[:] = data.qvel
    inverse_data.ctrl[:] = data.ctrl
    mujoco.mj_forward(model, inverse_data)
    inverse_data.qacc[:] = a_full
    mujoco.mj_inverse(model, inverse_data)
    inverse_arm = inverse_data.qfrc_inverse[arm_dofadr].copy()
    inverse_diff = float(np.max(np.abs(inverse_arm - output.tau_unclipped)))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / "inverse_dynamics_validation.txt"
    output_path.write_text(
        "\n".join(
            [
                f"manual_max_abs_diff={manual_diff:.12e}",
                f"inverse_dynamics_max_abs_diff={inverse_diff:.12e}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"manual M a + bias - passive max abs diff = {manual_diff:.12e}")
    print(f"mj_inverse max abs diff = {inverse_diff:.12e}")
    print(f"saved = {output_path}")
    if manual_diff > 1e-9:
        raise RuntimeError(f"Manual torque reconstruction mismatch: {manual_diff:.12e}")
    if inverse_diff > 1e-6:
        raise RuntimeError(f"mj_inverse torque mismatch: {inverse_diff:.12e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
