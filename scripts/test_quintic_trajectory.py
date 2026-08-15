from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectories.quintic import quintic_joint_trajectory


RESULT_DIR = PROJECT_ROOT / "results" / "trajectory"


def main() -> None:
    q_start = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
    q_goal = q_start + np.array([0.25, -0.20, 0.20, -0.20, 0.15, 0.20, 0.15])
    duration = 3.0

    q0, qdot0, qddot0 = quintic_joint_trajectory(q_start, q_goal, duration, 0.0)
    qT, qdotT, qddotT = quintic_joint_trajectory(q_start, q_goal, duration, duration)

    np.testing.assert_allclose(q0, q_start)
    np.testing.assert_allclose(qdot0, np.zeros(7))
    np.testing.assert_allclose(qddot0, np.zeros(7))
    np.testing.assert_allclose(qT, q_goal)
    np.testing.assert_allclose(qdotT, np.zeros(7))
    np.testing.assert_allclose(qddotT, np.zeros(7))

    times = np.linspace(0.0, duration, 401)
    q_log = []
    qdot_log = []
    qddot_log = []
    for t in times:
        q, qdot, qddot = quintic_joint_trajectory(q_start, q_goal, duration, float(t))
        q_log.append(q)
        qdot_log.append(qdot)
        qddot_log.append(qddot)

    q_log = np.vstack(q_log)
    qdot_log = np.vstack(qdot_log)
    qddot_log = np.vstack(qddot_log)
    for name, values in [("q_des", q_log), ("qdot_des", qdot_log), ("qddot_des", qddot_log)]:
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{name} contains NaN or Inf")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    joint_idx = 0
    axes[0].plot(times, q_log[:, joint_idx], linewidth=1.8)
    axes[0].set_ylabel("q_des [rad]")
    axes[1].plot(times, qdot_log[:, joint_idx], linewidth=1.8)
    axes[1].set_ylabel("qdot_des [rad/s]")
    axes[2].plot(times, qddot_log[:, joint_idx], linewidth=1.8)
    axes[2].set_ylabel("qddot_des [rad/s^2]")
    axes[2].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.suptitle("Quintic trajectory sanity check: joint1")
    fig.tight_layout()

    output_path = RESULT_DIR / "quintic_sanity.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print("Quintic trajectory sanity checks passed.")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
