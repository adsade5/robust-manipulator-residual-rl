from __future__ import annotations

import numpy as np


def quintic_joint_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    duration: float,
    t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 7-DOF quintic point-to-point desired state at time t."""
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    if q_start.shape != (7,):
        raise ValueError(f"q_start must have shape (7,), got {q_start.shape}")
    if q_goal.shape != (7,):
        raise ValueError(f"q_goal must have shape (7,), got {q_goal.shape}")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"duration must be a positive finite scalar, got {duration}")
    if not np.isfinite(t):
        raise ValueError(f"t must be finite, got {t}")
    if not np.all(np.isfinite(q_start)):
        raise ValueError("q_start contains NaN or Inf")
    if not np.all(np.isfinite(q_goal)):
        raise ValueError("q_goal contains NaN or Inf")

    zero = np.zeros(7)
    if t <= 0.0:
        return q_start.copy(), zero.copy(), zero.copy()
    if t >= duration:
        return q_goal.copy(), zero.copy(), zero.copy()

    s = t / duration
    dq = q_goal - q_start

    h = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    hdot = 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4
    hddot = 60.0 * s - 180.0 * s**2 + 120.0 * s**3

    q_des = q_start + h * dq
    qdot_des = (hdot / duration) * dq
    qddot_des = (hddot / duration**2) * dq
    return q_des, qdot_des, qddot_des
