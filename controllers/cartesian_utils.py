from __future__ import annotations

import numpy as np


def position_error(p_des: np.ndarray, p: np.ndarray) -> np.ndarray:
    p_des = np.asarray(p_des, dtype=float)
    p = np.asarray(p, dtype=float)
    if p_des.shape != (3,) or p.shape != (3,):
        raise ValueError(f"p_des and p must have shape (3,), got {p_des.shape}, {p.shape}")
    if not np.all(np.isfinite(p_des)) or not np.all(np.isfinite(p)):
        raise ValueError("position contains NaN or Inf")
    return p_des - p


def _skew_to_vec(matrix: np.ndarray) -> np.ndarray:
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation contains NaN or Inf")

    cos_theta = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    skew_part = 0.5 * (rotation - rotation.T)
    vee = _skew_to_vec(skew_part)

    if theta < 1e-8:
        return vee
    if np.pi - theta < 1e-5:
        axis = np.empty(3)
        diag = np.diag(rotation)
        axis_index = int(np.argmax(diag))
        axis[axis_index] = np.sqrt(max(0.0, (diag[axis_index] + 1.0) * 0.5))
        denom = 2.0 * max(axis[axis_index], 1e-12)
        if axis_index == 0:
            axis[1] = (rotation[0, 1] + rotation[1, 0]) / denom
            axis[2] = (rotation[0, 2] + rotation[2, 0]) / denom
        elif axis_index == 1:
            axis[0] = (rotation[0, 1] + rotation[1, 0]) / denom
            axis[2] = (rotation[1, 2] + rotation[2, 1]) / denom
        else:
            axis[0] = (rotation[0, 2] + rotation[2, 0]) / denom
            axis[1] = (rotation[1, 2] + rotation[2, 1]) / denom
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.zeros(3)
        return theta * axis / norm

    return theta / np.sin(theta) * vee


def orientation_error(r_des: np.ndarray, r_current: np.ndarray) -> np.ndarray:
    r_des = np.asarray(r_des, dtype=float)
    r_current = np.asarray(r_current, dtype=float)
    if r_des.shape != (3, 3) or r_current.shape != (3, 3):
        raise ValueError(
            f"r_des and r_current must have shape (3, 3), got {r_des.shape}, {r_current.shape}"
        )
    if not np.all(np.isfinite(r_des)) or not np.all(np.isfinite(r_current)):
        raise ValueError("orientation contains NaN or Inf")
    r_err = r_des @ r_current.T
    return rotation_matrix_to_rotvec(r_err)


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,):
        raise ValueError(f"axis must have shape (3,), got {axis.shape}")
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
