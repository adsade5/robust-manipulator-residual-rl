from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.cartesian_utils import orientation_error, rotation_matrix_from_axis_angle


def main() -> None:
    identity = np.eye(3)
    np.testing.assert_allclose(orientation_error(identity, identity), np.zeros(3), atol=1e-12)

    angle = 1e-3
    for axis_index, axis_name in enumerate(["X", "Y", "Z"]):
        axis = np.zeros(3)
        axis[axis_index] = 1.0
        r_des = rotation_matrix_from_axis_angle(axis, angle)
        err = orientation_error(r_des, identity)
        expected = angle * axis
        np.testing.assert_allclose(err, expected, rtol=1e-5, atol=1e-9)

        err_opposite = orientation_error(identity, r_des)
        np.testing.assert_allclose(err_opposite, -expected, rtol=1e-5, atol=1e-9)
        print(f"{axis_name} small-angle orientation error sign passed: {err}")

    print("Orientation error tests passed.")


if __name__ == "__main__":
    main()
