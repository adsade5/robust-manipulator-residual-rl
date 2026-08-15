from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "joint_tracking"


def _load_metrics(controller: str) -> dict[str, object]:
    path = RESULT_DIR / f"{controller}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    pd = _load_metrics("pd")
    pd_gc = _load_metrics("pd_gc")

    pd_rmse = float(pd["overall_rmse"])
    pd_gc_rmse = float(pd_gc["overall_rmse"])
    rmse_difference = pd_rmse - pd_gc_rmse
    improvement = 100.0 * rmse_difference / pd_rmse if pd_rmse > 0 else 0.0

    print("Joint-space trajectory tracking")
    print()
    print(f"{'':18s}{'PD':>14s}{'PD+GC':>14s}")
    print(f"{'overall RMSE':18s}{pd_rmse:14.8f}{pd_gc_rmse:14.8f}")
    print(
        f"{'max error':18s}"
        f"{float(pd['overall_max_abs_error']):14.8f}"
        f"{float(pd_gc['overall_max_abs_error']):14.8f}"
    )
    print(
        f"{'max qvel':18s}"
        f"{float(pd['max_abs_qvel']):14.8f}"
        f"{float(pd_gc['max_abs_qvel']):14.8f}"
    )
    print(
        f"{'max torque':18s}"
        f"{float(pd['max_abs_tau']):14.8f}"
        f"{float(pd_gc['max_abs_tau']):14.8f}"
    )
    print()
    print(f"overall RMSE difference (PD - PD+GC): {rmse_difference:.8f}")
    print(f"RMSE improvement percentage: {improvement:.2f}%")
    if improvement < 0.0:
        print("PD+GC did not outperform PD in this experiment.")


if __name__ == "__main__":
    main()
