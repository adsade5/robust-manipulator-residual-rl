from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "cartesian_impedance"
STIFFNESSES = ["low", "medium", "high"]
LABELS = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}


def _load(stiffness: str) -> dict[str, object]:
    path = RESULT_DIR / f"disturbance_{stiffness}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_recovery(value: object) -> str:
    if value is None:
        return "not recovered"
    return f"{float(value):.4f}"


def main() -> None:
    metrics = {name: _load(name) for name in STIFFNESSES}
    print(f"{'':28s}{'LOW':>14s}{'MEDIUM':>14s}{'HIGH':>14s}")
    rows = [
        ("Kx", lambda m: f"{m['k_pos'][0]:.1f}"),
        ("steady dx", lambda m: f"{m['steady_x_displacement_abs']:.6f}"),
        ("F/K theoretical dx", lambda m: f"{m['theoretical_x_displacement']:.6f}"),
        ("max displacement", lambda m: f"{m['maximum_displacement']:.6f}"),
        ("recovery time", lambda m: _fmt_recovery(m["recovery_time"])),
        ("max torque", lambda m: f"{m['max_joint_torque']:.6f}"),
        ("clipping", lambda m: str(sum(int(x) for x in m["clip_counts"]))),
    ]
    for label, getter in rows:
        print(f"{label:28s}" + "".join(f"{getter(metrics[name]):>14s}" for name in STIFFNESSES))

    steady = [float(metrics[name]["steady_x_displacement_abs"]) for name in STIFFNESSES]
    monotonic = steady[0] > steady[1] > steady[2]
    print()
    if monotonic:
        print("Observed higher stiffness -> smaller steady X displacement.")
    else:
        print("Did not observe a strictly monotonic stiffness/displacement relationship.")
        print("Possible causes include 6D coupling, redundancy, damping, Jacobian changes, and torque limits.")


if __name__ == "__main__":
    main()
