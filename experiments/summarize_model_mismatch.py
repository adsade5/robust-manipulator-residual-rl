from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "model_mismatch"
SCENARIOS = [
    ("nominal", "Nominal"),
    ("inertial_mild", "Inertial mild"),
    ("inertial_medium", "Inertial medium"),
    ("inertial_strong", "Inertial strong"),
    ("damping_mild", "Damping mild"),
    ("damping_medium", "Damping medium"),
    ("damping_strong", "Damping strong"),
    ("disturbance", "External disturbance"),
]


def _load(name: str) -> dict[str, object]:
    path = RESULT_DIR / f"{name}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    metrics = {name: _load(name) for name, _ in SCENARIOS}
    nominal_rmse = float(metrics["nominal"]["overall_rmse"])

    print("CTC Robustness Baseline")
    print()
    print(
        f"{'Scenario':24s}{'RMSE':>12s}{'Motion RMSE':>14s}"
        f"{'Max Error':>12s}{'Torque RMS':>14s}{'Deg Ratio':>12s}{'Increase %':>12s}"
    )
    for name, label in SCENARIOS:
        item = metrics[name]
        rmse = float(item["overall_rmse"])
        ratio = rmse / nominal_rmse if nominal_rmse > 0.0 else 0.0
        increase = 100.0 * (ratio - 1.0)
        print(
            f"{label:24s}{rmse:12.8f}{float(item['motion_rmse']):14.8f}"
            f"{float(item['overall_max_error']):12.8f}{float(item['torque_rms']):14.8f}"
            f"{ratio:12.3f}{increase:12.2f}"
        )

    print()
    print("If a mismatch does not degrade performance, it is reported as-is; no perturbation was retuned.")


if __name__ == "__main__":
    main()
