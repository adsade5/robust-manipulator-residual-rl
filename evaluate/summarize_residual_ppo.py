from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "results" / "residual_ppo" / "evaluation"
SCALES = [1.00, 1.25, 1.50, 1.75, 2.00]


def _safe(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def _load(scale: float, controller: str) -> dict[str, object]:
    path = EVAL_DIR / f"scale_{_safe(scale)}_{controller}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _classify(abs_improvement: float, baseline: float) -> str:
    tolerance = max(1e-5, 0.02 * baseline)
    if abs_improvement > tolerance:
        return "improved"
    if abs_improvement < -tolerance:
        return "degraded"
    return "similar"


def main() -> None:
    print("Scale    Controller    Overall RMSE    Motion RMSE    Max Error    Torque RMS    Residual RMS")
    for scale in SCALES:
        for controller in ["ctc", "rl"]:
            metrics = _load(scale, controller)
            name = "CTC" if controller == "ctc" else "CTC+RL"
            print(
                f"{scale:<8.2f}{name:<14s}"
                f"{metrics['overall_rmse']:14.8f}{metrics['motion_rmse']:14.8f}"
                f"{metrics['max_tracking_error']:13.8f}{metrics['torque_rms']:14.8f}"
                f"{metrics['residual_torque_rms']:14.8f}"
            )
        ctc = _load(scale, "ctc")
        rl = _load(scale, "rl")
        abs_improvement = ctc["motion_rmse"] - rl["motion_rmse"]
        rel_improvement = 100.0 * abs_improvement / ctc["motion_rmse"] if ctc["motion_rmse"] > 0 else 0.0
        classification = _classify(abs_improvement, ctc["motion_rmse"])
        print(
            f"  change scale {scale:.2f}: abs={abs_improvement:.8f}, "
            f"relative={rel_improvement:.2f}%, {classification}"
        )


if __name__ == "__main__":
    main()
