from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "results" / "residual_ppo_v3" / "evaluation"
SCALES = [1.00, 1.25, 1.50, 1.75, 2.00]


def _safe(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def _load(scale: float, controller: str) -> dict[str, object]:
    return json.loads((EVAL_DIR / f"scale_{_safe(scale)}_{controller}_metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    print("Residual PPO v1 vs v2 vs v3 Motion RMSE")
    print("Scale    CTC Motion    PPO v1 Motion    PPO v2 Motion    PPO v3 Motion")
    print("-" * 76)
    for scale in SCALES:
        ctc = _load(scale, "ctc")
        v1 = _load(scale, "ppo_v1")
        v2 = _load(scale, "ppo_v2")
        v3 = _load(scale, "ppo_v3")
        print(
            f"{scale:<8.2f}"
            f"{float(ctc['motion_rmse']):<14.8f}"
            f"{float(v1['motion_rmse']):<17.8f}"
            f"{float(v2['motion_rmse']):<17.8f}"
            f"{float(v3['motion_rmse']):<.8f}"
        )

    print("\nPPO v3 full metrics")
    print(
        "Scale    Overall RMSE    Motion RMSE     Max Error       Torque RMS      "
        "Residual RMS    Max Residual    Action RMS    Torque Clip    Action Sat"
    )
    print("-" * 142)
    for scale in SCALES:
        v3 = _load(scale, "ppo_v3")
        print(
            f"{scale:<8.2f}"
            f"{float(v3['overall_rmse']):<16.8f}"
            f"{float(v3['motion_rmse']):<16.8f}"
            f"{float(v3['max_tracking_error']):<16.8f}"
            f"{float(v3['torque_rms']):<16.6f}"
            f"{float(v3['residual_torque_rms']):<16.6f}"
            f"{float(v3['max_residual_torque']):<16.6f}"
            f"{float(v3['action_rms']):<14.6f}"
            f"{int(v3['total_torque_clipping_count']):<15d}"
            f"{int(v3['residual_action_clipping_count'])}"
        )


if __name__ == "__main__":
    main()
