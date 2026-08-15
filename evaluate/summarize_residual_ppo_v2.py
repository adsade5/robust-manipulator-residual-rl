from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "results" / "residual_ppo_v2" / "evaluation"
SCALES = [1.00, 1.25, 1.50, 1.75, 2.00]


def _safe(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def _load(scale: float, controller: str) -> dict[str, object]:
    return json.loads((EVAL_DIR / f"scale_{_safe(scale)}_{controller}_metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    print("Residual PPO v1 vs v2 Motion RMSE")
    print("Scale    CTC Motion    PPO v1 Motion    PPO v2 Motion    v2-CTC       v2 rel change    v2-v1")
    print("-" * 102)
    for scale in SCALES:
        ctc = _load(scale, "ctc")
        v1 = _load(scale, "ppo_v1")
        v2 = _load(scale, "ppo_v2")
        ctc_motion = float(ctc["motion_rmse"])
        v1_motion = float(v1["motion_rmse"])
        v2_motion = float(v2["motion_rmse"])
        v2_vs_ctc = v2_motion - ctc_motion
        v2_vs_v1 = v2_motion - v1_motion
        rel = 100.0 * v2_vs_ctc / ctc_motion if ctc_motion > 0 else 0.0
        rel_text = f"{rel:+.2f}%".ljust(17)
        print(
            f"{scale:<8.2f}"
            f"{ctc_motion:<14.8f}"
            f"{v1_motion:<17.8f}"
            f"{v2_motion:<17.8f}"
            f"{v2_vs_ctc:<13.8f}"
            f"{rel_text}"
            f"{v2_vs_v1:<.8f}"
        )


if __name__ == "__main__":
    main()
