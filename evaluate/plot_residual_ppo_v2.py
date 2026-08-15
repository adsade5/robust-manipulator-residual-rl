from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "residual_ppo_v2"
EVAL_DIR = RESULT_DIR / "evaluation"
FIG_DIR = RESULT_DIR / "figures"
SCALES = [1.00, 1.25, 1.50, 1.75, 2.00]


def _safe(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def _metrics(scale: float, controller: str) -> dict[str, object]:
    return json.loads((EVAL_DIR / f"scale_{_safe(scale)}_{controller}_metrics.json").read_text(encoding="utf-8"))


def _arrays(scale: float, controller: str) -> dict[str, np.ndarray]:
    data = np.load(EVAL_DIR / f"scale_{_safe(scale)}_{controller}.npz")
    try:
        return {key: data[key].copy() for key in data.files}
    finally:
        data.close()


def plot_rmse_vs_scale() -> None:
    ctc = [_metrics(scale, "ctc")["motion_rmse"] for scale in SCALES]
    v1 = [_metrics(scale, "ppo_v1")["motion_rmse"] for scale in SCALES]
    v2 = [_metrics(scale, "ppo_v2")["motion_rmse"] for scale in SCALES]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(SCALES, ctc, marker="o", label="CTC")
    axis.plot(SCALES, v1, marker="s", label="Residual PPO v1")
    axis.plot(SCALES, v2, marker="^", label="Residual PPO v2")
    axis.set_xlabel("inertial scale")
    axis.set_ylabel("motion RMSE [rad]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    out = FIG_DIR / "rmse_vs_inertial_scale_v1_v2.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def plot_tracking_error(scale: float, name: str) -> None:
    ctc = _arrays(scale, "ctc")
    v1 = _arrays(scale, "ppo_v1")
    v2 = _arrays(scale, "ppo_v2")
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(ctc["time"], np.linalg.norm(ctc["q_error"], axis=1), label="CTC")
    axis.plot(v1["time"], np.linalg.norm(v1["q_error"], axis=1), label="Residual PPO v1")
    axis.plot(v2["time"], np.linalg.norm(v2["q_error"], axis=1), label="Residual PPO v2")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("tracking error norm [rad]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    out = FIG_DIR / name
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def plot_residual_norm(scale: float, name: str) -> None:
    v1 = _arrays(scale, "ppo_v1")
    v2 = _arrays(scale, "ppo_v2")
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(v1["time"], np.linalg.norm(v1["delta_tau_rl"], axis=1), label="Residual PPO v1")
    axis.plot(v2["time"], np.linalg.norm(v2["delta_tau_rl"], axis=1), label="Residual PPO v2")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("residual torque norm [Nm]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    out = FIG_DIR / name
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_rmse_vs_scale()
    plot_tracking_error(1.00, "nominal_tracking_error_v1_v2.png")
    plot_residual_norm(1.00, "nominal_residual_torque_v1_v2.png")
    plot_tracking_error(1.50, "tracking_error_scale_150_v1_v2.png")
    plot_residual_norm(1.50, "residual_torque_scale_150_v1_v2.png")


if __name__ == "__main__":
    main()
