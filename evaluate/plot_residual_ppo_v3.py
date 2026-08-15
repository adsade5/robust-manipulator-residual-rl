from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "residual_ppo_v3"
EVAL_DIR = RESULT_DIR / "evaluation"
FIG_DIR = RESULT_DIR / "figures"
SCALES = [1.00, 1.25, 1.50, 1.75, 2.00]
REP_JOINTS = [1, 3, 5]


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
    fig, axis = plt.subplots(figsize=(9, 5))
    for controller, label, marker in [
        ("ctc", "CTC", "o"),
        ("ppo_v1", "Residual PPO v1", "s"),
        ("ppo_v2", "Residual PPO v2", "^"),
        ("ppo_v3", "Residual PPO v3", "D"),
    ]:
        axis.plot(SCALES, [_metrics(scale, controller)["motion_rmse"] for scale in SCALES], marker=marker, label=label)
    axis.set_xlabel("inertial scale")
    axis.set_ylabel("Motion RMSE [rad]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    out = FIG_DIR / "rmse_vs_inertial_scale_v1_v2_v3.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def plot_tracking_error(scale: float, name: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    for controller, label in [
        ("ctc", "CTC"),
        ("ppo_v1", "Residual PPO v1"),
        ("ppo_v2", "Residual PPO v2"),
        ("ppo_v3", "Residual PPO v3"),
    ]:
        data = _arrays(scale, controller)
        axis.plot(data["time"], np.linalg.norm(data["q_error"], axis=1), label=label)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("tracking error norm [rad]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    out = FIG_DIR / name
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def plot_residual_torque(scale: float, name: str) -> None:
    data_by_controller = {
        "v1": _arrays(scale, "ppo_v1"),
        "v2": _arrays(scale, "ppo_v2"),
        "v3": _arrays(scale, "ppo_v3"),
    }
    fig, axes = plt.subplots(len(REP_JOINTS), 1, figsize=(11, 8), sharex=True)
    for axis, joint_idx in zip(axes, REP_JOINTS):
        for label, data in data_by_controller.items():
            axis.plot(data["time"], data["delta_tau_rl"][:, joint_idx], label=f"{label} delta_tau_RL", linewidth=1.2)
        axis.set_ylabel(f"joint{joint_idx + 1} [Nm]")
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    out = FIG_DIR / name
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def plot_residual_rms_vs_scale() -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for controller, label, marker in [
        ("ppo_v1", "Residual PPO v1", "s"),
        ("ppo_v2", "Residual PPO v2", "^"),
        ("ppo_v3", "Residual PPO v3", "D"),
    ]:
        axis.plot(SCALES, [_metrics(scale, controller)["residual_torque_rms"] for scale in SCALES], marker=marker, label=label)
    axis.set_xlabel("inertial scale")
    axis.set_ylabel("Residual torque RMS [Nm]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    out = FIG_DIR / "residual_rms_vs_scale.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_rmse_vs_scale()
    plot_tracking_error(1.00, "nominal_tracking_error.png")
    plot_residual_torque(1.00, "nominal_residual_torque.png")
    plot_residual_torque(1.00, "nominal_residual_v1_v2_v3.png")
    plot_tracking_error(1.50, "mismatch_150_tracking_error.png")
    plot_residual_torque(1.50, "mismatch_150_residual_torque.png")
    plot_residual_rms_vs_scale()


if __name__ == "__main__":
    main()
