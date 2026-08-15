from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "residual_ppo"
EVAL_DIR = RESULT_DIR / "evaluation"
FIG_DIR = RESULT_DIR / "figures"
SCALES = [1.00, 1.25, 1.50, 1.75, 2.00]
REP_JOINTS = [1, 3, 5]


def _safe(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def _metrics(scale: float, controller: str) -> dict[str, object]:
    return json.loads((EVAL_DIR / f"scale_{_safe(scale)}_{controller}_metrics.json").read_text(encoding="utf-8"))


def _npz(scale: float, controller: str) -> np.lib.npyio.NpzFile:
    return np.load(EVAL_DIR / f"scale_{_safe(scale)}_{controller}.npz")


def plot_training_curve() -> None:
    records_path = RESULT_DIR / "training_eval_log.json"
    if not records_path.exists():
        print(f"Missing {records_path}; skipping training curve")
        return
    records = json.loads(records_path.read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot([r["timesteps"] for r in records], [r["mean_motion_rmse"] for r in records], marker="o")
    axis.set_xlabel("training timestep")
    axis.set_ylabel("evaluation mean motion RMSE")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "training_curve.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def plot_rmse_vs_scale() -> None:
    ctc = [_metrics(s, "ctc")["motion_rmse"] for s in SCALES]
    rl = [_metrics(s, "rl")["motion_rmse"] for s in SCALES]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(SCALES, ctc, marker="o", label="CTC")
    axis.plot(SCALES, rl, marker="s", label="CTC + Residual PPO")
    axis.set_xlabel("inertial scale")
    axis.set_ylabel("motion RMSE [rad]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    out = FIG_DIR / "rmse_vs_inertial_scale.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def plot_tracking_error(scale: float, name: str) -> None:
    ctc = _npz(scale, "ctc")
    rl = _npz(scale, "rl")
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(ctc["time"], np.linalg.norm(ctc["q_error"], axis=1), label="CTC")
    axis.plot(rl["time"], np.linalg.norm(rl["q_error"], axis=1), label="CTC + RL")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("tracking error norm [rad]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    out = FIG_DIR / name
    fig.savefig(out, dpi=160)
    plt.close(fig)
    ctc.close()
    rl.close()
    print(f"Saved {out}")


def plot_residual_torque() -> None:
    data = _npz(1.50, "rl")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, joint_idx in zip(axes, REP_JOINTS):
        axis.plot(data["time"], data["tau_ctc"][:, joint_idx], label="tau_CTC", linewidth=1.3)
        axis.plot(data["time"], data["delta_tau_rl"][:, joint_idx], label="delta_tau_RL", linewidth=1.3)
        axis.plot(data["time"], data["tau_total"][:, joint_idx], label="tau_total", linewidth=1.3)
        axis.set_ylabel(f"joint{joint_idx + 1} [Nm]")
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    out = FIG_DIR / "residual_torque.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    data.close()
    print(f"Saved {out}")


def plot_torque_rms() -> None:
    ctc = [_metrics(s, "ctc")["torque_rms"] for s in SCALES]
    rl = [_metrics(s, "rl")["torque_rms"] for s in SCALES]
    x = np.arange(len(SCALES))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, ctc, width=width, label="CTC")
    axis.bar(x + width / 2, rl, width=width, label="CTC + RL")
    axis.set_xticks(x)
    axis.set_xticklabels([f"{s:.2f}" for s in SCALES])
    axis.set_xlabel("inertial scale")
    axis.set_ylabel("torque RMS [Nm]")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    out = FIG_DIR / "torque_rms_comparison.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_curve()
    plot_rmse_vs_scale()
    plot_tracking_error(1.50, "tracking_error_scale_150.png")
    plot_tracking_error(2.00, "tracking_error_scale_200.png")
    plot_residual_torque()
    plot_torque_rms()


if __name__ == "__main__":
    main()
