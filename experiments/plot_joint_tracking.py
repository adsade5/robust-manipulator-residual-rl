from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "joint_tracking"
REPRESENTATIVE_JOINTS = [0, 1, 3]


def _load_npz(mode: str) -> np.lib.npyio.NpzFile | None:
    path = RESULT_DIR / f"{mode}.npz"
    if not path.exists():
        print(f"Missing {path}; skipping data that depends on {mode}.")
        return None
    return np.load(path)


def _load_metrics(mode: str) -> dict[str, object] | None:
    path = RESULT_DIR / f"{mode}_metrics.json"
    if not path.exists():
        print(f"Missing {path}; RMSE summary may be incomplete.")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def plot_joint_position_tracking(pd: np.lib.npyio.NpzFile, pd_gc: np.lib.npyio.NpzFile) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, joint_idx in zip(axes, REPRESENTATIVE_JOINTS):
        axis.plot(pd["time"], pd["q_des"][:, joint_idx], color="black", linewidth=2.0, label="desired")
        axis.plot(pd["time"], pd["q"][:, joint_idx], linewidth=1.5, label="PD actual")
        axis.plot(pd_gc["time"], pd_gc["q"][:, joint_idx], linewidth=1.5, label="PD+GC actual")
        axis.set_ylabel(f"joint{joint_idx + 1} [rad]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()
    output_path = RESULT_DIR / "joint_position_tracking.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def plot_tracking_error(pd: np.lib.npyio.NpzFile, pd_gc: np.lib.npyio.NpzFile) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, joint_idx in zip(axes, REPRESENTATIVE_JOINTS):
        axis.plot(pd["time"], pd["tracking_error"][:, joint_idx], linewidth=1.5, label="PD error")
        axis.plot(pd_gc["time"], pd_gc["tracking_error"][:, joint_idx], linewidth=1.5, label="PD+GC error")
        axis.set_ylabel(f"joint{joint_idx + 1} error [rad]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()
    output_path = RESULT_DIR / "tracking_error.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def plot_rmse_comparison(pd_metrics: dict[str, object], pd_gc_metrics: dict[str, object]) -> Path:
    joints = np.arange(1, 8)
    width = 0.36
    pd_rmse = np.array(pd_metrics["per_joint_rmse"], dtype=float)
    pd_gc_rmse = np.array(pd_gc_metrics["per_joint_rmse"], dtype=float)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(joints - width / 2, pd_rmse, width=width, label="PD")
    axis.bar(joints + width / 2, pd_gc_rmse, width=width, label="PD+GC")
    axis.set_xlabel("joint")
    axis.set_ylabel("position RMSE [rad]")
    axis.set_xticks(joints)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(loc="best")
    axis.set_title(
        f"Overall RMSE: PD={pd_metrics['overall_rmse']:.5f}, "
        f"PD+GC={pd_gc_metrics['overall_rmse']:.5f}"
    )
    fig.tight_layout()
    output_path = RESULT_DIR / "rmse_comparison.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")
    print(f"Overall RMSE PD = {pd_metrics['overall_rmse']:.8f}")
    print(f"Overall RMSE PD+GC = {pd_gc_metrics['overall_rmse']:.8f}")
    return output_path


def plot_torque_commands(pd: np.lib.npyio.NpzFile, pd_gc: np.lib.npyio.NpzFile) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, joint_idx in zip(axes, REPRESENTATIVE_JOINTS):
        axis.plot(pd["time"], pd["tau_command"][:, joint_idx], linewidth=1.5, label="PD torque")
        axis.plot(pd_gc["time"], pd_gc["tau_command"][:, joint_idx], linewidth=1.5, label="PD+GC torque")
        axis.set_ylabel(f"joint{joint_idx + 1} [Nm]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()
    output_path = RESULT_DIR / "torque_commands.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd = _load_npz("pd")
    pd_gc = _load_npz("pd_gc")
    pd_metrics = _load_metrics("pd")
    pd_gc_metrics = _load_metrics("pd_gc")

    if pd is not None and pd_gc is not None:
        plot_joint_position_tracking(pd, pd_gc)
        plot_tracking_error(pd, pd_gc)
        plot_torque_commands(pd, pd_gc)
    else:
        print("Need both pd.npz and pd_gc.npz for trajectory comparison plots.")

    if pd_metrics is not None and pd_gc_metrics is not None:
        plot_rmse_comparison(pd_metrics, pd_gc_metrics)
    else:
        print("Need both metrics JSON files for RMSE comparison plot.")

    for data in [pd, pd_gc]:
        if data is not None:
            data.close()


if __name__ == "__main__":
    main()
