from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PDGC = PROJECT_ROOT / "results" / "joint_tracking"
RESULT_CTC = PROJECT_ROOT / "results" / "computed_torque"
REPRESENTATIVE_JOINTS = [0, 1, 3]


def _load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return np.load(path)


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_key(data: np.lib.npyio.NpzFile, key: str, label: str) -> np.ndarray:
    if key not in data:
        raise RuntimeError(f"{label} is missing required metadata key: {key}")
    return data[key]


def _validate_same_experiment(pdgc: np.lib.npyio.NpzFile, ctc: np.lib.npyio.NpzFile) -> None:
    checks = [
        ("q_home", "q_home"),
        ("q_target", "q_target"),
        ("trajectory_schedule", "trajectory timing"),
        ("torque_limits", "torque limits"),
    ]
    for key, label in checks:
        left = _require_key(pdgc, key, "PD+GC")
        right = _require_key(ctc, key, "CTC")
        if not np.allclose(left, right):
            raise RuntimeError(f"Mismatch in {label}: PD+GC={left}, CTC={right}")

    for key, label in [("total_duration", "duration"), ("timestep", "simulation timestep")]:
        left = float(_require_key(pdgc, key, "PD+GC"))
        right = float(_require_key(ctc, key, "CTC"))
        if not np.isclose(left, right):
            raise RuntimeError(f"Mismatch in {label}: PD+GC={left}, CTC={right}")

    if not np.allclose(pdgc["time"], ctc["time"]):
        raise RuntimeError("Time arrays differ; refusing to compare misaligned experiments")
    if not np.allclose(pdgc["q_des"], ctc["q_des"]):
        raise RuntimeError("Desired trajectories differ; refusing to compare")
    if not np.allclose(pdgc["qdot_des"], ctc["qdot_des"]):
        raise RuntimeError("Desired velocity trajectories differ; refusing to compare")
    if not np.allclose(pdgc["qddot_des"], ctc["qddot_des"]):
        raise RuntimeError("Desired acceleration trajectories differ; refusing to compare")


def _plot_positions(pdgc: np.lib.npyio.NpzFile, ctc: np.lib.npyio.NpzFile) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, joint_idx in zip(axes, REPRESENTATIVE_JOINTS):
        axis.plot(ctc["time"], ctc["q_des"][:, joint_idx], color="black", linewidth=2.0, label="desired")
        axis.plot(pdgc["time"], pdgc["q"][:, joint_idx], linewidth=1.5, label="PD+GC")
        axis.plot(ctc["time"], ctc["q"][:, joint_idx], linewidth=1.5, label="CTC")
        axis.set_ylabel(f"joint{joint_idx + 1} [rad]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()
    output_path = RESULT_CTC / "pdgc_vs_ctc_position.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def _plot_errors(pdgc: np.lib.npyio.NpzFile, ctc: np.lib.npyio.NpzFile) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, joint_idx in zip(axes, REPRESENTATIVE_JOINTS):
        axis.plot(pdgc["time"], pdgc["tracking_error"][:, joint_idx], linewidth=1.5, label="PD+GC error")
        axis.plot(ctc["time"], ctc["tracking_error"][:, joint_idx], linewidth=1.5, label="CTC error")
        axis.set_ylabel(f"joint{joint_idx + 1} error [rad]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()
    output_path = RESULT_CTC / "pdgc_vs_ctc_error.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def _plot_rmse(pdgc_metrics: dict[str, object], ctc_metrics: dict[str, object]) -> Path:
    joints = np.arange(1, 8)
    width = 0.36
    pdgc_rmse = np.array(pdgc_metrics["per_joint_rmse"], dtype=float)
    ctc_rmse = np.array(ctc_metrics["per_joint_rmse"], dtype=float)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(joints - width / 2, pdgc_rmse, width=width, label="PD+GC")
    axis.bar(joints + width / 2, ctc_rmse, width=width, label="CTC")
    axis.set_xlabel("joint")
    axis.set_ylabel("position RMSE [rad]")
    axis.set_xticks(joints)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(loc="best")
    axis.set_title(
        f"Overall RMSE: PD+GC={pdgc_metrics['overall_rmse']:.5f}, "
        f"CTC={ctc_metrics['overall_rmse']:.5f}"
    )
    fig.tight_layout()
    output_path = RESULT_CTC / "pdgc_vs_ctc_rmse.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def _plot_torque_components(ctc: np.lib.npyio.NpzFile) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, joint_idx in zip(axes, REPRESENTATIVE_JOINTS):
        inertial = ctc["tau_inertial"][:, joint_idx]
        bias = ctc["qfrc_bias"][:, joint_idx]
        negative_passive = -ctc["qfrc_passive"][:, joint_idx]
        total = ctc["tau_unclipped"][:, joint_idx]
        axis.plot(ctc["time"], inertial, linewidth=1.4, label="M(q)a_cmd")
        axis.plot(ctc["time"], bias, linewidth=1.4, label="qfrc_bias")
        axis.plot(ctc["time"], negative_passive, linewidth=1.4, label="-qfrc_passive")
        axis.plot(ctc["time"], total, linewidth=1.8, color="black", label="total tau")
        axis.set_ylabel(f"joint{joint_idx + 1} [Nm]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="best", ncol=2)
    fig.tight_layout()
    output_path = RESULT_CTC / "ctc_torque_components.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def _plot_acceleration_tracking(ctc: np.lib.npyio.NpzFile) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, joint_idx in zip(axes, REPRESENTATIVE_JOINTS):
        axis.plot(ctc["time"], ctc["a_cmd"][:, joint_idx], linewidth=1.5, label="a_cmd")
        axis.plot(ctc["time"], ctc["qacc"][:, joint_idx], linewidth=1.5, label="actual qacc")
        axis.set_ylabel(f"joint{joint_idx + 1} [rad/s^2]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()
    output_path = RESULT_CTC / "acceleration_tracking.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def main() -> None:
    RESULT_CTC.mkdir(parents=True, exist_ok=True)
    pdgc = _load_npz(RESULT_PDGC / "pd_gc.npz")
    ctc = _load_npz(RESULT_CTC / "ctc.npz")
    pdgc_metrics = _load_json(RESULT_PDGC / "pd_gc_metrics.json")
    ctc_metrics = _load_json(RESULT_CTC / "ctc_metrics.json")

    try:
        _validate_same_experiment(pdgc, ctc)
        _plot_positions(pdgc, ctc)
        _plot_errors(pdgc, ctc)
        _plot_rmse(pdgc_metrics, ctc_metrics)
        _plot_torque_components(ctc)
        _plot_acceleration_tracking(ctc)
    finally:
        pdgc.close()
        ctc.close()


if __name__ == "__main__":
    main()
