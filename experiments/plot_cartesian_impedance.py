from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "cartesian_impedance"
STIFFNESSES = ["low", "medium", "high"]
LABELS = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
REPRESENTATIVE_JOINTS = [0, 1, 3]


def _load_npz(stiffness: str) -> np.lib.npyio.NpzFile | None:
    path = RESULT_DIR / f"disturbance_{stiffness}.npz"
    if not path.exists():
        print(f"Missing {path}")
        return None
    return np.load(path)


def _load_metrics(stiffness: str) -> dict[str, object] | None:
    path = RESULT_DIR / f"disturbance_{stiffness}_metrics.json"
    if not path.exists():
        print(f"Missing {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def plot_ee_x_response(results: dict[str, np.lib.npyio.NpzFile]) -> Path | None:
    if not results:
        return None
    fig, axis = plt.subplots(figsize=(10, 5))
    for stiffness, data in results.items():
        axis.plot(data["time"], data["position_error"][:, 0], linewidth=1.6, label=LABELS[stiffness])
    axis.axvspan(2.0, 4.0, color="gray", alpha=0.15, label="external force")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("x_des - x_actual [m]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    output = RESULT_DIR / "ee_x_response.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")
    return output


def plot_stiffness_vs_displacement(metrics: dict[str, dict[str, object]]) -> Path | None:
    if not metrics:
        return None
    ordered = [name for name in STIFFNESSES if name in metrics]
    kx = np.array([metrics[name]["k_pos"][0] for name in ordered], dtype=float)
    measured = np.array([metrics[name]["steady_x_displacement_abs"] for name in ordered], dtype=float)
    theoretical = np.array([metrics[name]["theoretical_x_displacement"] for name in ordered], dtype=float)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(kx, measured, marker="o", linewidth=1.8, label="measured")
    axis.plot(kx, theoretical, marker="s", linewidth=1.8, label="F_ext / Kx reference")
    axis.set_xlabel("Kx [N/m]")
    axis.set_ylabel("steady |x displacement| [m]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    output = RESULT_DIR / "stiffness_vs_displacement.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")
    return output


def plot_cartesian_force(results: dict[str, np.lib.npyio.NpzFile]) -> Path | None:
    if not results:
        return None
    fig, axis = plt.subplots(figsize=(10, 5))
    for stiffness, data in results.items():
        axis.plot(data["time"], data["cartesian_force"][:, 0], linewidth=1.5, label=f"{LABELS[stiffness]} Fx")
    axis.axvspan(2.0, 4.0, color="gray", alpha=0.15)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("controller Cartesian force X [N]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    output = RESULT_DIR / "cartesian_force.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")
    return output


def plot_joint_torque(results: dict[str, np.lib.npyio.NpzFile]) -> Path | None:
    if not results:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, joint_idx in zip(axes, REPRESENTATIVE_JOINTS):
        for stiffness, data in results.items():
            axis.plot(
                data["time"],
                data["tau_command"][:, joint_idx],
                linewidth=1.3,
                label=LABELS[stiffness],
            )
        axis.axvspan(2.0, 4.0, color="gray", alpha=0.12)
        axis.set_ylabel(f"joint{joint_idx + 1} [Nm]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()
    output = RESULT_DIR / "joint_torque.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")
    return output


def plot_orientation_error(results: dict[str, np.lib.npyio.NpzFile]) -> Path | None:
    if not results:
        return None
    fig, axis = plt.subplots(figsize=(10, 5))
    for stiffness, data in results.items():
        norm_err = np.linalg.norm(data["orientation_error"], axis=1)
        axis.plot(data["time"], norm_err, linewidth=1.5, label=LABELS[stiffness])
    axis.axvspan(2.0, 4.0, color="gray", alpha=0.15)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("orientation error norm [rad]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    output = RESULT_DIR / "orientation_error.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")
    return output


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    metrics = {}
    for stiffness in STIFFNESSES:
        data = _load_npz(stiffness)
        metric = _load_metrics(stiffness)
        if data is not None:
            results[stiffness] = data
        if metric is not None:
            metrics[stiffness] = metric

    plot_ee_x_response(results)
    plot_stiffness_vs_displacement(metrics)
    plot_cartesian_force(results)
    plot_joint_torque(results)
    plot_orientation_error(results)

    for data in results.values():
        data.close()


if __name__ == "__main__":
    main()
