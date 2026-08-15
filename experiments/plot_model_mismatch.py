from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "model_mismatch"
LEVELS = ["mild", "medium", "strong"]


def _load_npz(name: str) -> np.lib.npyio.NpzFile:
    path = RESULT_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return np.load(path)


def _load_metrics(name: str) -> dict[str, object]:
    path = RESULT_DIR / f"{name}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _error_norm(data: np.lib.npyio.NpzFile) -> np.ndarray:
    return np.linalg.norm(data["position_error"], axis=1)


def _plot_family(prefix: str, output_name: str) -> Path:
    nominal = _load_npz("nominal")
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(nominal["time"], _error_norm(nominal), linewidth=1.8, label="nominal")
    handles = [nominal]
    for level in LEVELS:
        data = _load_npz(f"{prefix}_{level}")
        handles.append(data)
        axis.plot(data["time"], _error_norm(data), linewidth=1.5, label=level)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("tracking error norm [rad]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    output = RESULT_DIR / output_name
    fig.savefig(output, dpi=160)
    plt.close(fig)
    for handle in handles:
        handle.close()
    print(f"Saved {output}")
    return output


def plot_disturbance_response() -> Path:
    nominal = _load_npz("nominal")
    disturbance = _load_npz("disturbance")
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(nominal["time"], _error_norm(nominal), linewidth=1.7, label="nominal")
    axis.plot(disturbance["time"], _error_norm(disturbance), linewidth=1.7, label="disturbance")
    axis.axvspan(2.5, 3.0, color="gray", alpha=0.15, label="external force")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("tracking error norm [rad]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    output = RESULT_DIR / "disturbance_response.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    nominal.close()
    disturbance.close()
    print(f"Saved {output}")
    return output


def plot_rmse_degradation() -> Path:
    names = [
        "nominal",
        "inertial_mild",
        "inertial_medium",
        "inertial_strong",
        "damping_mild",
        "damping_medium",
        "damping_strong",
        "disturbance",
    ]
    labels = ["nom", "I mild", "I med", "I strong", "D mild", "D med", "D strong", "dist"]
    values = [float(_load_metrics(name)["overall_rmse"]) for name in names]
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(labels, values)
    axis.set_ylabel("overall RMSE [rad]")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output = RESULT_DIR / "rmse_degradation.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")
    return output


def plot_torque_rms() -> Path:
    names = [
        "nominal",
        "inertial_mild",
        "inertial_medium",
        "inertial_strong",
        "damping_mild",
        "damping_medium",
        "damping_strong",
        "disturbance",
    ]
    labels = ["nom", "I mild", "I med", "I strong", "D mild", "D med", "D strong", "dist"]
    values = [float(_load_metrics(name)["torque_rms"]) for name in names]
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(labels, values)
    axis.set_ylabel("torque RMS [Nm]")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output = RESULT_DIR / "torque_rms.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")
    return output


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_family("inertial", "inertial_tracking_error.png")
    _plot_family("damping", "damping_tracking_error.png")
    plot_disturbance_response()
    plot_rmse_degradation()
    plot_torque_rms()


if __name__ == "__main__":
    main()
