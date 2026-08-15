from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "minimal_torque"
MODES = ["none", "gravity", "pd", "pd_gc"]
MODE_LABELS = {
    "none": "none",
    "gravity": "gravity",
    "pd": "PD",
    "pd_gc": "PD + GC",
}
REPRESENTATIVE_JOINTS = [0, 1, 3, 5]


def _load_results() -> dict[str, np.lib.npyio.NpzFile]:
    loaded = {}
    missing = []
    for mode in MODES:
        path = RESULT_DIR / f"{mode}.npz"
        if path.exists():
            loaded[mode] = np.load(path)
        else:
            missing.append(path)
    if missing:
        print("Missing result files:")
        for path in missing:
            print(f"  {path}")
    return loaded


def plot_joint_positions(results: dict[str, np.lib.npyio.NpzFile]) -> Path | None:
    if not results:
        print("No result files found; skipping joint position plot.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for axis, joint_idx in zip(axes, REPRESENTATIVE_JOINTS):
        for mode, data in results.items():
            axis.plot(data["time"], data["q"][:, joint_idx], label=MODE_LABELS[mode], linewidth=1.6)
        axis.set_title(f"joint{joint_idx + 1}")
        axis.set_ylabel("q [rad]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[-2].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()

    output_path = RESULT_DIR / "joint_positions.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def plot_pd_vs_pd_gc_error(results: dict[str, np.lib.npyio.NpzFile]) -> Path | None:
    required = ["pd", "pd_gc"]
    missing = [mode for mode in required if mode not in results or "q_error" not in results[mode]]
    if missing:
        print(f"Missing PD error data for {missing}; skipping PD vs PD+GC error plot.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for axis, joint_idx in zip(axes, REPRESENTATIVE_JOINTS):
        for mode in required:
            data = results[mode]
            axis.plot(
                data["time"],
                data["q_error"][:, joint_idx],
                label=MODE_LABELS[mode],
                linewidth=1.6,
            )
        axis.set_title(f"joint{joint_idx + 1} error")
        axis.set_ylabel("q_des - q [rad]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[-2].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()

    output_path = RESULT_DIR / "pd_vs_pd_gc_error.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_results()
    plot_joint_positions(results)
    plot_pd_vs_pd_gc_error(results)
    for data in results.values():
        data.close()


if __name__ == "__main__":
    main()
