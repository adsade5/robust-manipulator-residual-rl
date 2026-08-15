from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "final_multiseed"
AGG_DIR = RESULT_ROOT / "aggregate"
DEFAULT_SEEDS = [7, 17, 27]
SCALES = [1.00, 1.25, 1.50, 1.75, 2.00]
IMPROVEMENT_SCALES = [1.25, 1.50, 1.75, 2.00]
REP_JOINTS = [1, 3, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final multi-seed Residual PPO results.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def safe_scale(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def load_seed_results(seed: int) -> list[dict[str, object]]:
    path = RESULT_ROOT / f"seed_{seed}" / "evaluation" / "seed_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(seed: int, scale: float, controller: str) -> dict[str, object]:
    path = RESULT_ROOT / f"seed_{seed}" / "evaluation" / f"scale_{safe_scale(scale)}_{controller}_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(seed: int, scale: float, controller: str) -> dict[str, np.ndarray]:
    path = RESULT_ROOT / f"seed_{seed}" / "evaluation" / f"scale_{safe_scale(scale)}_{controller}.npz"
    data = np.load(path)
    try:
        return {key: data[key].copy() for key in data.files}
    finally:
        data.close()


def sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def aggregate(seeds: list[int]) -> dict[str, object]:
    per_seed: dict[str, dict[str, object]] = {str(seed): {} for seed in seeds}
    rows = []
    aggregate_rows = []
    for scale in SCALES:
        ctc_values = [float(load_metrics(seed, scale, "ctc")["motion_rmse"]) for seed in seeds]
        ctc_motion = float(ctc_values[0])
        ppo_motion = [float(load_metrics(seed, scale, "ppo_v3")["motion_rmse"]) for seed in seeds]
        ppo_residual = [float(load_metrics(seed, scale, "ppo_v3")["residual_torque_rms"]) for seed in seeds]
        ppo_torque = [float(load_metrics(seed, scale, "ppo_v3")["torque_rms"]) for seed in seeds]
        ppo_ratio = [residual / torque for residual, torque in zip(ppo_residual, ppo_torque)]
        improvements = [100.0 * (ctc_motion - value) / ctc_motion for value in ppo_motion]
        clipping = [int(load_metrics(seed, scale, "ppo_v3")["total_torque_clipping_count"]) for seed in seeds]
        saturation = [int(load_metrics(seed, scale, "ppo_v3")["residual_action_clipping_count"]) for seed in seeds]
        for seed, ppo, residual, torque, ratio, improvement in zip(seeds, ppo_motion, ppo_residual, ppo_torque, ppo_ratio, improvements):
            per_seed[str(seed)][f"{scale:.2f}"] = {
                "ctc_motion_rmse": ctc_motion,
                "ppo_motion_rmse": ppo,
                "improvement_percent": improvement,
                "residual_torque_rms": residual,
                "torque_rms": torque,
                "residual_torque_ratio": ratio,
            }
            rows.append(
                {
                    "seed": seed,
                    "scale": scale,
                    "ctc_motion_rmse": ctc_motion,
                    "ppo_motion_rmse": ppo,
                    "improvement_percent": improvement,
                    "residual_torque_rms": residual,
                    "torque_rms": torque,
                    "residual_torque_ratio": ratio,
                }
            )
        agg = {
            "scale": scale,
            "ctc_motion_rmse": ctc_motion,
            "ppo_motion_rmse_mean": float(np.mean(ppo_motion)),
            "ppo_motion_rmse_std": sample_std(ppo_motion),
            "ppo_residual_torque_rms_mean": float(np.mean(ppo_residual)),
            "ppo_residual_torque_rms_std": sample_std(ppo_residual),
            "ppo_torque_rms_mean": float(np.mean(ppo_torque)),
            "ppo_torque_rms_std": sample_std(ppo_torque),
            "ppo_residual_torque_ratio_mean": float(np.mean(ppo_ratio)),
            "ppo_residual_torque_ratio_std": sample_std(ppo_ratio),
            "total_torque_clipping_sum": int(np.sum(clipping)),
            "residual_action_saturation_sum": int(np.sum(saturation)),
        }
        if scale in IMPROVEMENT_SCALES:
            agg["improvement_percent_mean"] = float(np.mean(improvements))
            agg["improvement_percent_std"] = sample_std(improvements)
        else:
            agg["nominal_absolute_rmse_increase_mean"] = float(np.mean([value - ctc_motion for value in ppo_motion]))
            agg["nominal_absolute_rmse_increase_std"] = sample_std([value - ctc_motion for value in ppo_motion])
            agg["nominal_guard_pass_count"] = int(np.sum([value <= 0.005 for value in ppo_motion]))
        aggregate_rows.append(agg)

    return {"seeds": seeds, "per_seed": per_seed, "per_seed_rows": rows, "aggregate": aggregate_rows}


def write_outputs(summary: dict[str, object]) -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    (AGG_DIR / "final_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    aggregate_rows = summary["aggregate"]
    with (AGG_DIR / "final_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in aggregate_rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with (AGG_DIR / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = summary["per_seed_rows"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rmse(summary: dict[str, object]) -> None:
    rows = summary["aggregate"]
    scales = [float(row["scale"]) for row in rows]
    ctc = [float(row["ctc_motion_rmse"]) for row in rows]
    mean = [float(row["ppo_motion_rmse_mean"]) for row in rows]
    std = [float(row["ppo_motion_rmse_std"]) for row in rows]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(scales, ctc, marker="o", label="CTC")
    axis.errorbar(scales, mean, yerr=std, marker="s", capsize=4, label="Residual PPO v3 mean +/- std")
    axis.set_xlabel("inertial scale")
    axis.set_ylabel("Motion RMSE [rad]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(AGG_DIR / "rmse_vs_inertial_scale_multiseed.png", dpi=160)
    plt.close(fig)


def plot_improvement(summary: dict[str, object]) -> None:
    rows = [row for row in summary["aggregate"] if float(row["scale"]) in IMPROVEMENT_SCALES]
    scales = [float(row["scale"]) for row in rows]
    mean = [float(row["improvement_percent_mean"]) for row in rows]
    std = [float(row["improvement_percent_std"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar([str(scale) for scale in scales], mean, yerr=std, capsize=4)
    axis.set_xlabel("inertial scale")
    axis.set_ylabel("Motion RMSE improvement [%]")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(AGG_DIR / "multiseed_improvement.png", dpi=160)
    plt.close(fig)


def plot_tracking(seed: int, scale: float, filename: str) -> None:
    ctc = load_npz(seed, scale, "ctc")
    ppo = load_npz(seed, scale, "ppo_v3")
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(ctc["time"], np.linalg.norm(ctc["q_error"], axis=1), label="CTC")
    axis.plot(ppo["time"], np.linalg.norm(ppo["q_error"], axis=1), label=f"Residual PPO v3 (seed {seed})")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("tracking error norm [rad]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(AGG_DIR / filename, dpi=160)
    plt.close(fig)


def plot_residual_torque(seed: int, scale: float) -> None:
    ppo = load_npz(seed, scale, "ppo_v3")
    fig, axes = plt.subplots(len(REP_JOINTS), 1, figsize=(11, 8), sharex=True)
    for axis, joint_idx in zip(axes, REP_JOINTS):
        axis.plot(ppo["time"], ppo["tau_ctc"][:, joint_idx], label="tau_CTC", linewidth=1.2)
        axis.plot(ppo["time"], ppo["delta_tau_rl"][:, joint_idx], label="delta_tau_RL", linewidth=1.2)
        axis.plot(ppo["time"], ppo["tau_total"][:, joint_idx], label="tau_total", linewidth=1.2)
        axis.set_ylabel(f"joint{joint_idx + 1} [Nm]")
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(AGG_DIR / "residual_torque_scale_150.png", dpi=160)
    plt.close(fig)


def print_summary(summary: dict[str, object]) -> None:
    print("Final 3-Seed Evaluation")
    print("Scale | CTC Motion RMSE | Residual PPO mean +/- std | Mean Improvement")
    for row in summary["aggregate"]:
        scale = float(row["scale"])
        improvement = "nominal n/a"
        if scale in IMPROVEMENT_SCALES:
            improvement = f"{row['improvement_percent_mean']:.2f} +/- {row['improvement_percent_std']:.2f}%"
        print(
            f"{scale:.2f} | {row['ctc_motion_rmse']:.8f} | "
            f"{row['ppo_motion_rmse_mean']:.8f} +/- {row['ppo_motion_rmse_std']:.8f} | {improvement}"
        )
    print("\nRaw seed table")
    seeds = summary["seeds"]
    print("Scale | " + " | ".join([f"Seed {seed}" for seed in seeds]))
    per_seed = summary["per_seed"]
    for scale in SCALES:
        values = [f"{per_seed[str(seed)][f'{scale:.2f}']['ppo_motion_rmse']:.8f}" for seed in seeds]
        print(f"{scale:.2f} | " + " | ".join(values))


def main() -> None:
    args = parse_args()
    summary = aggregate(args.seeds)
    write_outputs(summary)
    plot_rmse(summary)
    plot_improvement(summary)
    plot_tracking(7, 1.50, "tracking_error_scale_150.png")
    plot_tracking(7, 2.00, "tracking_error_scale_200.png")
    plot_residual_torque(7, 1.50)
    print_summary(summary)
    print(f"saved aggregate outputs = {AGG_DIR}")


if __name__ == "__main__":
    main()
