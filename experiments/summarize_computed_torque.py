from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDGC_DIR = PROJECT_ROOT / "results" / "joint_tracking"
CTC_DIR = PROJECT_ROOT / "results" / "computed_torque"


def _load(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    pdgc = _load(PDGC_DIR / "pd_gc_metrics.json")
    ctc = _load(CTC_DIR / "ctc_metrics.json")

    pdgc_rmse = float(pdgc["overall_rmse"])
    ctc_rmse = float(ctc["overall_rmse"])
    relative_change = 100.0 * (ctc_rmse - pdgc_rmse) / pdgc_rmse if pdgc_rmse > 0.0 else 0.0
    clipping_pdgc = sum(int(x) for x in pdgc["clip_counts"])
    clipping_ctc = sum(int(x) for x in ctc["clip_counts"])

    print("PD+GC vs Computed Torque")
    print()
    print(f"{'':22s}{'PD+GC':>14s}{'CTC':>14s}")
    print(f"{'overall RMSE':22s}{pdgc_rmse:14.8f}{ctc_rmse:14.8f}")
    print(
        f"{'max error':22s}"
        f"{float(pdgc['overall_max_abs_error']):14.8f}"
        f"{float(ctc['overall_max_abs_error']):14.8f}"
    )
    print(
        f"{'max qvel':22s}"
        f"{float(pdgc['max_abs_qvel']):14.8f}"
        f"{float(ctc['max_abs_qvel']):14.8f}"
    )
    print(
        f"{'max torque':22s}"
        f"{float(pdgc['max_abs_tau']):14.8f}"
        f"{float(ctc['max_abs_tau']):14.8f}"
    )
    print(f"{'clipping count':22s}{clipping_pdgc:14d}{clipping_ctc:14d}")
    print()
    print(f"CTC RMSE relative change: {relative_change:.2f}%")
    if relative_change >= 0.0:
        print("CTC did not outperform PD+GC in this experiment.")


if __name__ == "__main__":
    main()
