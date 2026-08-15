from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".toml", ".yaml", ".yml", ".xml", ".gitignore"}
LOCAL_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:" + re.escape("\\")),
    re.compile(r"/home/[^/\s]+"),
    re.compile(r"/Users/[^/\s]+"),
]
SECRET_PATTERNS = [
    re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"secret\s*[:=]", re.IGNORECASE),
    re.compile(r"password\s*[:=]", re.IGNORECASE),
    re.compile(r"token\s*[:=]", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
]


def git_candidates() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return [PROJECT_ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return [path for path in PROJECT_ROOT.rglob("*") if path.is_file()]


def is_text_file(path: Path) -> bool:
    return path.name == ".gitignore" or path.suffix.lower() in TEXT_SUFFIXES


def safe_read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError:
        return None


def main() -> None:
    candidates = [path for path in git_candidates() if path.exists()]
    hardcoded_paths: list[str] = []
    secret_hits: list[str] = []
    oversized_50: list[tuple[str, int]] = []
    oversized_100: list[tuple[str, int]] = []
    blocked_dirs: list[str] = []

    for path in candidates:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        size = path.stat().st_size
        if size > 50 * 1024 * 1024:
            oversized_50.append((rel, size))
        if size > 100 * 1024 * 1024:
            oversized_100.append((rel, size))
        lowered = rel.lower()
        if "__pycache__" in lowered or ".idea" in lowered or ".vscode" in lowered:
            blocked_dirs.append(rel)
        if "tensorboard" in lowered or "checkpoints" in lowered:
            blocked_dirs.append(rel)
        if rel == "scripts/github_preflight.py":
            continue
        if not is_text_file(path):
            continue
        text = safe_read(path)
        if text is None:
            continue
        if any(pattern.search(text) for pattern in LOCAL_PATH_PATTERNS):
            hardcoded_paths.append(rel)
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            secret_hits.append(rel)

    required = {
        "README.md": (PROJECT_ROOT / "README.md").exists(),
        "requirements.txt": (PROJECT_ROOT / "requirements.txt").exists(),
        "final_results.json": (PROJECT_ROOT / "results" / "final_multiseed" / "aggregate" / "final_results.json").exists(),
        "final_results.csv": (PROJECT_ROOT / "results" / "final_multiseed" / "aggregate" / "final_results.csv").exists(),
        "main_final_figure": (PROJECT_ROOT / "results" / "final_multiseed" / "aggregate" / "rmse_vs_inertial_scale_multiseed.png").exists(),
    }

    report = {
        "candidate_file_count": len(candidates),
        "hardcoded_path_files": hardcoded_paths,
        "oversized_files_over_50mb": oversized_50,
        "oversized_files_over_100mb": oversized_100,
        "blocked_artifact_candidates": blocked_dirs,
        "secret_pattern_files": secret_hits,
        "required_files": required,
        "pass": not hardcoded_paths
        and not oversized_100
        and not blocked_dirs
        and not secret_hits
        and all(required.values()),
    }

    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
