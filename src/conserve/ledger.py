"""Append-only record of every run. Nothing counts as a result unless it is here."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path


def _git_commit(cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "uncommitted"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def append(path: str | Path, experiment: str, config: dict, metrics: dict, seed: int | None = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": experiment,
        "seed": seed,
        "commit": _git_commit(path.parent),
        "host": platform.node(),
        "config": config,
        "metrics": metrics,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry
