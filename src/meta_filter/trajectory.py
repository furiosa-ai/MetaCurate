"""Trajectory loading and helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from meta_filter.metadata_free import strip_row
from meta_filter.schemas import Trajectory


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """Load trajectories from a JSONL file, stripped. Empty lines are ignored.

    Every row goes through `meta_filter.metadata_free.strip_row` before it
    becomes a `Trajectory`: only the allow-listed metadata keys survive and
    instance-bearing ids are pseudonymized. This is the one place that
    happens, so nothing built from the returned list can see what was
    dropped.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"trajectory file not found: {path}")

    trajectories: list[Trajectory] = []
    with path.open("r", encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} — invalid JSON: {exc.msg}") from exc
            trajectories.append(Trajectory.model_validate(strip_row(obj)))
    return trajectories


def dump_trajectories(trajectories: Iterable[Trajectory], path: str | Path) -> None:
    """Write trajectories to JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for traj in trajectories:
            fp.write(traj.model_dump_json())
            fp.write("\n")
