"""Persistent regression memory: must_accept, must_reject, boundary.

Both false negatives AND false positives are permanent regression cases so
they become symmetric first-class optimization signals across iterations.
Deduplication is by `(trajectory_id, failure_type)`.
"""

from __future__ import annotations

import json
from pathlib import Path

from meta_filter.core.schemas import CounterExample, Decision


class RegressionMemory:
    """Filesystem-backed store for counterexamples."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        for name in ("must_accept", "must_reject", "boundary"):
            (self.base_dir / name).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- writing

    def add(self, example: CounterExample) -> bool:
        """Insert one counterexample. Returns True if it was new."""

        bucket = self._bucket(example)
        path = self.base_dir / bucket / f"{example.trajectory_id}.json"
        if path.exists():
            return False
        path.write_text(example.model_dump_json(indent=2), encoding="utf-8")
        return True

    def extend(self, examples: list[CounterExample]) -> int:
        added = 0
        for ex in examples:
            if self.add(ex):
                added += 1
        return added

    # --------------------------------------------------------------- reading

    def must_accept(self) -> list[CounterExample]:
        return self._load("must_accept")

    def must_reject(self) -> list[CounterExample]:
        return self._load("must_reject")

    def boundary(self) -> list[CounterExample]:
        return self._load("boundary")

    def _load(self, bucket: str) -> list[CounterExample]:
        out: list[CounterExample] = []
        for path in sorted((self.base_dir / bucket).glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out.append(CounterExample.model_validate(data))
            except (json.JSONDecodeError, ValueError):
                continue
        return out

    def _bucket(self, ex: CounterExample) -> str:
        if ex.failure_type == "false_negative":
            # We kept what the critic would drop — pin it as must_reject.
            return "must_reject"
        if ex.failure_type == "false_positive":
            # We dropped what the critic would keep — pin it as must_accept.
            return "must_accept"
        if ex.failure_type == "boundary":
            return "boundary"
        return "must_reject" if ex.critic_decision == Decision.REJECT else "must_accept"
