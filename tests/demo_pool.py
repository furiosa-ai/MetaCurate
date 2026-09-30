"""The demo pool as a hidden selection set, for runs that judge by id."""

from __future__ import annotations

import json
from pathlib import Path

from meta_filter.llm.base import LLMResponse

ROOT = Path(__file__).resolve().parents[1]
DEMO_TRAIN = ROOT / "data" / "demo_train.jsonl"


def selection_rows(tmp_path: Path) -> Path:
    """The demo pool again under `sel_` ids: a selection set disjoint from
    the train rows that a judge keyed on `good` in the id can still order."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "selection.jsonl"
    rows = []
    for line in DEMO_TRAIN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["trajectory_id"] = f"sel_{row['trajectory_id']}"
            rows.append(json.dumps(row))
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out


class IdJudgeClient:
    """A pair judge that prefers whichever trajectory's id says it is good,
    in either presentation order."""

    model = "stub"

    def generate(self, prompt, **_kw):
        first = prompt.split("=== TRAJECTORY A (", 1)[1].split(")", 1)[0]
        letter = "A" if "good" in first else "B"
        return LLMResponse(
            text=f'{{"winner": "{letter}", "rationale": "it verified the fix"}}',
            prompt_tokens=0,
            completion_tokens=0,
        )


class IdJudgeBackend:
    """A pair-judging backend that rules on no individual trajectory."""

    client = IdJudgeClient()

    def critic_verdict(self, trajectory):
        return None
