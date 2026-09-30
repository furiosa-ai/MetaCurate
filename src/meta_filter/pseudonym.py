"""Per-run pseudonyms for trajectory ids on proposer-visible surfaces.

The agentic proposer works inside its run directory and is told to read the
evidence there. The loop is scored against the pair judge's verdicts, and
those, like the (diagnostic) critic outputs, name trajectories by id. An id
that is stable and meaningful is enough for a candidate to stop generalizing
and start enumerating answers: ``if traj.trajectory_id == "...": score it
high``. Nothing in the training-side numbers would show it.

Replacing the id with a pseudonym keeps what the proposer legitimately uses —
that the *same* trajectory keeps failing across iterations — and removes the
key it would need to hardcode a verdict.

Two design points that are easy to get wrong:

* **The salt must be secret.** A plain digest of the id does not help:
  `rules.py` executes with the real trajectory in hand and could hash it at
  runtime to rebuild the mapping. The salt is random per run and written
  outside the run directory, where the tool guard cannot reach it.

* **Redact the serialized text, not the fields.** Ids also appear inside
  prose the models write — judge rationales, proposer diagnoses, rationale
  docstrings — where no field-level rewrite would find them. Substituting
  over the rendered JSON/Markdown catches those too.

This narrows the cheat to subtler keys that cannot be removed without gutting
the evidence (the task text still describes the issue). It is a deterrent,
not a proof. The generalization gap against a real held-out set is what would
actually detect memorization.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# `pandas-dev__pandas-48540::att0` is the trajectory; the part before `::` is
# the SWE-bench instance, which shows up on its own in `metadata.instance_id`.
_ATTEMPT_SEP = "::"


@dataclass
class Pseudonymizer:
    """Maps a fixed set of trajectory ids to stable per-run pseudonyms."""

    salt: str
    mapping: dict[str, str] = field(default_factory=dict)

    @classmethod
    def for_pool(cls, trajectory_ids: Iterable[str], salt: str | None = None) -> Pseudonymizer:
        pseudo = cls(salt=salt or secrets.token_hex(16))
        pseudo.add(trajectory_ids)
        return pseudo

    def add(self, trajectory_ids: Iterable[str]) -> None:
        for raw in trajectory_ids:
            if not raw:
                continue
            self.mapping.setdefault(raw, self._digest(raw))
            instance, sep, _ = raw.partition(_ATTEMPT_SEP)
            if sep and instance:
                # Same pseudonym, so the two spellings stay joinable.
                self.mapping.setdefault(instance, self.mapping[raw])

    def _digest(self, raw: str) -> str:
        return "traj_" + hashlib.sha256(f"{self.salt}:{raw}".encode()).hexdigest()[:8]

    def redact(self, text: str) -> str:
        """Replace every known id in `text`, longest first.

        Longest-first matters: rewriting the bare instance id before the full
        trajectory id would leave a dangling `::att0` behind.
        """

        if not text or not self.mapping:
            return text
        for raw in sorted(self.mapping, key=len, reverse=True):
            if raw in text:
                text = text.replace(raw, self.mapping[raw])
        return text

    def save(self, path: Path) -> None:
        """Persist the map so the viewer and analysis can resolve pseudonyms.

        Must be written outside the run directory: it is the key the proposer
        is not allowed to have.

        The reverse direction picks the longest id behind each pseudonym. Two
        spellings share one pseudonym, and the full `…::att0` form is the one
        that identifies a trajectory; naively inverting the dict would drop it
        for whichever spelling happened to come last.
        """

        reverse: dict[str, str] = {}
        for raw, pseudo in self.mapping.items():
            if len(raw) > len(reverse.get(pseudo, "")):
                reverse[pseudo] = raw

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "salt": self.salt,
                    "pseudonym_to_id": reverse,
                    "id_to_pseudonym": dict(self.mapping),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
