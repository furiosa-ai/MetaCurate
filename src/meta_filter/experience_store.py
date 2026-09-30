"""Persistent record of an optimization run.

Layout::

    runs/<run_id>/
        config.yaml
        summary.csv
        iterations/
            000/
                current_filter/
                candidates/
                    cand_00/
                        rubric.md
                        rules.py
                        policy.py
                solver_outputs/
                critic_outputs/
                boundary_flags/
                proposer/
                    prompt.txt
                    response.json
                validation.json
                selection.json
                diff.md
                report.md
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from meta_filter.filter_loader import LoadedFilter
from meta_filter.pseudonym import Pseudonymizer
from meta_filter.schemas import (
    CandidateEvaluation,
    FilterDecision,
    IterationRecord,
    RunConfig,
)


@dataclass
class ExperienceStore:
    """Filesystem-backed run record."""

    run_dir: Path
    # Set for agentic runs, where the proposer reads this directory with
    # tools. Every write below goes through it, so a trajectory id cannot
    # reach the run tree through a writer someone forgets to update.
    pseudonymizer: Pseudonymizer | None = None
    # Where held-out metrics go instead of into the run tree. Must be a path
    # outside `run_dir`, for the same reason the pseudonym map is.
    heldout_sink: Path | None = None

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "iterations").mkdir(exist_ok=True)

    def _write(self, path: Path, text: str) -> None:
        if self.pseudonymizer is not None:
            text = self.pseudonymizer.redact(text)
        path.write_text(text, encoding="utf-8")

    # --------------------------------------------------------------- config

    def save_config(self, config: RunConfig) -> None:
        self._write(
            self.run_dir / "config.yaml",
            yaml.safe_dump(config.model_dump(), sort_keys=False),
        )

    # ---------------------------------------------------------------- paths

    def iteration_dir(self, iteration: int) -> Path:
        d = self.run_dir / "iterations" / f"{iteration:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def candidate_dir(self, iteration: int, candidate_id: str) -> Path:
        d = self.iteration_dir(iteration) / "candidates" / candidate_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -------------------------------------------------------------- writers

    def save_current_filter(self, iteration: int, loaded_filter: LoadedFilter) -> None:
        target = self.iteration_dir(iteration) / "current_filter"
        src = loaded_filter.directory.resolve()
        if src == target.resolve():
            return
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)

    def save_proposer_io(
        self,
        iteration: int,
        prompt: str,
        response_text: str,
    ) -> None:
        d = self.iteration_dir(iteration) / "proposer"
        d.mkdir(parents=True, exist_ok=True)
        self._write(d / "prompt.txt", prompt)
        self._write(d / "response.json", response_text)

    def save_solver_output(
        self,
        iteration: int,
        candidate_id: str,
        decisions: list[FilterDecision],
    ) -> None:
        d = self.iteration_dir(iteration) / "solver_outputs"
        d.mkdir(parents=True, exist_ok=True)
        payload = [dec.model_dump() for dec in decisions]
        self._write(d / f"{candidate_id}.json", json.dumps(payload, indent=2, default=str))

    def save_critic_output(
        self,
        iteration: int,
        candidate_id: str,
        false_negatives: list,
        false_positives: list,
        pair_misorders: list | None = None,
    ) -> None:
        """Everything in here is a SECOND OPINION disagreeing with the filter.

        `boundary` records are not written here. BoundaryCritic is a
        self-check over the filter's own decisions, so its records carry
        `critic_decision == predicted_decision` — the filter's verdict
        relabelled as a judge's. With a judge that produces pair verdicts
        rather than per-row ones, those self-copies would be most of what
        `critic_outputs/` contained, while the proposer is told to read the
        directory for "where the judge disagreed". They live in
        `boundary_flags/`, which is what they are. See `save_boundary_flags`.
        """

        d = self.iteration_dir(iteration) / "critic_outputs"
        d.mkdir(parents=True, exist_ok=True)
        self._write(
            d / f"{candidate_id}.json",
            json.dumps(
                {
                    "false_negatives": [ex.model_dump() for ex in false_negatives],
                    "false_positives": [ex.model_dump() for ex in false_positives],
                    # The judge speaks about PAIRS, so its
                    # disagreements have this shape rather than the per-row
                    # one: each record is a row the filter ranked on the wrong
                    # side of a pair the judge ordered.
                    "pair_misorders": [ex.model_dump() for ex in (pair_misorders or [])],
                },
                indent=2,
                default=str,
            ),
        )

    def save_boundary_flags(
        self,
        iteration: int,
        candidate_id: str,
        boundary: list,
    ) -> None:
        """BoundaryCritic's fragility flags — the filter grading itself.

        Deliberately not under `critic_outputs/`: no judge saw these rows, and
        every record's `critic_decision` is a copy of the filter's own
        decision. Kept because the flags are real diagnostics (low confidence,
        competing rubric signals, rules overridden by policy) and because
        regression memory is built from them.
        """

        d = self.iteration_dir(iteration) / "boundary_flags"
        d.mkdir(parents=True, exist_ok=True)
        self._write(
            d / f"{candidate_id}.json",
            json.dumps(
                {
                    "source": "boundary_heuristic",
                    "note": (
                        "The filter's own decisions, flagged as structurally "
                        "fragile. `critic_decision` here is a copy of "
                        "`predicted_decision`; no second opinion was consulted."
                    ),
                    "boundary": [ex.model_dump() for ex in boundary],
                },
                indent=2,
                default=str,
            ),
        )

    def save_validation(
        self,
        iteration: int,
        evaluations: list[CandidateEvaluation],
    ) -> None:
        payload = [ev.model_dump() for ev in evaluations]
        if self.heldout_sink is not None:
            self._divert_heldout(iteration, payload)
        self._write(
            self.iteration_dir(iteration) / "validation.json",
            json.dumps(payload, indent=2, default=str),
        )

    def _divert_heldout(self, iteration: int, payload: list[dict]) -> None:
        """Move held-out metrics out of `payload` and into the sink.

        Held-out numbers are the memorization detector: the train↔held-out
        gap only means something if the proposer cannot see the held-out
        side and start optimizing against it. Nothing selects on them, so
        dropping them from the run tree costs the loop nothing.
        """

        rows = []
        for entry in payload:
            metrics = entry.pop("heldout_metrics", None)
            if metrics is not None:
                rows.append(
                    {
                        "iteration": iteration,
                        "candidate_id": entry.get("candidate_id"),
                        "is_baseline": entry.get("is_baseline"),
                        "heldout_metrics": metrics,
                    }
                )
        if not rows:
            return
        self.heldout_sink.parent.mkdir(parents=True, exist_ok=True)
        with self.heldout_sink.open("a", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, default=str) + "\n")

    def save_selection(
        self,
        iteration: int,
        selected_candidate: str,
        reason: str,
        *,
        pareto_ids_complexity: list[str] | None = None,
        pareto_ids_cost: list[str] | None = None,
        winner_pareto_safe: bool = False,
        gate: dict | None = None,
        per_candidate: dict | None = None,
    ) -> None:
        """Persist the iteration's selection decision and, when the paired
        gate ran, the statistics it decided on.

        `gate` is the `SelectionResult`'s paired telemetry (decision, b, c,
        z, threshold, ...) and `per_candidate` is every challenger's paired
        statistics against the baseline.
        """

        payload: dict = {
            "selected_candidate": selected_candidate,
            "reason": reason,
            "pareto_ids_complexity": pareto_ids_complexity or [],
            "pareto_ids_cost": pareto_ids_cost or [],
            "winner_pareto_safe": winner_pareto_safe,
        }
        if gate is not None:
            payload["gate"] = gate
        if per_candidate is not None:
            payload["per_candidate"] = per_candidate
        self._write(
            self.iteration_dir(iteration) / "selection.json",
            json.dumps(payload, indent=2, default=str),
        )

    def save_diff(self, iteration: int, diff: str) -> None:
        self._write(self.iteration_dir(iteration) / "diff.md", diff)

    def save_report(self, iteration: int, report_md: str) -> None:
        self._write(self.iteration_dir(iteration) / "report.md", report_md)

    def save_iteration_record(self, record: IterationRecord) -> None:
        path = self.iteration_dir(record.iteration) / "record.json"
        self._write(path, record.model_dump_json(indent=2))

    def save_v2_critic_reports(
        self,
        iteration: int,
        *,
        rubric_report,
        rule_report,
        curated_trajectory_ids: list[str],
    ) -> None:
        """Persist RubricCritic + RuleCritic outputs for one iteration.

        Files land alongside the other iteration artifacts. Both reports must
        expose `.to_dict()` returning a JSON-serialisable dict — that's the
        contract every design-v2 critic follows.
        """

        d = self.iteration_dir(iteration) / "v2_critics"
        d.mkdir(parents=True, exist_ok=True)
        self._write(
            d / "rubric_critic.json",
            json.dumps(rubric_report.to_dict(), indent=2, ensure_ascii=False),
        )
        self._write(
            d / "rule_critic.json",
            json.dumps(rule_report.to_dict(), indent=2, ensure_ascii=False),
        )
        self._write(
            d / "curated_trajectory_ids.json",
            json.dumps(curated_trajectory_ids, indent=2),
        )

    # ----------------------------------------------------------- summary csv

    def append_summary(self, row: dict) -> None:
        path = self.run_dir / "summary.csv"
        keys = list(row.keys())
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=keys)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
