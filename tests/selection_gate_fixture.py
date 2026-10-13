"""A mock run whose selection-row behaviour is fully controlled.

Shared by the gate and wiring tests. Every row states two numbers in its task
text: `truth`, the order the judge enforces, and `noisy`, `truth` plus a fixed
perturbation that inverts every other adjacent pair. Train rows additionally
state `boost = truth - noisy`, which the selection rows do not have. (The
numbers live in the task because the loop strips every metadata key outside
its allow-list at load.) From those three numbers three filters follow:

* the BASELINE scores `noisy` — right on most pairs, wrong on the inverted ones;
* the TRUTH candidate scores `truth` — better everywhere, including on the
  selection rows, so the paired gate has `b > 0, c = 0` to accept on;
* the TRAIN_ONLY candidate scores `noisy + boost` — perfect on train,
  identical to the baseline on the selection rows (`b = c = 0`), so the gate
  keeps it out.

The judge reads the two ids off the pair prompt and prefers the higher
`truth`, in whichever order it is asked, so every planned pair resolves.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from meta_filter.core.filter_loader import materialize_candidate
from meta_filter.core.solver import ScorerCost, Solver
from meta_filter.gate.selector import Selector
from meta_filter.gate.validator import Validator
from meta_filter.llm.base import LLMResponse
from meta_filter.loop.optimizer import Optimizer, OptimizerConfig
from meta_filter.propose.proposer import Proposer, ProposerResult

RUBRIC = """# Rubric

## Criterion: Truth
+2 when the row is what the judge would rank first.

## Criterion: Noisy
+2 when the row looks good at a glance.

## Criterion: TrainBoost
Correction available only where it was measured.
"""

RULES = "def rule_noop(trajectory):\n    return None\n"

BASELINE_POLICY = """RUBRIC_CRITERIA = ['Noisy']


def decide(trajectory, scores, triggered):
    return {'decision': 'REVIEW', 'score': scores['Noisy'], 'confidence': 0.5}
"""

TRUTH_POLICY = """RUBRIC_CRITERIA = ['Truth']


def decide(trajectory, scores, triggered):
    return {'decision': 'REVIEW', 'score': scores['Truth'], 'confidence': 0.5}
"""

TRAIN_ONLY_POLICY = """RUBRIC_CRITERIA = ['Noisy', 'TrainBoost']


def decide(trajectory, scores, triggered):
    return {
        'decision': 'REVIEW',
        'score': scores['Noisy'] + scores['TrainBoost'],
        'confidence': 0.5,
    }
"""

# Adjacent rows are 10 apart in `truth`; a +14 / -14 alternation inverts
# every even-indexed adjacent pair and nothing else.
_NOISE = (14.0, -14.0)


def write_pool(path: Path, prefix: str, n: int, *, train: bool) -> list[str]:
    """Write `n` synthetic rows and return their ids, in truth order."""

    ids: list[str] = []
    with path.open("w", encoding="utf-8") as fp:
        for i in range(n):
            tid = f"{prefix}-{i:03d}"
            truth = 10.0 * i
            noisy = truth + _NOISE[i % 2]
            task = f"task {prefix} {i} truth={truth} noisy={noisy}"
            if train:
                task += f" boost={truth - noisy}"
            fp.write(
                json.dumps(
                    {
                        "trajectory_id": tid,
                        "task": task,
                        "turns": [
                            {"role": "assistant", "action": "edit", "content": f"turn {j}"}
                            for j in range(2 + i % 3)
                        ],
                        "metadata": {"source": "fixture"},
                    }
                )
                + "\n"
            )
            ids.append(tid)
    return ids


def task_number(task: str, name: str) -> float:
    """The `name=<float>` a fixture row states in its task text, else 0.0."""

    match = re.search(rf"\b{name}=(-?[0-9.]+)", task or "")
    return float(match.group(1)) if match else 0.0


def truth_by_id(*paths: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["trajectory_id"]] = task_number(row["task"], "truth")
    return out


class MetaScorer:
    """Scores every criterion straight off the row's task text; deterministic."""

    FIELDS = {"Truth": "truth", "Noisy": "noisy", "TrainBoost": "boost"}

    def score(self, trajectory, criteria, rubric_md=""):
        scores = {c: task_number(trajectory.task, self.FIELDS[c]) for c in criteria}
        return scores, [], ScorerCost()


@dataclass
class TruthJudgeClient:
    """Prefers the higher `truth`, whichever side it is shown as A."""

    truths: dict[str, float]
    model: str = "truth-judge"
    pair_calls: int = 0
    pair_ids_seen: set[str] = field(default_factory=set)

    def generate(self, prompt, **_kw):
        if "=== TRAJECTORY A (" not in prompt:
            return LLMResponse(text="{}", prompt_tokens=0, completion_tokens=0)
        id_a = prompt.split("=== TRAJECTORY A (", 1)[1].split(")", 1)[0]
        id_b = prompt.split("=== TRAJECTORY B (", 1)[1].split(")", 1)[0]
        self.pair_calls += 1
        self.pair_ids_seen.update((id_a, id_b))
        a, b = self.truths[id_a], self.truths[id_b]
        letter = "TIE" if a == b else ("A" if a > b else "B")
        return LLMResponse(
            text=json.dumps({"winner": letter, "rationale": "higher truth"}),
            prompt_tokens=0,
            completion_tokens=0,
        )


class Backend:
    """The one attribute the optimizer's pair judge needs, plus a silent critic."""

    def __init__(self, client: TruthJudgeClient) -> None:
        self.client = client

    def critic_verdict(self, trajectory):
        return None


class ScriptedProposer(Proposer):
    """Writes the candidates a script names for each iteration.

    `script[k]` is a list of ``(name, policy_py)`` or an exception instance
    to raise (a proposer failure). Iterations not in the script propose
    nothing. The prompt is a dump of everything the optimizer handed over,
    which is what the exposure guard greps.
    """

    def __init__(self, script: dict[int, list[tuple[str, str]] | Exception], client) -> None:
        self.script = script
        self.client = client
        self.prompts: list[str] = []
        self.revise_calls = 0

    def propose(self, *, candidates_root, iteration, **kwargs):
        step = self.script.get(iteration, [])
        if isinstance(step, Exception):
            raise step
        prompt = json.dumps({"iteration": iteration, **kwargs}, default=str, sort_keys=True)
        self.prompts.append(prompt)
        dirs = []
        for index, (name, policy) in enumerate(step):
            cand = Path(candidates_root) / f"iter{iteration:03d}_{index:02d}_{name}"
            materialize_candidate(cand, rubric_md=RUBRIC, rules_py=RULES, policy_py=policy)
            dirs.append(cand)
        return ProposerResult(
            prompt=prompt,
            response=LLMResponse(text="{}", prompt_tokens=0, completion_tokens=0),
            candidate_dirs=dirs,
        )

    def revise(self, *, result, precheck_block, candidates_root, iteration):
        self.revise_calls += 1
        prompt = json.dumps(
            {"iteration": iteration, "precheck_block": precheck_block, "leg": "revise"},
            default=str,
        )
        self.prompts.append(prompt)
        return ProposerResult(
            prompt=prompt,
            response=result.response,
            candidate_dirs=result.candidate_dirs,
            legs=result.legs + 1,
        )


@dataclass
class MockRun:
    root: Path
    train: Path
    selection: Path
    initial: Path
    selection_ids: list[str]
    train_ids: list[str]
    judge: TruthJudgeClient

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"


def prepare(tmp_path: Path, *, n_train: int = 12, n_selection: int = 16) -> MockRun:
    """Write the pools and the initial filter.

    Sixteen selection rows and (by default) 60 planned pairs: the baseline
    inverts exactly one pair per even row, so `TRUTH` fixes 8 pairs and
    breaks none, and 60 pairs is enough for the hard half of the plan to be
    the whole small-gap quartile, which is where all 8 inversions sit. The
    gate is on the exact one-sided binomial p below 25 discordant pairs, and
    8-0 (p = 0.0039) clears the three-candidate threshold (p = 0.0076); with
    twelve rows the most `TRUTH` could fix was 6-0 (p = 0.0156), which the
    normal `z` let through and the exact test — correctly — does not.
    """

    train = tmp_path / "train.jsonl"
    selection = tmp_path / "selection.jsonl"
    train_ids = write_pool(train, "train", n_train, train=True)
    selection_ids = write_pool(selection, "sel", n_selection, train=False)
    initial = tmp_path / "initial"
    materialize_candidate(initial, rubric_md=RUBRIC, rules_py=RULES, policy_py=BASELINE_POLICY)
    judge = TruthJudgeClient(truth_by_id(train, selection))
    return MockRun(
        root=tmp_path,
        train=train,
        selection=selection,
        initial=initial,
        selection_ids=selection_ids,
        train_ids=train_ids,
        judge=judge,
    )


def build_optimizer(
    run: MockRun,
    *,
    script: dict[int, list[tuple[str, str]] | Exception],
    iterations: int,
    run_id: str,
    selection_pairs: int | str = 60,
    selection_set: Path | None = None,
    patience: int | None = None,
    min_iterations: int = 4,
    resume: bool = False,
    gate_z: float | None = None,
) -> Optimizer:
    proposer = ScriptedProposer(script, run.judge)
    return Optimizer(
        initial_filter_dir=run.initial,
        train_data=run.train,
        validation_data=None,
        runs_dir=run.runs_dir,
        memory_dir=run.root / "memory",
        proposer=proposer,
        selector=Selector(),
        solver=Solver(rubric_scorer=MetaScorer()),
        validator=Validator(loss_weights={}, critic_backend=Backend(run.judge)),
        config=OptimizerConfig(
            iterations=iterations,
            candidates_per_iteration=2,
            pair_budget=8,
            max_pairs=200,
            # Scores are 12 distinct floats; the pool-derived floor would be
            # 10 and the guard is not what these tests are about.
            min_distinct_levels=3,
            pair_judge_cache_dir=run.root / "pair_cache",
            resume=resume,
            selection_data=run.selection,
            selection_pairs=selection_pairs,
            selection_seed=4242,
            selection_set=selection_set,
            gate_z=gate_z,
            gate_min_discordant=5,
            patience=patience,
            min_iterations=min_iterations,
        ),
        run_id=run_id,
    )
