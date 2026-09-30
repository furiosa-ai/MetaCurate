# meta-filter

An LLM-driven optimizer for **hybrid trajectory filters** that rank
software-engineering agent trajectories as SFT training data. A filter is
three files:

* `rubric.md` — natural-language criteria, each scored by an LLM on a
  +2 … −2 scale;
* `rules.py` — deterministic `rule_*` functions over the transcript and the
  submitted diff;
* `policy.py` — `decide()`, which composes rubric scores and rule verdicts into
  one continuous score per trajectory.

A tool-using proposer edits all three files; candidates are measured against
a pairwise LLM judge; and a candidate replaces the current filter only when a
paired test on a **hidden** set of judged pairs says it orders them better.
No component of the loop ever sees a harness outcome (resolved, test results,
reference patches): the filter is learned from what the agent did and what it
shipped.

## The loop

One iteration:

1. **Score the current filter** on the train pool (LLM scorer, one cached
   call per trajectory × criterion; rules and policy are free).
2. **Propose.** The proposer (Codex CLI in a sandboxed copy of the run
   directory) writes `candidates_per_iteration` candidate filters, one
   hypothesis each. It is shown the current filter, the judged pairs the
   filter orders wrongly, per-criterion rank lift, criterion health (dead,
   redundant, unread or zero-weight axes) and loop telemetry, and it can score
   a draft criterion or check a draft rule on a few train rows
   (`meta_filter.score_tool`, `meta_filter.rule_tool`).
3. **Precheck and one revision.** Every candidate is scored on the pairs
   already judged and checked against the growth caps, the deletion floor and
   the outcome-field ban (`constraints.py`); the proposer sees the readout and
   gets exactly one revision leg. Invalid, duplicate and no-op candidates are
   held back before any new scoring is paid for.
4. **Judge.** Pairs every candidate orders the same way are free
   pseudo-truth; a stratified sample of the disputed pairs (plus a consensus
   audit share) goes to the pair judge, which sees both trajectories in both
   presentation orders and resolves a pair only when the two orders agree.
   Verdicts accumulate across iterations into the train-side exam.
5. **Select.** Each candidate is scored on the hidden selection rows and
   compared with the current filter pair-for-pair (McNemar: `b` pairs it
   fixes, `c` pairs it breaks). It is accepted only if its one-sided `p`
   clears a Šidák-corrected threshold — one z-unit stricter for a candidate
   over a soft growth cap — and its lead over any other eligible candidate is
   itself tested. A candidate smaller than the current filter in every
   complexity component is accepted on `b ≥ c`.
6. **Confirm.** The winner is re-scored on a fresh draw, which must still
   favour it; the fresh draw becomes the baseline's number. Patience stops
   the run after `--patience` iterations without an acceptance (never before
   `--min-iterations`).

The train-side score (`pair_concordance − 0.15 · complexity`, with a
redundancy penalty reported beside it) is evidence shown to the proposer; it
never picks the winner.

## Install

```bash
uv sync
```

Python 3.12. The four run cells call models through OpenRouter and need the
`codex` CLI on `PATH`; the offline demo and the test suite need neither.

## Offline demo

The mock provider stands in for every LLM, so this runs in seconds, offline,
for free:

```bash
uv run python -m meta_filter.cli run \
    --llm-provider mock --solver-backend llm \
    --selection-data data/demo_selection.jsonl \
    --complexity-ceiling rubric_criteria=9,rule_count=13,rules_loc=522,rubric_chars=8785 \
    --iterations 2 --run-id demo \
    --solver-cache-dir /tmp/demo/scorer_cache \
    --judge-cache-dir /tmp/demo/judge_cache \
    --pair-judge-cache-dir /tmp/demo/pair_judge_cache \
    --runs-dir /tmp/demo/runs --memory-dir /tmp/demo/memory
```

It starts from `filters/initial_blank_wired_v2` (the `configs/default.yaml`
seed) on `data/demo_train.jsonl`. The rubric and rule critics keep their
caches under `data/` in the working directory. The mock proposer's candidates
rarely disagree on a pair, so the demo usually buys no judged pair and logs a
warning that it is scoring on the sampled pairs; it exercises the plumbing,
not the search.

Other commands:

```bash
uv run python -m meta_filter.cli evaluate --filter filters/initial_blank_wired_v2 \
    --data data/demo_validation.jsonl          # one filter's decision counts
uv run python -m meta_filter.cli analyze-run /tmp/demo/runs/demo
```

## The four run cells

`configs/metadata_free/` holds the four configurations the method is
evaluated with — two trajectory pools (DSv4Pro, Nemotron) × two initial
filters (`filters/initial_blank_wired_v2`, `filters/initial_ranking_nometa`),
everything else shared. `configs/metadata_free/README.md` lists every data
file and cache directory they expect and how the loop keeps outcome fields
out.

```bash
uv run python -m scripts.run_metadata_free --all --dry-run       # print the commands
uv run python -m scripts.run_metadata_free --cell nemotron_blank # launch one, detached
```

The launcher reads `OPENROUTER_API_KEY` from `~/.env`, refuses to start until
every input exists, and logs to `runs/_logs/<run_id>.log`. The cells share
three LLM caches, `data/scorer_cache/`, `data/judge_cache/` and
`data/pair_judge_cache/`; existing caches built with the same models can be
placed there and are reused, so only missing keys cost a call. `--seed N` moves
the DSv4Pro cells to their per-seed train and probe pools.

### One-time: the selection set

Every run of a pool is graded on one judged pair list over rows disjoint
from train and probe, built once before any run:

```bash
uv run python -m scripts.build_selection_set \
    --rows data/nemotron_selection220.jsonl \
    --judge-model openai/gpt-5.6-luna \
    --judge-cache-dir data/pair_judge_cache \
    --exclude data/nemotron_train50.jsonl data/nemotron_probe25.jsonl \
    --out data/nemotron_selection220_judged.json
```

The builder strips the rows exactly as a run does and judges every pair in
both orders (`--pairs all`, the default); the pair-judge cache is the one the
runs use, so a re-run asks the model nothing.

## Building a training subset

After a run, the filter in `runs/<run_id>/iterations/<k>/current_filter` is
applied to a whole pool and cut into a subset:

```bash
OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
uv run python -m scripts.build_ranking_subset \
    --filter-dir runs/<run_id>/iterations/<k>/current_filter \
    --pool data/nemotron_superset.jsonl --superset-dir data/nemotron_superset \
    --scorer-engine openai_compatible --scorer-model openai/gpt-5.6-luna \
    --scorer-cache-dir data/scorer_cache \
    --top 1000 --name nemotron_blank_1k \
    --readme README_subset.md --note "one line for manifest/recipe" \
    --out runs/_subsets
```

`--top N` takes the N highest-scoring rows; `--accept-all`, `--accepts-only`
and `--accept-random N` are the other cuts. Rows are copied byte for byte
from a rendered superset directory (`<superset>/train.jsonl` plus
`manifest.json`; see `scripts/package_subset.py`), and each subset ships an
SFT recipe renamed after it (`scripts/sft_config.py`).
`scripts/score_pool_baseline.py` scores a pool with the two comparison
baselines — a generated rubric (`grm`) and a no-rubric keep/drop judge
(`llmfilter`) — into a ledger a subset can be cut from.

## Repository layout

```
src/meta_filter/     the loop: optimizer, proposer, scorer, pair judge,
                     selection set and gate, precheck, constraints, critics
filters/             the two initial filters the cells start from
configs/             default.yaml (the demo) and the four cells
scripts/             launcher, selection-set builder, subset tooling
data/                demo pools (in git); real pools and caches (not in git)
tests/               unit and end-to-end tests, synthetic fixture pools
```

A run writes `runs/<run_id>/`: `config.yaml`, `summary.csv` (one row per
iteration: train score, selection statistics, gate decision, per-role cost),
and `iterations/<k>/` with every candidate, its `precheck.json`,
`validation.json`, `selection.json`, `ranking.json`, `ranking_pairs.jsonl`
and the proposer transcript. The hidden selection set, the pseudonym map and
the source-pool paths live beside the run (`runs/_selection/`,
`runs/_idmap/`, `runs/_metadata_free/`), outside the proposer's workspace.

## Tests

```bash
uv run pytest tests/ -q
```

Everything runs offline.
