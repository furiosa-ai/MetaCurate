# The four run cells

Four runs of the optimizer loop in which no component — proposer, scorer,
pair judge, trajectory critic, filter critics, the proposer's scoring and
rule tools — ever sees a harness outcome. The grid is pool x initial filter
(blank / seeded); every cell uses RNG seed 29:

| cell | train pool | probe | selection rows | selection verdicts | initial filter |
| --- | --- | --- | --- | --- | --- |
| `dsv4pro_blank` | `data/dsv4pro_train50_s29.jsonl` | `data/dsv4pro_probe25_s29.jsonl` | `data/dsv4pro_selection220.jsonl` | `data/dsv4pro_selection220_judged.json` | `filters/initial_blank_wired_v2` |
| `dsv4pro_seeded` | `data/dsv4pro_train50_s29.jsonl` | `data/dsv4pro_probe25_s29.jsonl` | `data/dsv4pro_selection220.jsonl` | `data/dsv4pro_selection220_judged.json` | `filters/initial_ranking_nometa` |
| `nemotron_blank` | `data/nemotron_train50.jsonl` | `data/nemotron_probe25.jsonl` | `data/nemotron_selection220.jsonl` | `data/nemotron_selection220_judged.json` | `filters/initial_blank_wired_v2` |
| `nemotron_seeded` | `data/nemotron_train50.jsonl` | `data/nemotron_probe25.jsonl` | `data/nemotron_selection220.jsonl` | `data/nemotron_selection220_judged.json` | `filters/initial_ranking_nometa` |

The DSv4Pro train and probe pools are per-seed samples: `--seed N` switches
them to `dsv4pro_train50_sN.jsonl` / `dsv4pro_probe25_sN.jsonl` and the
launcher refuses a seed whose files are absent. The Nemotron cells keep one
pool at every seed.

Everything else is shared by all four and lives in each config: 20
iterations, 3 candidates, patience 5 after at least 4 iterations; proposer
`openai/gpt-5.6-sol-pro` through the Codex harness; solver and pair judge
`openai/gpt-5.6-luna` over OpenRouter; pair budget 50, pair universe 1225,
consensus excluded with a 0.25 audit share; the default score weights
(`lambda_complexity` 0.15); $4.00 of scoring-tool budget per proposer leg;
a run-wide hard cap of 9 rubric criteria, 13 rules, 522 rules.py lines and
8785 rubric characters (`--complexity-ceiling`).
`tests/test_metadata_free.py` fails if any of these drift apart.

`DSv4Pro` is a mini-SWE-agent pool whose rows carry the harness grading
fields on disk; `Nemotron` is the Nemotron-SFT-SWE mini-SWE-agent subset,
whose rows never carried them.

## Expected files

The pools are trajectory data and are not in git. A checkout that runs the
cells needs, under `data/`:

| file | what it is |
| --- | --- |
| `dsv4pro_train50_s29.jsonl` | DSv4Pro train pool, 50 rows (one file per seed) |
| `dsv4pro_probe25_s29.jsonl` | DSv4Pro probe pool, 25 rows (one file per seed) |
| `dsv4pro_selection220.jsonl` | DSv4Pro selection rows, 220, disjoint from train and probe |
| `dsv4pro_selection220_judged.json` | the judged selection pairs over those rows |
| `nemotron_train50.jsonl` | Nemotron train pool, 50 rows |
| `nemotron_probe25.jsonl` | Nemotron probe pool, 25 rows |
| `nemotron_selection220.jsonl` | Nemotron selection rows, 220, disjoint from train and probe |
| `nemotron_selection220_judged.json` | the judged selection pairs over those rows |
| `demo_validation.jsonl` | the held-out file every cell names (in git) |

and three cache directories the cells share, created on first use:

| directory | keyed by |
| --- | --- |
| `data/scorer_cache/` | trajectory id, criterion section text, scorer model, and the fields the row lacks |
| `data/judge_cache/` | trajectory id, critic model, and the label view of the row |
| `data/pair_judge_cache/` | the two trajectory ids, judge model, view signature, label view and presentation order |

The cells name these three directories on their command line
(`--solver-cache-dir`, `--judge-cache-dir`, `--pair-judge-cache-dir`).
Existing caches built under the same scorer and judge models can be placed
there as they are and are reused: a cell asks the model only for keys it
does not find. No cache key contains a file name, so renaming a pool file
re-keys no cache.
The one place a file name matters is a selection-verdict file's
`provenance.rows_file` (below).

## Running a cell

```bash
uv run python -m scripts.run_metadata_free --cell nemotron_blank --dry-run   # print, do nothing
uv run python -m scripts.run_metadata_free --cell nemotron_blank             # launch detached
uv run python -m scripts.run_metadata_free --all --dry-run
```

The launcher runs `meta_filter.cli run --config configs/metadata_free/<cell>.yaml`
plus the config's `launch.args` and `launch.pool`. A dry run reads no key,
writes nothing and lists any input missing on this machine. A real launch
refuses to start until every input exists, reads `OPENROUTER_API_KEY` from
`~/.env`, and logs to `runs/_logs/<run_id>.log` with run id
`run_<YYYYMMDD>_metafree_<cell>_s<seed>`.

## The selection verdicts, built once

Every run of a pool is graded on the same judged pair list, built once
outside any run and read with `--selection-set`. The builder strips the rows
exactly as a run does, judges every pair in both presentation orders and
stamps the output `metadata_free: true`:

```bash
uv run python -m scripts.build_selection_set \
    --rows data/dsv4pro_selection220.jsonl \
    --judge-model openai/gpt-5.6-luna \
    --judge-cache-dir data/pair_judge_cache \
    --exclude data/dsv4pro_train50_s29.jsonl data/dsv4pro_probe25_s29.jsonl \
    --out data/dsv4pro_selection220_judged.json
```

`--pairs-from <existing set>` re-judges exactly that set's pair list instead
of planning a new one. A run accepts a verdict file only when its provenance
says it was judged on stripped rows, or when the rows file its provenance
names (`rows_file`) opens, matches the recorded `rows_sha256`, and carries no
field the pair judge renders. A Nemotron verdict file judged on rows that
never had a label is accepted the second way — so if its `rows_file` still
names the rows under an earlier file name, point `rows_file` at
`data/nemotron_selection220.jsonl` (same bytes, so the same sha256) or the
run refuses it at boot.

## How the loop keeps outcomes out

Implemented in `src/meta_filter/metadata_free.py`:

* **Strip at load.** `load_trajectories` keeps only the metadata keys on an
  allow-list — `source`, `dataset`, `license`, `submission`,
  `submission_bytes`, `api_calls`, `n_messages` (and `problem_statement`,
  which no pool carries natively) — and drops every other key and every
  top-level extra. Train, validation, probe and selection rows all go through
  it, as do the scoring and rule tools. A DSv4Pro row loses `resolved`,
  `f2p_pass`, `p2p_pass`, `exit_status`, `probe_stratum`, `instance_id`,
  `host` and `attempt_idx`; a Nemotron row, whose keys are all on the list, is
  unchanged byte for byte.
* **Instance-free ids.** A DSv4Pro `trajectory_id` is `<instance_id>::att<n>`,
  which would put the task instance in the judge's header, every cache key
  and every run file. Such ids become `traj_` + the first 16 hex digits of
  sha256(`meta-filter/metadata-free/v1:` + id) at load, so instance names do
  not appear in prompts. This is not a secret: the prefix is a fixed
  constant (kept fixed so caches stay reusable), the hash is deterministic so
  every pool, cache and exam agrees, and anyone with the source pool can
  recompute the mapping. Ids that name no instance (Nemotron's
  `nemotron.<uuid>`) are left as they are. The subset builders recompute the
  mapping from the source pool, so a subset still carries the real ids.
* **Same renderers for both pools.** The scorer's metadata block, the pair
  judge's metadata line and the trajectory critic's metadata render only the
  allow-listed keys a row carries. The pair judge's line is the constant
  `{"resolved": null, "f2p_pass": null, "p2p_pass": null, "exit_status": null}`
  for every row: it is part of the prompt every cached verdict was formed
  on, so it stays byte-identical, and it is the only place those names reach
  a prompt.
* **Separate cache keys.** The scorer key records which baseline fields a row
  lacks, and the pair-judge and critic keys record the label view, so the
  stripped view of a DSv4Pro row never reads a verdict formed on a labelled
  view — while a Nemotron row keeps the key it always had.
* **The run directory holds only stripped rows.** `config.yaml` names
  `train_pool.metadata_free.jsonl` in the run directory as `train_data`,
  which is what the tool-using proposer and its tools open. The source pool
  appears only as `train_data_source_sha256`; its path, and the other pools'
  paths and hashes, are in `runs/_metadata_free/<run_id>.json`, outside the
  workspace.
* **Reads outside the workspace invalidate the proposal.** The Codex sandbox
  cannot deny reads, so the harness audit is checked after each proposer leg:
  any path read outside the run directory other than the Python installation
  the tools run on invalidates that iteration's proposal. Its candidates move
  to `runs/_invalidated/<run_id>/` (outside the workspace) with the paths it
  read, nothing is scored, and the iteration counts as a proposer failure —
  three in a row end the run.
* **Refusals at boot.** An initial filter that names an outcome field, and a
  selection verdict file that passes neither acceptance route above.
* **Strict precheck.** A candidate whose `rules.py`, `policy.py` or
  `rubric.md` names an outcome field is invalid and never scored. `resolved`
  counts only when shaped like a key (quoted, backticked, `metadata.resolved`,
  a `<!-- reads: -->` entry), since the plain word is English.

## The initial filters

* `filters/initial_blank_wired_v2` — one generic rubric criterion, one
  illustrative rule, and a policy that prices rules into the score.
* `filters/initial_ranking_nometa` — four transcript criteria (Task
  Grounding, Edit Justification, Verification Completeness, Error Recovery),
  seven transcript rules after the Nemotron 3 Ultra filtering heuristics, and
  a policy composing them. The submission rule reads the submitted diff
  alone, so every rule fires the same way on both pools.

## Limits worth knowing

* The Codex harness sandbox confines writes, not reads, so a proposer that
  opened a raw pool under `data/` would see its labels before the audit
  catches it. The audit throws that proposal away; it cannot un-read it for
  the model. Nothing in the prompt, the config or the run directory points
  it there.
* Transcripts are the agent's own text and are not rewritten: repository
  names and file paths inside them stay.
