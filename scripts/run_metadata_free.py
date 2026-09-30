"""Launch one run cell from `configs/metadata_free/<cell>.yaml`.

    uv run python -m scripts.run_metadata_free --cell nemotron_blank --seed 29 --dry-run
    uv run python -m scripts.run_metadata_free --cell dsv4pro_seeded

Each cell is one `meta_filter.cli run --config <cell>.yaml` process. The CLI
reads the config's nested keys (seed, iterations, llm, paths); this script
appends the config's `launch.args` (flags the CLI has no config key for) and
`launch.pool` (probe and selection files), and gives the run its id.

`--dry-run` prints the environment (key redacted) and the exact command and
does nothing else: it reads no key, writes no file and starts no process.
Inputs that are missing on this machine are listed on stderr.

A real launch checks every input exists, then starts the run detached with
its log at `runs/_logs/<run_id>.log`. Every LLM call goes through OpenRouter:
the proposer's Codex harness reads `OPENROUTER_API_KEY`, and the solver and
judge use an OpenAI-compatible client pointed at the OpenRouter endpoint. The
key is read from `OPENROUTER_API_KEY` in `~/.env`; no first-party key is
placed in the run's environment.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CONFIG_DIR = Path("configs/metadata_free")
CELLS = ("dsv4pro_blank", "dsv4pro_seeded", "nemotron_blank", "nemotron_seeded")
KEY_PLACEHOLDER = "<openrouter-key>"


def load_cell(cell: str, config_dir: Path = CONFIG_DIR) -> tuple[Path, dict[str, Any]]:
    """The cell's config path and parsed YAML; refuses an unknown cell."""

    if cell not in CELLS:
        raise SystemExit(f"unknown cell {cell!r}; choose from {', '.join(CELLS)}")
    path = Path(config_dir) / f"{cell}.yaml"
    cfg = yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8")) or {}
    if (cfg.get("launch") or {}).get("cell") != cell:
        raise SystemExit(f"{path} names cell {cfg.get('launch', {}).get('cell')!r}, not {cell!r}")
    return path, cfg


def run_id_for(cell: str, seed: int, date_tag: str) -> str:
    return f"run_{date_tag}_metafree_{cell}_s{seed}"


def build_command(
    cell: str, *, seed: int | None = None, date_tag: str, config_dir: Path = CONFIG_DIR
) -> list[str]:
    """The exact `meta_filter.cli run` argv for one cell.

    `seed` other than the config's changes the seed, the run id and the
    memory directory, and — for a cell with `launch.seeded_pools` (the DSv4Pro
    cells, whose train and probe pools are per-seed samples) — the train and
    probe files. A Nemotron cell keeps its single pool at every seed.
    """

    path, cfg = load_cell(cell, config_dir)
    config_seed = int(cfg["seed"])
    seed = config_seed if seed is None else int(seed)
    launch = cfg["launch"]
    pool = launch["pool"]
    pools = pools_for(cfg, seed)
    # fmt: off
    cmd = [
        "uv", "run", "python", "-m", "meta_filter.cli", "run",
        "--config", str(path),
        "--run-id", run_id_for(cell, seed, date_tag),
    ]
    # fmt: on
    if seed != config_seed:
        cmd += ["--seed", str(seed), "--memory-dir", f"memory/{cell}_s{seed}"]
    for flag in launch["args"]:
        cmd += [str(part) for part in flag]
    # fmt: off
    cmd += [
        "--train", str(pools["train_data"]),
        "--probe", str(pools["probe"]),
        "--selection-data", str(pool["selection_data"]),
        "--selection-set", str(pool["selection_set"]),
    ]
    # fmt: on
    return cmd


def pools_for(cfg: dict[str, Any], seed: int) -> dict[str, str]:
    """The train and probe files for `seed`: `launch.seeded_pools` templates
    (`{seed}` filled in) when the cell has them, else the config's own."""

    seeded = cfg["launch"].get("seeded_pools")
    if seeded:
        return {name: str(template).format(seed=seed) for name, template in seeded.items()}
    return {"train_data": cfg["paths"]["train_data"], "probe": cfg["launch"]["pool"]["probe"]}


def check_seeded_pools(cell: str, seed: int, root: Path | None = None) -> None:
    """Refuse a seed whose per-seed pool is not on this machine, by name."""

    root = REPO_ROOT if root is None else root
    _, cfg = load_cell(cell)
    if not cfg["launch"].get("seeded_pools"):
        return
    absent = [p for p in pools_for(cfg, seed).values() if not (root / p).is_file()]
    if absent:
        raise SystemExit(
            f"{cell}: no pool for seed {seed} — {', '.join(absent)} not found. The DSv4Pro "
            "cells' train and probe pools are per-seed samples; build them for this seed "
            "first, or pick a seed that has them"
        )


def inputs(cell: str, seed: int | None = None, config_dir: Path = CONFIG_DIR) -> list[Path]:
    """Every file or directory the cell's run reads at boot."""

    _, cfg = load_cell(cell, config_dir)
    pool = cfg["launch"]["pool"]
    paths = cfg["paths"]
    pools = pools_for(cfg, int(cfg["seed"]) if seed is None else seed)
    return [
        Path(pools["train_data"]),
        Path(paths["validation_data"]),
        Path(paths["initial_filter"]),
        Path(pools["probe"]),
        Path(pool["selection_data"]),
        Path(pool["selection_set"]),
    ]


def missing_inputs(cell: str, seed: int | None = None, root: Path | None = None) -> list[Path]:
    root = REPO_ROOT if root is None else root
    return [p for p in inputs(cell, seed) if not (root / p).exists()]


# First-party credentials and endpoints that must not reach the run: with
# any of them set, a client could bill a different account than the one the
# OpenRouter key names.
_SCRUBBED = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
)


def build_env(key: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """The run's environment: OpenRouter for every role, no first-party key."""

    env = dict(os.environ if base is None else base)
    for name in _SCRUBBED:
        env.pop(name, None)
    env["OPENROUTER_API_KEY"] = key  # the Codex harness (proposer)
    env["OPENAI_API_KEY"] = key  # the OpenAI-compatible solver and judge client
    env["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
    env["PYTHONUNBUFFERED"] = "1"
    return env


def read_env_key(name: str, env_file: Path | None = None) -> str:
    """One key from `~/.env`, read line by line, never sourced or echoed."""

    path = env_file or (Path.home() / ".env")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise SystemExit(f"{name} not found in {path}")


def redacted_env() -> dict[str, str]:
    """What `build_env` sets, with the key shown as a placeholder.

    Built from a placeholder rather than by reading the key and masking it,
    so a dry run never touches the key file.
    """

    env = build_env(KEY_PLACEHOLDER, base={})
    return {
        k: v
        for k, v in env.items()
        if k.startswith(("OPENAI_", "ANTHROPIC_", "OPENROUTER_", "PYTHONUNBUFFERED"))
    }


def launch(cmd: list[str], run_id: str) -> int:
    logs = REPO_ROOT / "runs" / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{run_id}.log"
    env = build_env(read_env_key("OPENROUTER_API_KEY"))
    with log_path.open("ab") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    logger.info("%s -> pid %d, log %s", run_id, proc.pid, log_path)
    return proc.pid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell", choices=CELLS)
    group.add_argument("--all", action="store_true", help="every cell, in CELLS order")
    parser.add_argument("--seed", type=int, default=None, help="default: the config's seed")
    parser.add_argument(
        "--date", default=None, help="run-id date tag, YYYYMMDD (default: today, UTC)"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    date_tag = args.date or datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")
    cells = list(CELLS) if args.all else [args.cell]
    if args.dry_run:
        print("env:", json.dumps(redacted_env()))
        for cell in cells:
            _, cfg = load_cell(cell)
            if args.seed is not None and args.seed != int(cfg["seed"]):
                # A seed other than the config's is a deliberate choice, so a
                # pool that does not exist for it is an error even here.
                check_seeded_pools(cell, args.seed)
            cmd = build_command(cell, seed=args.seed, date_tag=date_tag)
            print(f"# cell {cell}")
            print(shlex.join(cmd))
            for path in missing_inputs(cell, args.seed):
                print(f"# {cell}: missing on this machine: {path}", file=sys.stderr)
        return 0

    for cell in cells:
        _, cfg = load_cell(cell)
        check_seeded_pools(cell, args.seed if args.seed is not None else int(cfg["seed"]))
    missing = {cell: missing_inputs(cell, args.seed) for cell in cells}
    if any(missing.values()):
        detail = "; ".join(
            f"{cell}: {', '.join(map(str, paths))}" for cell, paths in missing.items() if paths
        )
        raise SystemExit(f"refusing to launch, inputs missing — {detail}")
    for cell in cells:
        cmd = build_command(cell, seed=args.seed, date_tag=date_tag)
        _, cfg = load_cell(cell)
        seed = args.seed if args.seed is not None else int(cfg["seed"])
        launch(cmd, run_id_for(cell, seed, date_tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
