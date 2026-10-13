# MetaCurate

MetaCurate evolves a curation filter for selecting SFT data from coding-agent
trajectories, using pairwise LLM judgments. A filter is a set of rules, an
LLM-scored rubric, and a policy that combines them into one score per
trajectory.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m meta_filter.cli run --llm-provider mock \
    --selection-data data/demo_selection.jsonl
```

This runs the loop offline with a mock LLM and writes the result to `runs/`.

## Tests

```bash
uv run pytest
```

## License

[MIT](LICENSE)
