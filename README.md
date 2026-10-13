# MetaCurate

MetaCurate learns a filter that picks SFT data from coding-agent trajectories.
An LLM rewrites the filter, and a rewrite is kept only if it agrees more often
with a pairwise LLM judge on held-out pairs. Test results are never used.

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
