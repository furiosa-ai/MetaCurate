"""A packaged subset's training config must name the subset, not the template.

Every subset ships `sft_base_lr1e-5_3ep.yaml` and the training job runs it
verbatim. A config copied from the superset would carry the template's
`output_dir`, W&B `name` and artifact `path_prefix`, so two subsets trained
from their shipped files would write checkpoints to one directory, log to one
W&B run and upload over each other, with no error. The identifiers must move;
the settings must not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.sft_config import SFT_YAML_NAME, render_sft_yaml, slug_for, write_sft_yaml

BARE = "pool_superset"
SLUG = "pool-superset-qwen35-9b-lr1e-5-3ep"

TEMPLATE = f"""\
# SFT recipe — {SLUG}
#   cp .../subsets/{BARE}@000101/train.jsonl ./
#     --config {SLUG}.yaml \\
data:
  dataset_path: null  # set via --data (subsets/{BARE}@000101/train.jsonl)
training:
  output_dir: outputs/{SLUG}
  num_train_epochs: 3
  learning_rate: 1.0e-5
  per_device_train_batch_size: 1
wandb:
  name: {SLUG}
  tags: [filtering, {BARE}, qwen3.5-9b, sft, lr1e-5, 3ep]
upload:
  path_prefix: subsets/{BARE}@000101/runs/{SLUG}
"""


def test_every_identifier_moves_to_the_new_subset():
    out = render_sft_yaml(TEMPLATE, "nemotron_blank_1k@000102")

    assert "outputs/nemotron-blank-1k-qwen35-9b-lr1e-5-3ep" in out
    assert "name: nemotron-blank-1k-qwen35-9b-lr1e-5-3ep" in out
    assert "subsets/nemotron_blank_1k@000102/train.jsonl" in out
    assert "nemotron_blank_1k," in out  # the tags entry
    # No mention of the template survives in any of the identifier roles.
    assert SLUG not in out
    assert BARE not in out


def test_no_hyperparameter_changes():
    """Only identifiers move. Subsets are comparable only if the recipe is not."""

    subset = "dsv4pro_seeded_1k"
    out = render_sft_yaml(TEMPLATE, subset)

    # Compare only the lines that carry no identifier in EITHER version.
    def settings(text: str) -> list[str]:
        names = (SLUG, BARE, slug_for(subset), subset)
        return [line for line in text.splitlines() if not any(n in line for n in names)]

    assert settings(out) == settings(TEMPLATE)
    for keep in ("num_train_epochs: 3", "learning_rate: 1.0e-5", "per_device_train_batch_size: 1"):
        assert keep in out


def test_two_subsets_never_collide_on_any_identifier():
    a = render_sft_yaml(TEMPLATE, "nemotron_blank_1k@000102")
    b = render_sft_yaml(TEMPLATE, "nemotron_seeded_1k@000102")

    def field(text: str, key: str) -> str:
        return next(line.strip() for line in text.splitlines() if line.strip().startswith(key))

    for key in ("output_dir:", "name:", "path_prefix:"):
        assert field(a, key) != field(b, key), f"{key} collides between the two subsets"


def test_a_template_that_names_nothing_is_refused():
    """The reachable failure is drift: the superset's config renamed so that
    substitution would match nothing and an unchanged file would ship."""

    with pytest.raises(ValueError, match="names no"):
        render_sft_yaml("training:\n  output_dir: outputs/something-else\n", "nemotron_blank_1k")


def test_packaging_the_superset_itself_is_a_no_op():
    assert render_sft_yaml(TEMPLATE, f"{BARE}@000101") == TEMPLATE


def test_write_sft_yaml_lands_the_rendered_file(tmp_path: Path):
    template_path = tmp_path / "template.yaml"
    template_path.write_text(TEMPLATE, encoding="utf-8")
    dest = tmp_path / "subset"
    dest.mkdir()

    write_sft_yaml(template_path, dest, "nemotron_blank_accept_all")

    written = (dest / SFT_YAML_NAME).read_text(encoding="utf-8")
    assert slug_for("nemotron_blank_accept_all") in written
    assert SLUG not in written


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("nemotron_blank_1k@000102", "nemotron-blank-1k-qwen35-9b-lr1e-5-3ep"),
        ("dsv4pro_seeded_top_1k", "dsv4pro-seeded-top-1k-qwen35-9b-lr1e-5-3ep"),
    ],
)
def test_slug_drops_the_date_and_hyphenates(subset: str, expected: str):
    assert slug_for(subset) == expected


def test_the_subset_builder_renders_rather_than_copies():
    source = (Path(__file__).resolve().parents[1] / "scripts/build_ranking_subset.py").read_text(
        "utf-8"
    )
    assert "write_sft_yaml(" in source
    assert f'copyfile(\n        args.superset_dir / "{SFT_YAML_NAME}"' not in source
