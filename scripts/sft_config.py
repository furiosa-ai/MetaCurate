"""Render a subset's training config so it names ITSELF.

Every packaged subset ships an `sft_base_lr1e-5_3ep.yaml` that the training
job runs verbatim, rendered from the superset's template. A bare copy would
ship the template's own `output_dir`, W&B `name` and artifact `path_prefix`,
so two subsets trained from their shipped files would write checkpoints to
one directory, log to one W&B run and upload over each other, with no error.

What this module changes and what it must not: the recipe's SETTINGS — lr,
epochs, batch, cap, schedule — are carried through byte for byte, because
subsets are comparable only if they share them. Only identifiers change.
Nothing here invents a hyperparameter, and `render_sft_yaml` refuses to
return a config whose identifiers it could not rewrite.
"""

from __future__ import annotations

import re
from pathlib import Path

SFT_YAML_NAME = "sft_base_lr1e-5_3ep.yaml"

_SLUG_SUFFIX = "-qwen35-9b-lr1e-5-3ep"


def template_identifiers(template: str) -> tuple[str, str]:
    """The (slug, bare pool name) this template carries.

    Read off the template's `name:` or `output_dir:` line, whichever ends in
    `<name>-qwen35-9b-lr1e-5-3ep`.

    Raises:
        ValueError: No such identifier: substitution would match nothing and
            an unchanged config would ship.
    """

    for line in template.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("name:", "output_dir:")):
            continue
        value = stripped.split(":", 1)[1].strip().strip("\"'")
        value = value.rsplit("/", 1)[-1]
        if value.endswith(_SLUG_SUFFIX):
            return value, value[: -len(_SLUG_SUFFIX)].replace("-", "_")
    raise ValueError(
        f"the template names no `<name>{_SLUG_SUFFIX}` identifier, so nothing would be "
        "renamed and the subset would ship a config naming someone else"
    )


def slug_for(subset_name: str) -> str:
    """`dsv4pro_blank_1k` -> `dsv4pro-blank-1k-qwen35-9b-lr1e-5-3ep`."""

    stem = subset_name.split("@")[0].replace("_", "-")
    return f"{stem}{_SLUG_SUFFIX}"


def render_sft_yaml(template: str, subset_name: str) -> str:
    """`template` with every identifier pointed at `subset_name`.

    The guard asserts the substitution actually fired, and fired everywhere
    the template said it would: the reachable failure is a template that
    drifts (renamed, or `output_dir` becomes `save_dir`) so that
    substitution matches nothing and a silently unchanged config ships.
    """

    slug, bare = template_identifiers(template)

    if subset_name.split("@")[0] == bare:
        # The template packaging itself. Nothing to rewrite, and rewriting
        # would rename the very run the template is named after.
        return template

    sites = template.count(slug)
    # The @-qualified pool name first, so the longer match wins; the bare
    # name survives in `tags:` and is replaced after it.
    out = re.sub(re.escape(bare) + r"@[\w.-]+", subset_name, template)
    out = out.replace(slug, slug_for(subset_name))
    out = out.replace(bare, subset_name.split("@")[0])

    landed = out.count(slug_for(subset_name))
    if landed != sites:
        raise ValueError(
            f"{subset_name}: expected {sites} identifier sites, rewrote {landed}. "
            "The template's shape changed; check it by hand before shipping."
        )
    return out


def write_sft_yaml(template_path, dest_dir, subset_name) -> None:
    """Render the superset's template into `dest_dir` under this subset's name."""

    template = Path(template_path).read_text(encoding="utf-8")
    Path(dest_dir, SFT_YAML_NAME).write_text(
        render_sft_yaml(template, subset_name), encoding="utf-8"
    )
