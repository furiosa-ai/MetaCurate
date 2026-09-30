"""Select packaged training rows out of a rendered superset.

A subset is delivered as rows copied byte for byte out of a superset
directory that already holds every pool row rendered for training
(`<superset>/train.jsonl`, with the ids it covers listed in
`<superset>/manifest.json`), so every subset cut from one pool shares one
formatting. `index_superset` builds the row_id -> raw line map that
`scripts.build_ranking_subset` copies from.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def index_superset(train_jsonl: Path) -> dict[str, str]:
    """row_id -> raw jsonl line, discovering which field carries the id.

    The superset's manifest lists the row ids it covers. Which record field
    holds that string is checked against the manifest rather than assumed;
    an id of the form `<instance>::att<n>` is also matched as
    `<instance>.<n>`.

    Raises:
        SystemExit: Fewer than 99% of the manifest's ids could be keyed, which
            means the id mapping is wrong.
    """

    manifest = json.loads((train_jsonl.parent / "manifest.json").read_text(encoding="utf-8"))
    wanted = set(manifest["row_ids"])
    index: dict[str, str] = {}
    candidates: dict[str, int] = {}
    with train_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            hit = None
            for field in ("row_id", "trajectory_id", "id"):
                val = row.get(field)
                if isinstance(val, str) and val in wanted:
                    hit = val
                    candidates[field] = candidates.get(field, 0) + 1
                    break
            if hit is None:
                # Reconstruct from parts if no field matches directly.
                inst = row.get("instance_id")
                tid = str(row.get("trajectory_id") or "")
                if inst and tid.startswith(str(inst)):
                    cand = tid.replace("::att", ".")
                    if cand in wanted:
                        hit = cand
                        candidates["reconstructed"] = candidates.get("reconstructed", 0) + 1
            if hit is not None:
                index[hit] = line
    logger.info("superset index: %d/%d rows matched via %s", len(index), len(wanted), candidates)
    if len(index) < len(wanted) * 0.99:
        raise SystemExit(
            f"only {len(index)} of {len(wanted)} superset rows could be keyed — "
            "the row_id mapping assumption is wrong; inspect one row before uploading anything"
        )
    return index
