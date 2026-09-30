# Initial ranking filter, metadata-free

The non-blank seed of the metadata-free cells (`configs/metadata_free/`,
`dsv4pro_seeded` and `nemotron_seeded`). One seed serves both pools.

It is the composite ranking seed reduced to what a metadata-free row can
support: four rubric criteria that read the transcript and the submitted
diff, the seven Nemotron 3 Ultra p.17 transcript rules, and a policy that
composes the two. Nothing in rubric.md, rules.py or policy.py reads any
metadata key other than `submission`.

| file | what it holds |
| --- | --- |
| `rubric.md` | Task Grounding, Edit Justification, Verification Completeness, Error Recovery |
| `rules.py` | `rule_nemotron_*`, the seven p.17 signals, all REJECT |
| `policy.py` | `50 + 25 * rubric_mean`, -15 per rejecting rule (two at most), -10 at a -2 floor |

Signal 1 (`rule_nemotron_no_valid_submission`) reads the submitted diff
alone: a trajectory whose submission is not a `diff --git` has not ended in
a patch. Every DSv4Pro and every Nemotron row ships a git diff, so the rule
fires the same way on both pools.
