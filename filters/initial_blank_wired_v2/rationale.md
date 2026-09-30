# Wired blank-slate initial filter

A blank-slate seed plus two things a blank seed lacks: a rule that reads the
real trajectory fields, and a policy that prices rules into the score.

Neither addition carries domain knowledge. The rule says a transcript with
zero turns is not evidence; the policy reads verdicts generically, so a new
rule is priced by one line in `RULE_WEIGHTS` rather than by rewriting
`decide()`. What they carry is the API and the pattern.

## Why `rules.py` gives instructions rather than a field list

`rules.py`'s docstring reaches the proposer verbatim in its prompt. A list of
`metadata` keys stated as fact is only right for the pool it was read off:
the two pools carry different key sets, and on one of them three of the four
keys hold the same value on every row and so cannot order anything. A rule
written against a key the rows do not have fires on none of them, and
nothing — no error, warning or log line — says so.

So the docstring keeps the pool-independent structure (`trajectory_id`,
`task`, `turns`, and `metadata` as a dict whose keys vary by pool), keeps the
warning that an absent field silently returns None on every row, and tells the
proposer to read the keys off an actual row of this run's train pool, to check
that a key varies before ranking on it, and to verify a draft rule's firing
count with `meta_filter.rule_tool` before proposing it.

`rule_no_turns` is a valid illustration of the return contract, and whether
it fires is something the proposer can check for itself.
