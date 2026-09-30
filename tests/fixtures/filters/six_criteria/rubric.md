# Trajectory Filtering Rubric (test fixture, SWE agent domain)

We are filtering agent rollouts that attempt to solve a single software-repair
task (typically an issue / PR description). Each criterion is scored on a
symmetric +2 / +1 / 0 / -1 / -2 scale. Anchors describe both what a *good*
and a *bad* trajectory looks like so the same rubric supports FP and FN
critics.

Score every criterion and cite one line of concrete evidence per non-zero
score. **When the evidence a criterion needs is absent or ambiguous, the
score is 0 — doubt is 0, never a guessed +1 or -1.**

Navigation-only actions (`cd`, `pwd`, `ls`) are structural bookkeeping — do
NOT count them as "information gain" or "repetition"; look at the *content*
actions (`grep`, `read/cat`, `edit/apply_patch`, `test/pytest`, `python`,
`run_tests`, `sed`, `awk`) instead.

Two criteria read reference material that ships with the task
(the problem statement and a reference patch). They judge the *task*
and the agent's *relation* to the reference — they must never be scored by
similarity alone, and when the reference fields are missing the score is 0
with a note saying the reference was unavailable.

---

## Criterion: Task Grounding

Does the agent read and internalize the task before acting?

+2: Agent quotes / paraphrases the task and localizes the affected symbol
    within the first ~5 content actions.
+1: Agent references the task but jumps to editing without full localization.
 0: Task is acknowledged but not clearly used to steer actions; or the
    transcript is too truncated to tell — doubt is 0.
-1: Agent starts editing before demonstrating any understanding of the task.
-2: Agent hallucinates a task or edits files unrelated to the request.

## Criterion: Query/PR Quality

Is the task itself worth training on: is the problem statement specific
enough to act on, and is the reference (gold) patch a coherent fix for what
the statement describes?

+2: The problem statement names concrete symptoms (error text, reproducible
    example, expected vs actual behavior) AND the gold patch, when present,
    plainly addresses that described behavior.
+1: The statement is actionable but underspecified in places (e.g. no repro,
    vague expected behavior); the gold patch still matches its topic.
 0: Quality cannot be judged — the statement or gold patch is missing,
    truncated, or generic boilerplate. Doubt is 0.
-1: The statement is ambiguous enough that two reasonable fixes would differ
    (no repro, no expected behavior, mixed concerns), or the gold patch fixes
    something the statement never mentions.
-2: The statement is incoherent, self-contradictory, or the gold patch is
    unrelated to it (e.g. pure refactor / version bump against a bug report):
    any agent behavior on this task is unlearnable noise.

## Criterion: Gold-Patch Relation

Does the agent's submitted diff engage the same defect the reference patch
fixes — same file/function region, or a defensible alternative route?

+2: The submitted diff edits the same file(s) and the same function/region
    as the reference patch, or the transcript shows the agent found the
    gold patch's site and chose a different layer with stated reasons that
    the tests then support.
+1: Same file(s) as the gold patch but a different region, and the edit
    plausibly reaches the same behavior; or a different file with an explicit,
    test-backed argument connecting it to the defect.
 0: the reference patch is absent, or either diff is unreadable — the
    relation cannot be judged. Doubt is 0. Never score similarity for its
    own sake: a textually different but behaviorally equivalent fix is not
    a defect.
-1: The submitted diff shares no files with the gold patch and the transcript
    offers no evidence (test run, traced call path) connecting the edited
    site to the defect.
-2: The submitted diff works around the defect instead of engaging it —
    hardcodes expected output, edits/disables the failing test, or patches a
    caller to mask the symptom the gold patch fixes at the source.

## Criterion: Edit Justification

Do the applied edits address the task described in the trajectory?

+2: Every edit maps to a specific claim about the bug/feature; edits are
    surgical (few, targeted hunks).
+1: Edits address the task but include some unrelated churn.
 0: Edits partially address the task; hard to tell if the fix is on the
    right layer — doubt is 0.
-1: Edits are speculative — the agent guesses and reverts multiple times.
-2: Edits contradict the task, or the submitted patch is empty / trivial
    (<400 bytes for a non-trivial task) with no verification.

## Criterion: Verification Completeness

Does the trajectory actually verify that the change works?

+2: Runs relevant tests (`pytest tests/…`, `python -m pytest`, project's
    build/lint) AFTER the final edit and observes success.
+1: Runs some tests (smoke / lint) but not the full relevant suite.
 0: Runs tests but only intermittently; verification is unclear — doubt is 0.
-1: Runs tests early but not after the final edit, OR declares completion
    without running any test after the edit.
-2: Declares completion (`task complete`, `all done`, submits patch) with no
    test/build invocation anywhere in the trajectory.

## Criterion: Error Recovery

When commands fail, does the agent adapt?

+2: Every non-zero exit or traceback is followed by a diagnostic action
    that materially changes the next step.
+1: Most errors are handled; occasional retry-without-change.
 0: Errors present but mostly benign (search misses etc.), or too few errors
    to judge — doubt is 0.
-1: Same failing command is repeated without a plan change.
-2: Persistent unrecovered errors (repeated identical failing commands
    or unhandled tracebacks in the last third of the trajectory).
