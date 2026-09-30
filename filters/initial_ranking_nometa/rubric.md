# Trajectory Filtering Rubric — composite v1, metadata-free (SWE agent domain)

The non-blank seed of the metadata-free cells. Four criteria, each read off
the transcript and the submitted diff.

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

Every criterion here reads the transcript and the submitted diff and nothing
else. Do not reward a trajectory for *claiming* its tests passed: the evidence
is what the agent did and what it shipped.

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
