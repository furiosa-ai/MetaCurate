# Trajectory Filtering Rubric — blank slate

We are filtering agent rollouts that attempt to solve a single software-repair
task. Score the criterion below on a symmetric +2 / +1 / 0 / -1 / -2 scale and
cite one line of concrete evidence for any non-zero score. When the evidence a
criterion needs is absent or ambiguous, the score is 0 — doubt is 0, never a
guessed +1 or -1.

---

## Criterion: Overall Training Value

How valuable is this trajectory as supervised fine-tuning data?

+2: clearly among the most valuable rollouts available.
+1: somewhat more valuable than a typical rollout.
 0: typical, or the transcript does not support a judgement — doubt is 0.
-1: somewhat less valuable than a typical rollout.
-2: clearly among the least valuable rollouts available.
