"""Retrospective analysis utilities.

These modules inspect completed run artifacts. They MUST NOT be imported by
the optimizer, the proposer, the validator, or the selector. Whatever labels
they attach (for example, "rubric shrank after a rule was introduced") are
strictly for the researcher — not for the optimizer to see.
"""
