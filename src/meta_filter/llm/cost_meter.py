"""Who spent what, per role, while a run was executing.

A run's four LLM roles have wildly different unit costs and wildly different
call counts, and `estimated_filter_cost` measures the scorer cache's state
rather than money. So "a run cost $50" could not otherwise be broken into the
proposer call that cost $3 and the 900 scoring calls that cost the rest, and
a solver swap could not be priced before it was run.

The measurement lives in a wrapper rather than in the clients because the
clients are shared: the rubric critic is handed the *same object* as the
proposer (`optimizer.py`), so a counter on the client itself would file both
under one name. Wrapping lets the same underlying client be metered twice
under two roles, which is exactly the situation.

The ledger is a process-level singleton. One `cli run` is one process and one
Optimizer, and role names are unique within it, so there is nothing to
disambiguate; and a shared accumulator is what lets a *clone* of a client keep
reporting to its original role. The agentic proposer calls
`client.with_workspace(run_dir)` on every call, which returns a new client —
with a per-instance counter, every agentic proposer call would be recorded
against an object that is discarded moments later.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from typing import Any

logger = logging.getLogger(__name__)

# The roles a run reports, in the order their columns appear. Fixed rather
# than discovered so `summary.csv` has the same header whether or not a given
# role made a call this iteration.
ROLES: tuple[str, ...] = ("solver", "judge", "proposer", "critic")


@dataclass
class RoleCost:
    """One role's spend."""

    calls: int = 0
    cost_usd: float = 0.0
    tokens: int = 0
    # Calls whose model has neither a reported cost nor a list price. Their
    # 0.0 is in `cost_usd` and is a hole, so a total is only complete when
    # this is zero.
    unpriced_calls: int = 0

    def __sub__(self, other: RoleCost) -> RoleCost:
        return RoleCost(
            calls=self.calls - other.calls,
            cost_usd=round(self.cost_usd - other.cost_usd, 6),
            tokens=self.tokens - other.tokens,
            unpriced_calls=self.unpriced_calls - other.unpriced_calls,
        )


class CostLedger:
    """Thread-safe per-role totals. Scoring runs in a thread pool."""

    def __init__(self) -> None:
        self._roles: dict[str, RoleCost] = {}
        self._lock = threading.Lock()

    def add(self, role: str, response: Any) -> None:
        cost = float(getattr(response, "cost_usd", 0.0) or 0.0)
        source = str(getattr(response, "cost_source", "unpriced") or "unpriced")
        tokens = int(getattr(response, "total_tokens", 0) or 0)
        calls = int(getattr(response, "calls", 1) or 1)
        with self._lock:
            entry = self._roles.setdefault(role, RoleCost())
            entry.calls += calls
            entry.cost_usd = round(entry.cost_usd + cost, 6)
            entry.tokens += tokens
            if source == "unpriced":
                entry.unpriced_calls += calls

    def snapshot(self) -> dict[str, RoleCost]:
        """A frozen copy of every role's running total."""

        with self._lock:
            return {role: replace(entry) for role, entry in self._roles.items()}

    def since(self, mark: dict[str, RoleCost]) -> dict[str, RoleCost]:
        """Per-role spend since `mark` was taken."""

        now = self.snapshot()
        roles = set(now) | set(mark)
        return {role: now.get(role, RoleCost()) - mark.get(role, RoleCost()) for role in roles}

    def columns(self, mark: dict[str, RoleCost] | None = None) -> dict[str, float]:
        """`{"cost_solver": ..., ...}` for every role, spent or not.

        A role that made no call reports 0.0, which is true: the column means
        "this role cost nothing this iteration", and an absent column would
        instead mean the schema changed mid-run.
        """

        totals = self.since(mark) if mark is not None else self.snapshot()
        return {f"cost_{role}": round(totals.get(role, RoleCost()).cost_usd, 6) for role in ROLES}

    def unpriced_calls(self, mark: dict[str, RoleCost] | None = None) -> int:
        totals = self.since(mark) if mark is not None else self.snapshot()
        return sum(entry.unpriced_calls for entry in totals.values())

    def reset(self) -> None:
        with self._lock:
            self._roles.clear()


LEDGER = CostLedger()


@dataclass
class MeteredClient:
    """An `LLMClient` that reports every call's cost under one role name.

    Everything but `generate` is delegated, so a metered client is still a
    Claude Code client as far as the agentic proposer is concerned — it keeps
    `with_workspace`, `is_agentic`, `allowed_tools` and `audit_path`. Code
    that needs the real object (to name the provider in the run record, say)
    calls `unwrap`.
    """

    client: Any
    role: str
    ledger: CostLedger = LEDGER

    @property
    def model(self) -> str:
        return getattr(self.client, "model", "")

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        response = self.client.generate(prompt, **kwargs)
        self.ledger.add(self.role, response)
        return response

    def with_workspace(self, workspace: Any) -> MeteredClient:
        """Clone the inner client, keep reporting under the same role."""

        return replace(self, client=self.client.with_workspace(workspace))

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this wrapper does not define. Guarding
        # the private names keeps a half-initialized dataclass (and copy /
        # pickle, which probe for dunders) from recursing through `client`.
        if name.startswith("_") or name in {"client", "role", "ledger"}:
            raise AttributeError(name)
        return getattr(self.client, name)


def metered(client: Any, role: str, ledger: CostLedger | None = None) -> MeteredClient:
    """Wrap `client` so its calls are billed to `role`.

    A client that is already metered is re-wrapped around its *inner* client
    rather than around the meter, so one underlying client shared by two roles
    is counted once per role instead of twice for the outer one.
    """

    return MeteredClient(client=unwrap(client), role=role, ledger=ledger or LEDGER)


def replace_client(client: Any, **changes: Any) -> Any:
    """`dataclasses.replace` that sees through the meter.

    The meter delegates attribute reads to the inner client but does not own
    its fields, so `replace(meter, timeout=...)` raises TypeError. Replace the
    inner client and keep the meter (and its role) around the result. A plain
    client is replaced as usual.
    """

    if isinstance(client, MeteredClient):
        return replace(client, client=replace_client(client.client, **changes))
    return replace(client, **changes)


def unwrap(client: Any) -> Any:
    """The real client behind any number of meters."""

    while isinstance(client, MeteredClient):
        client = client.client
    return client
