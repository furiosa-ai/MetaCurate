"""Per-role spend accounting.

The two cases that decide whether the numbers add up. One underlying client
is handed to two roles — the rubric critic gets the proposer's client — so a
counter on the client would file both under one name and a naive wrap would
count the proposer's calls twice. And the agentic proposer replaces its
client on every call via `with_workspace`, so a per-instance counter records
the spend against an object that is thrown away.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest

from meta_filter.llm.base import LLMResponse
from meta_filter.llm.cost_meter import ROLES, CostLedger, metered, unwrap


@dataclass
class _Client:
    model: str = "stub-model"
    cost: float = 0.01
    cost_source: str = "openrouter"
    workspace: Any = None
    calls: int = 0

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text="ok",
            prompt_tokens=10,
            completion_tokens=5,
            model=self.model,
            cost_usd=self.cost,
            cost_source=self.cost_source,
        )

    def with_workspace(self, workspace) -> _Client:
        return replace(self, workspace=workspace)

    @property
    def is_agentic(self) -> bool:
        return True


def _ledger() -> CostLedger:
    return CostLedger()


def test_a_call_is_billed_to_its_role() -> None:
    ledger = _ledger()
    metered(_Client(), "solver", ledger).generate("hi")

    assert ledger.columns()["cost_solver"] == pytest.approx(0.01)
    assert ledger.columns()["cost_judge"] == 0.0


def test_every_role_gets_a_column_even_when_it_never_called() -> None:
    """A stable header beats an honest absence: a missing column reads as a
    schema change, not as a role that stayed home."""

    columns = _ledger().columns()
    assert list(columns) == [f"cost_{role}" for role in ROLES]
    assert set(columns.values()) == {0.0}


def test_one_client_shared_by_two_roles_is_counted_once_per_role() -> None:
    """The rubric critic is handed the proposer's own client object."""

    ledger = _ledger()
    client = _Client()
    proposer = metered(client, "proposer", ledger)
    critic = metered(proposer, "critic", ledger)

    proposer.generate("propose")
    critic.generate("criticize")

    columns = ledger.columns()
    assert columns["cost_proposer"] == pytest.approx(0.01)
    assert columns["cost_critic"] == pytest.approx(0.01)
    assert client.calls == 2


def test_a_workspace_clone_keeps_reporting_to_its_role() -> None:
    ledger = _ledger()
    proposer = metered(_Client(), "proposer", ledger)

    agent = proposer.with_workspace("/tmp/run")
    agent.generate("propose")

    assert ledger.columns()["cost_proposer"] == pytest.approx(0.01)
    assert unwrap(agent).workspace == "/tmp/run"


def test_the_wrapper_still_looks_like_the_client_it_wraps() -> None:
    client = _Client(model="minimax/minimax-m3")
    wrapped = metered(client, "judge", _ledger())

    assert wrapped.model == "minimax/minimax-m3"
    assert wrapped.is_agentic is True
    assert unwrap(wrapped) is client
    with pytest.raises(AttributeError):
        _ = wrapped.no_such_attribute


def test_unpriced_calls_are_counted_so_a_total_can_be_doubted() -> None:
    ledger = _ledger()
    meter = metered(_Client(cost=0.0, cost_source="unpriced"), "judge", ledger)
    meter.generate("hi")
    meter.generate("hi")

    assert ledger.columns()["cost_judge"] == 0.0
    assert ledger.unpriced_calls() == 2


def test_a_mark_turns_running_totals_into_this_iteration() -> None:
    ledger = _ledger()
    meter = metered(_Client(), "solver", ledger)
    meter.generate("iteration 0")

    mark = ledger.snapshot()
    meter.generate("iteration 1")
    meter.generate("iteration 1")

    assert ledger.columns(mark)["cost_solver"] == pytest.approx(0.02)
    assert ledger.columns()["cost_solver"] == pytest.approx(0.03)


def test_totals_survive_concurrent_calls() -> None:
    """Scoring fans out across a thread pool, so the counter is shared."""

    from concurrent.futures import ThreadPoolExecutor

    ledger = _ledger()
    meter = metered(_Client(), "solver", ledger)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: meter.generate("hi"), range(200)))

    assert ledger.columns()["cost_solver"] == pytest.approx(2.0)


def test_the_optimizer_raises_the_timeout_through_the_meter(tmp_path):
    """The optimizer raises a short proposer timeout to 3600 s.

    With the cost meter around the client, `dataclasses.replace` on the
    wrapper would raise TypeError and kill every run at startup; the inner
    client owns `timeout`, so that is what has to be replaced.
    """

    from dataclasses import dataclass, replace

    from meta_filter.llm.cost_meter import MeteredClient, metered

    @dataclass
    class _Timed:
        timeout: float = 1800.0

        def generate(self, prompt, **kwargs):  # pragma: no cover - never called
            raise AssertionError

    wrapped = metered(_Timed(), "proposer")
    assert isinstance(wrapped, MeteredClient)
    assert wrapped.timeout == 1800.0
    # This is the exact expression the optimizer uses.
    fixed = replace(wrapped, client=replace(wrapped.client, timeout=3600.0))
    assert fixed.timeout == 3600.0 and fixed.role == "proposer"


def test_replace_client_sees_through_the_meter_for_every_harness(tmp_path):
    """Both harness clients get re-shaped with dataclasses.replace by the
    agentic proposer (`_with_tools`), the agentic critic and the optimizer.
    With the meter in place every one of those would raise TypeError and kill
    a run at its first proposer call."""

    from meta_filter.critics.agentic import _agentic
    from meta_filter.llm.claude_code import ClaudeCodeHeadlessClient
    from meta_filter.llm.codex_cli import CodexCliClient
    from meta_filter.llm.cost_meter import MeteredClient, metered, replace_client, unwrap
    from meta_filter.propose.agentic_proposer import _with_tools

    for inner in (ClaudeCodeHeadlessClient(model="x"), CodexCliClient(model="y")):
        m = metered(inner, "proposer")
        shaped = _with_tools(m, ("Read", "Write"))
        assert isinstance(shaped, MeteredClient) and shaped.role == "proposer"
        assert unwrap(shaped).allowed_tools == ("Read", "Write")
        assert unwrap(shaped).append_system_prompt
        assert unwrap(m).allowed_tools != ("Read", "Write")  # original untouched
        timed = replace_client(m, timeout=3600.0)
        assert timed.timeout == 3600.0 and isinstance(timed, MeteredClient)
    critic = _agentic(metered(ClaudeCodeHeadlessClient(model="x"), "critic"), tmp_path)
    assert unwrap(critic).scratch_dir == tmp_path and critic.role == "critic"
    # A plain client still works the ordinary way.
    plain = replace_client(ClaudeCodeHeadlessClient(model="x"), timeout=7.0)
    assert isinstance(plain, ClaudeCodeHeadlessClient) and plain.timeout == 7.0


def test_the_gpt56_pro_and_terra_tiers_are_priced():
    from meta_filter.llm.pricing import price_for

    assert price_for("openai/gpt-5.6-sol-pro") == price_for("openai/gpt-5.6-sol")
    assert price_for("openai/gpt-5.6-luna-pro") == price_for("openai/gpt-5.6-luna")
    terra = price_for("openai/gpt-5.6-terra")
    assert terra is not None and terra.output == 12.0 and terra.input == 2.0
