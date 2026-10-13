"""LLM client protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class LLMResponse:
    """A single LLM completion."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    calls: int = 1
    # What this call cost and how that number was arrived at. Every client
    # fills both, because a run's per-role spend is only readable if a zero
    # can be told apart from an unpriced model. `cost_source` is one of:
    #   "openrouter"  — the route's own `usage.cost`, i.e. measured
    #   "harness"     — a CLI's reported spend (valid only where the CLI is
    #                   the thing being billed; see `claude_code.py`)
    #   "list_price"  — rate card x tokens, from `llm/pricing.py`
    #   "mock"        — an offline client; the 0.0 is real
    #   "unpriced"    — no measurement and no rate card; the 0.0 is a hole
    cost_usd: float = 0.0
    cost_source: str = "unpriced"
    # OpenRouter's serving provider ("MiniMax", "Fireworks", ...). Empty on
    # routes that do not name one. Worth recording because the same model
    # slug served by two providers is two different quantizations.
    provider: str = ""
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface used by both proposer and (optionally) solver.

    `cache_system` is a hint to the underlying client that the `system`
    payload is a large *constant* prefix that will be sent unchanged over
    many calls, so it is safe to attach a provider-specific prompt-cache
    marker. Clients that don't support caching may ignore it silently.
    """

    model: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> LLMResponse: ...


class UsageLimitError(RuntimeError):
    """The account cannot make calls at all, and waiting will not help.

    A spend cap, an exhausted credit balance, or a usage limit is neither a
    transient blip nor a misconfiguration: retrying is futile and the run has
    nothing left to do but stop and say why. It exists as its own class
    because `optimizer.py` treats a *streak* of proposer failures as a broken
    configuration and burns iterations discovering the streak — which is the
    right containment for a bad auth mode and the wrong one for a cap, where
    the very first failure is already conclusive. Treated as a streak, a cap
    burns further iterations proving what the first 400 already said, and is
    then reported as a configuration failure — sending whoever reads it to
    look at flags and auth modes rather than at the billing page.
    """


# Environment variable names that look like a credential, whatever service
# they belong to. Harness subprocesses run agent-written shell, so every such
# variable is stripped from their environment except the one the harness
# needs (`scrub_credentials(keep=...)`).
_CREDENTIAL_NAME_RE = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_KEY|ACCESS_KEY|PRIVATE_KEY|DEPLOY_KEY|_PAT)(_|$)"
)


def scrub_credentials(
    env: dict[str, str], *, prefixes: tuple[str, ...] = (), keep: tuple[str, ...] = ()
) -> dict[str, str]:
    """Drop every credential-looking variable, and every name under `prefixes`.

    `keep` names survive both rules. Returns `env`, modified in place.
    """

    for name in list(env):
        if name in keep:
            continue
        if name.startswith(prefixes) or _CREDENTIAL_NAME_RE.search(name):
            env.pop(name, None)
    return env


def usage_limit_detail(body: str | None, markers: tuple[str, ...]) -> str:
    """The cap sentence from a CLI failure body, or `""` if it is not a cap.

    Returns the matched line rather than a bool so the run's last log line
    carries the regain date the CLI reports, which is the one fact the reader
    needs. Matching is case-insensitive.
    """

    body = body or ""
    lowered = tuple(m.lower() for m in markers)
    if not any(m in body.lower() for m in lowered):
        return ""
    for line in reversed(body.strip().splitlines()):
        if any(m in line.lower() for m in lowered):
            return line.strip()[:400]
    return body.strip()[:400]
