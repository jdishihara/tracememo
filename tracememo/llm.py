"""LLM client interface with an Anthropic implementation and a fake for tests.

All LLM calls in tracememo go through :class:`LLMClient`, so no test needs the real API.
The model name always comes from configuration (``llm.model`` in ``project.yaml``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tracememo.config import LLMConfig

FAKE_ENV = "TRACEMEMO_FAKE_LLM"
"""If set to a path of a JSON file holding a list of canned responses, the CLI uses a fake."""


class LLMClient(Protocol):
    """Minimal text-in, text-out interface used by the drafter and the claim checker."""

    model: str

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        """Return the model's text reply to a single user message."""
        ...

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int | None = None
    ) -> Any:
        """Return the model's reply parsed as JSON conforming to ``schema``."""
        ...


@dataclass
class LLMCall:
    """Record of one call made to a client (used by tests and evaluation scripts)."""

    system: str
    user: str
    response: str


@dataclass
class FakeLLM:
    """Scripted client: replies with canned responses in order, or via a callable."""

    responses: list[str] = field(default_factory=list)
    responder: Callable[[str, str], str] | None = None
    model: str = "fake"
    calls: list[LLMCall] = field(default_factory=list)

    def _reply(self, system: str, user: str) -> str:
        if self.responder is not None:
            text = self.responder(system, user)
        elif self.responses:
            text = self.responses.pop(0)
        else:
            raise RuntimeError("FakeLLM has no responses left")
        self.calls.append(LLMCall(system=system, user=user, response=text))
        return text

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        """Return the next canned response."""
        return self._reply(system, user)

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int | None = None
    ) -> Any:
        """Return the next canned response parsed as JSON."""
        return json.loads(self._reply(system, user))


class AnthropicLLM:
    """Client backed by the Anthropic Python SDK (``ANTHROPIC_API_KEY`` from the environment)."""

    def __init__(self, cfg: LLMConfig) -> None:
        import anthropic

        self.model = cfg.model
        self.max_tokens = cfg.max_tokens
        self._client = anthropic.Anthropic()

    def _create(self, system: str, user: str, max_tokens: int | None, **extra: Any) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **extra,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused the request ({response.stop_details})")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("model output was truncated; raise llm.max_tokens")
        return "".join(block.text for block in response.content if block.type == "text")

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        """Single-turn text completion."""
        return self._create(system, user, max_tokens)

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int | None = None
    ) -> Any:
        """Single-turn completion constrained to a JSON schema (structured outputs)."""
        text = self._create(
            system,
            user,
            max_tokens,
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        return json.loads(text)


def make_client(cfg: LLMConfig) -> LLMClient:
    """Build the configured client, or a :class:`FakeLLM` if ``TRACEMEMO_FAKE_LLM`` is set."""
    fake = os.environ.get(FAKE_ENV)
    if fake:
        responses = json.loads(Path(fake).read_text(encoding="utf-8"))
        return FakeLLM(responses=list(responses))
    return AnthropicLLM(cfg)
