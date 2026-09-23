"""LLM interface: fake client behaviour and the env-var hook."""

import json
from pathlib import Path

import pytest

from tracememo.config import LLMConfig
from tracememo.llm import FAKE_ENV, FakeLLM, make_client


def test_fake_in_order_and_records_calls() -> None:
    fake = FakeLLM(responses=["a", '{"x": 1}'])
    assert fake.complete("s", "u") == "a"
    assert fake.complete_json("s", "u2", {}) == {"x": 1}
    assert [c.user for c in fake.calls] == ["u", "u2"]
    with pytest.raises(RuntimeError, match="no responses"):
        fake.complete("s", "u")


def test_fake_responder() -> None:
    fake = FakeLLM(responder=lambda s, u: u.upper())
    assert fake.complete("s", "hi") == "HI"


def test_make_client_uses_env_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "r.json"
    f.write_text(json.dumps(["one"]))
    monkeypatch.setenv(FAKE_ENV, str(f))
    client = make_client(LLMConfig())
    assert isinstance(client, FakeLLM) and client.complete("s", "u") == "one"


def test_default_model_from_config() -> None:
    assert LLMConfig().model == "claude-sonnet-5"
    assert LLMConfig(model="claude-opus-5").model == "claude-opus-5"
