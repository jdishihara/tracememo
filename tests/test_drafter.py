"""Drafter: prompt construction, placeholder conversion, checker-driven retry."""

from datetime import UTC, datetime
from pathlib import Path

from tracememo.draft.drafter import (
    build_user_prompt,
    convert_placeholders,
    draft_section,
    load_prompt,
    strip_code_fences,
)
from tracememo.llm import FakeLLM
from tracememo.store import Figure, Provenance, Table, Value, ValueStore

PROMPTS = Path(__file__).resolve().parent.parent / "tracememo" / "draft" / "prompts"


def _store() -> ValueStore:
    prov = Provenance(
        analysis_id="demo.stats",
        analysis_source_sha256="0" * 64,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    s = ValueStore("t")
    s.add_value(
        Value(
            id="demo.low_cm", value=4.6, unit="cm", fmt=".1f", description="Median", provenance=prov
        )
    )
    s.add_value(
        Value(
            id="demo.high_cm", value=7.3, unit="cm", fmt=".1f", description="Mean", provenance=prov
        )
    )
    s.add_figure(Figure(id="demo.fig", caption="Error over time.", provenance=prov))
    s.add_table(Table(id="demo.tab", caption="Events.", provenance=prov))
    return s


def test_prompt_files_exist_and_state_the_rule() -> None:
    for name in ("draft_system", "draft_user", "draft_fix", "claims_system", "claims_user"):
        assert (PROMPTS / f"{name}.txt").exists()
    assert "never write digits" in load_prompt("draft_system").lower()


def test_user_prompt_lists_ids() -> None:
    p = build_user_prompt(_store(), "results", "- error stats\n- figure")
    assert "demo.low_cm | 4.6 | cm | Median" in p
    assert "demo.fig | Error over time." in p and "demo.tab | Events." in p
    assert "{{val:id}}" in p and '"results"' in p


def test_convert_placeholders() -> None:
    text = "Mean {{val:demo.high_cm}} see {{fig:demo.fig}} and {{ tab:demo.tab }}."
    assert convert_placeholders(text, "markdown") == (
        'Mean {{ val("demo.high_cm", with_unit=True) }} see {{ fig("demo.fig") }} '
        'and {{ tab("demo.tab") }}.'
    )
    assert (
        convert_placeholders(text, "latex")
        == r"Mean \valu{demo.high_cm} see \fig{demo.fig} and \tab{demo.tab}."
    )
    assert convert_placeholders(text, "placeholder") == text


def test_strip_code_fences() -> None:
    assert strip_code_fences("```markdown\nhello\n```") == "hello\n"
    assert strip_code_fences("plain") == "plain\n"


def test_clean_draft_needs_one_call() -> None:
    fake = FakeLLM(
        responses=[
            "## Results\n\nThe mean {{val:demo.high_cm}} exceeded the median {{val:demo.low_cm}}.\n"
        ]
    )
    res = draft_section(fake, _store(), "results", "- stats")
    assert res.clean and res.attempts == 1 and len(fake.calls) == 1
    assert "{{val:demo.high_cm}}" in res.text


def test_bad_draft_is_retried_with_findings() -> None:
    fake = FakeLLM(
        responses=[
            "The mean error was 7.3 cm and {{val:demo.nope}} was lower than {{val:demo.high_cm}}.",
            "The mean {{val:demo.high_cm}} was higher than the median {{val:demo.low_cm}}.",
        ]
    )
    res = draft_section(fake, _store(), "results", "- stats", max_fix_rounds=1)
    assert res.clean and res.attempts == 2 and len(fake.calls) == 2
    fix_prompt = fake.calls[1].user
    assert "raw number '7.3'" in fix_prompt and "demo.nope" in fix_prompt
    assert "Previous draft:" in fix_prompt


def test_retry_budget_respected() -> None:
    fake = FakeLLM(responses=["It was 5 cm.", "Still 5 cm.", "Again 5 cm."])
    res = draft_section(fake, _store(), "results", "- stats", max_fix_rounds=1)
    assert not res.clean and res.attempts == 2 and len(fake.calls) == 2
    assert any(f.check == "raw_number" for f in res.findings)
