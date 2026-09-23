"""Draft a report section with an LLM, using value placeholders instead of numbers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from tracememo.check.models import Finding
from tracememo.check.numbers import check_text
from tracememo.llm import LLMClient
from tracememo.store.models import Manifest
from tracememo.store.store import ValueStore

PROMPTS_DIR = Path(__file__).parent / "prompts"
PLACEHOLDER = re.compile(r"\{\{\s*(?P<kind>val|fig|tab):(?P<id>[a-z0-9_]+(?:\.[a-z0-9_]+)+)\s*\}\}")


def load_prompt(name: str) -> str:
    """Read a prompt file from ``draft/prompts``."""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def describe_store(store: ValueStore) -> tuple[str, str, str]:
    """Plain-text listings of values, figures and tables for the prompt."""
    values = "\n".join(
        f"{v.id} | {v.formatted()} | {v.unit or '-'} | {v.description or '-'}"
        for v in (store.values[k] for k in sorted(store.values))
    )
    figures = "\n".join(
        f"{f.id} | {f.caption}" for f in (store.figures[k] for k in sorted(store.figures))
    )
    tables = "\n".join(
        f"{t.id} | {t.caption}" for t in (store.tables[k] for k in sorted(store.tables))
    )
    return values or "(none)", figures or "(none)", tables or "(none)"


def build_user_prompt(store: ValueStore, section: str, outline: str) -> str:
    """Fill the drafting prompt template."""
    values, figures, tables = describe_store(store)
    return load_prompt("draft_user").format(
        section=section, outline=outline.strip(), values=values, figures=figures, tables=tables
    )


def strip_code_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence if the model added one."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip() + "\n"


def convert_placeholders(text: str, syntax: str) -> str:
    """Turn ``{{val:id}}`` placeholders into Jinja (``markdown``) or LaTeX macros."""
    if syntax == "markdown":
        repl = {
            "val": '{{ val("%s", with_unit=True) }}',
            "fig": '{{ fig("%s") }}',
            "tab": '{{ tab("%s") }}',
        }
    elif syntax == "latex":
        repl = {"val": r"\valu{%s}", "fig": r"\fig{%s}", "tab": r"\tab{%s}"}
    elif syntax == "placeholder":
        return text
    else:
        raise ValueError(f"unknown syntax {syntax!r}")
    return PLACEHOLDER.sub(lambda m: repl[m.group("kind")] % m.group("id"), text)


@dataclass
class DraftResult:
    """A drafted section and the checker findings on the final attempt."""

    text: str
    findings: list[Finding] = field(default_factory=list)
    attempts: int = 1

    @property
    def clean(self) -> bool:
        """True when the deterministic checker found no errors."""
        return not any(f.severity == "error" for f in self.findings)


def draft_check(
    text: str, manifest: Manifest, source_hashes: dict[str, str] | None = None
) -> list[Finding]:
    """Run the deterministic checks that apply to a fresh draft (numbers, ids, comparisons)."""
    findings = check_text(text, "draft", manifest, "markdown", None, source_hashes or {})
    return [f for f in findings if f.check != "stale"]


def draft_section(
    client: LLMClient,
    store: ValueStore,
    section: str,
    outline: str,
    max_fix_rounds: int = 1,
) -> DraftResult:
    """Draft ``section`` from ``outline`` with placeholders; retry once on checker errors."""
    system = load_prompt("draft_system")
    user = build_user_prompt(store, section, outline)
    text = strip_code_fences(client.complete(system, user))
    manifest = store.to_manifest()
    findings = draft_check(text, manifest)
    attempts = 1
    while any(f.severity == "error" for f in findings) and attempts <= max_fix_rounds:
        report = "\n".join(
            f"- line {f.line}: {f.message}" for f in findings if f.severity == "error"
        )
        fix = load_prompt("draft_fix").format(findings=report, draft=text)
        text = strip_code_fences(client.complete(system, user + "\n\n" + fix))
        findings = draft_check(text, manifest)
        attempts += 1
    return DraftResult(text=text, findings=findings, attempts=attempts)
