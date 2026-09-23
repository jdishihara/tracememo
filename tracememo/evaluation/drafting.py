"""Drafting grounding rate: placeholders required vs. free-form numbers.

Both conditions get the same outline and the same list of values. We measure, per draft, how
many numbers are unsupported by the value store and how many comparison claims point the
wrong way. Condition (a) uses the production drafting prompt; its "numbers" are placeholders
plus any raw digits the model wrote anyway. Condition (b) lets the model write numbers freely;
each number is matched to the closest stored value within the precision it was written at.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from tracememo.check.extract import Ref, prose_mask
from tracememo.check.numbers import (
    DEFAULT_ALLOW,
    NUMBER,
    check_raw_numbers,
    check_unknown_references,
    evaluate_comparisons,
)
from tracememo.draft.drafter import (
    build_user_prompt,
    describe_store,
    draft_check,
    load_prompt,
    strip_code_fences,
)
from tracememo.llm import FakeLLM, LLMClient
from tracememo.report.render import format_with_unit
from tracememo.store.models import Manifest, Value
from tracememo.store.store import ValueStore


def perturbed(v: Value, factor: float, offset: float = 0.0) -> Value:
    """A copy of ``v`` with its number scaled by ``factor`` (ints stay ints, moved by >= 1)."""
    x = float(v.value) * factor + offset
    if isinstance(v.value, int):
        n = int(round(x))
        if n == v.value:
            n += 1
        return Value(id=v.id, value=n, unit=v.unit, fmt=v.fmt)
    return Value(id=v.id, value=x, unit=v.unit, fmt=v.fmt or ".3g")


OUTLINE = """- Summarize localization accuracy: mean, median, 95th percentile and share of samples
  under the threshold
- Compare the horizontal RMSE with the 3D RMSE
- Report the beacon failure events: counts per type, time lost to dropouts, availability
- Point the reader to the error-over-time figure and the events table
"""


@dataclass
class DraftScore:
    """Grounding metrics for one draft."""

    condition: str
    attempt: int
    n_numbers: int = 0
    n_unsupported: int = 0
    n_comparisons: int = 0
    n_wrong_comparisons: int = 0
    n_unverifiable_comparisons: int = 0
    unsupported: list[str] = field(default_factory=list)
    wrong: list[str] = field(default_factory=list)


# --- condition (a): placeholders ------------------------------------------------------------


def score_placeholder(text: str, manifest: Manifest, attempt: int = 1) -> DraftScore:
    """Numbers = placeholders + raw digits; unsupported = raw digits + unknown ids."""
    masked, refs = prose_mask(text, "markdown")
    val_refs = [r for r in refs if r.kind == "val"]
    raw = check_raw_numbers(masked, "draft")
    unknown = check_unknown_references(val_refs, manifest, "draft")
    comps = evaluate_comparisons(masked, refs, manifest)
    return DraftScore(
        condition="placeholders",
        attempt=attempt,
        n_numbers=len(val_refs) + len(raw),
        n_unsupported=len(raw) + len(unknown),
        n_comparisons=sum(c.status in ("ok", "wrong") for c in comps),
        n_wrong_comparisons=sum(c.status == "wrong" for c in comps),
        n_unverifiable_comparisons=sum(c.status not in ("ok", "wrong") for c in comps),
        unsupported=[f.message for f in raw + unknown],
        wrong=[c.text for c in comps if c.status == "wrong"],
    )


# --- condition (b): free-form numbers -------------------------------------------------------


def match_value(number_text: str, manifest: Manifest) -> str | None:
    """Id of a stored numeric value that this written number could be, or None."""
    try:
        x = float(number_text.replace(",", ""))
    except ValueError:
        return None
    decimals = len(number_text.split(".")[1]) if "." in number_text else 0
    best: tuple[float, str] | None = None
    for v in manifest.values.values():
        if not isinstance(v.value, int | float) or isinstance(v.value, bool):
            continue
        tol = max(0.5 * 10 ** (-decimals), 0.005 * abs(float(v.value)))
        err = abs(x - float(v.value))
        if err <= tol and (best is None or err < best[0]):
            best = (err, v.id)
    return best[1] if best else None


def score_freeform(text: str, manifest: Manifest, attempt: int = 1) -> DraftScore:
    """Match every written number to the store; check comparisons between matched numbers."""
    masked, _ = prose_mask(text, "markdown")
    allow = [re.compile(p) for p in DEFAULT_ALLOW]
    numbers: list[tuple[int, int, str, str | None]] = []  # start, end, text, matched id
    for m in NUMBER.finditer(masked):
        line_start = masked.rfind("\n", 0, m.start()) + 1
        line_end = masked.find("\n", m.end())
        line = masked[line_start : line_end if line_end >= 0 else len(masked)]
        rel = m.start() - line_start
        if any(
            a.start() <= rel and m.end() - line_start <= a.end()
            for p in allow
            for a in p.finditer(line)
        ):
            continue
        numbers.append((m.start(), m.end(), m.group(), match_value(m.group(), manifest)))

    # Rebuild the text with @R<n>@ tokens so the comparison logic can be reused.
    pieces, pos, refs = [], 0, []
    for n, (s, e, _num, vid) in enumerate(numbers):
        pieces.append(masked[pos:s])
        pieces.append(f"@R{n}@")
        refs.append(Ref("val", vid or f"unmatched.n{n}", s, e, masked.count("\n", 0, s) + 1))
        pos = e
    pieces.append(masked[pos:])
    tokenized = "".join(pieces)
    comps = evaluate_comparisons(tokenized, refs, manifest)
    unsupported = [num for _, _, num, vid in numbers if vid is None]
    return DraftScore(
        condition="free-form",
        attempt=attempt,
        n_numbers=len(numbers),
        n_unsupported=len(unsupported),
        n_comparisons=sum(c.status in ("ok", "wrong") for c in comps),
        n_wrong_comparisons=sum(c.status == "wrong" for c in comps),
        n_unverifiable_comparisons=sum(c.status not in ("ok", "wrong") for c in comps),
        unsupported=unsupported,
        wrong=[c.text for c in comps if c.status == "wrong"],
    )


# --- drafting -------------------------------------------------------------------------------


def draft_placeholder(
    client: LLMClient, store: ValueStore, outline: str, fix_rounds: int
) -> list[tuple[str, DraftScore]]:
    """Production prompt; returns (text, score) per attempt, so attempt 1 shows the raw rule."""
    manifest = store.to_manifest()
    system = load_prompt("draft_system")
    user = build_user_prompt(store, "results", outline)
    text = strip_code_fences(client.complete(system, user))
    out = [(text, score_placeholder(text, manifest, 1))]
    attempt = 1
    while attempt <= fix_rounds:
        findings = [f for f in draft_check(text, manifest) if f.severity == "error"]
        if not findings:
            break
        report = "\n".join(f"- line {f.line}: {f.message}" for f in findings)
        fix = load_prompt("draft_fix").format(findings=report, draft=text)
        text = strip_code_fences(client.complete(system, user + "\n\n" + fix))
        attempt += 1
        out.append((text, score_placeholder(text, manifest, attempt)))
    return out


def draft_freeform(client: LLMClient, store: ValueStore, outline: str) -> tuple[str, DraftScore]:
    """Free-number prompt; one attempt."""
    values, figures, tables = describe_store(store)
    system = load_prompt("draft_freeform_system")
    user = load_prompt("draft_freeform_user").format(
        section="results", outline=outline.strip(), values=values, figures=figures, tables=tables
    )
    text = strip_code_fences(client.complete(system, user))
    return text, score_freeform(text, store.to_manifest())


def _aggregate(scores: list[DraftScore], label: str) -> dict[str, Any]:
    n_num = sum(s.n_numbers for s in scores)
    n_uns = sum(s.n_unsupported for s in scores)
    n_cmp = sum(s.n_comparisons for s in scores)
    n_wrong = sum(s.n_wrong_comparisons for s in scores)
    return {
        "condition": label,
        "drafts": len(scores),
        "numbers": n_num,
        "unsupported_numbers": n_uns,
        "pct_unsupported": 100.0 * n_uns / n_num if n_num else 0.0,
        "drafts_with_any_unsupported": sum(s.n_unsupported > 0 for s in scores),
        "comparisons_checked": n_cmp,
        "wrong_direction": n_wrong,
        "pct_wrong_direction": 100.0 * n_wrong / n_cmp if n_cmp else 0.0,
        "unverifiable_comparisons": sum(s.n_unverifiable_comparisons for s in scores),
    }


def run(
    client: LLMClient, store: ValueStore, n: int = 20, fix_rounds: int = 1, outline: str = OUTLINE
) -> dict[str, Any]:
    """Draft ``n`` times per condition and aggregate the grounding metrics."""
    first: list[DraftScore] = []
    final: list[DraftScore] = []
    free: list[DraftScore] = []
    drafts: list[dict[str, Any]] = []
    for i in range(n):
        attempts = draft_placeholder(client, store, outline, fix_rounds)
        first.append(attempts[0][1])
        final.append(attempts[-1][1])
        for text, score in attempts:
            drafts.append({"i": i, "text": text, **asdict(score)})
        text, score = draft_freeform(client, store, outline)
        free.append(score)
        drafts.append({"i": i, "text": text, **asdict(score)})
    summary = [
        _aggregate(first, "placeholders required (first attempt)"),
        _aggregate(final, f"placeholders required (+ up to {fix_rounds} fix round)"),
        _aggregate(free, "free-form numbers"),
    ]
    return {
        "n": n,
        "model": client.model,
        "fix_rounds": fix_rounds,
        "summary": summary,
        "drafts": drafts,
    }


def to_markdown(result: dict[str, Any]) -> str:
    """Summary table of both conditions."""
    from tracememo.evaluation.common import md_table

    cols = [
        "condition",
        "drafts",
        "numbers",
        "unsupported_numbers",
        "pct_unsupported",
        "comparisons_checked",
        "wrong_direction",
        "pct_wrong_direction",
        "unverifiable_comparisons",
    ]
    return f"Model: `{result['model']}`, n = {result['n']} drafts per condition.\n\n" + md_table(
        result["summary"], cols
    )


# --- a scripted stand-in for dry runs and tests ----------------------------------------------


def fake_drafter(store: ValueStore, seed: int = 0, error_rate: float = 0.3) -> FakeLLM:
    """A FakeLLM that writes plausible drafts, with deliberate mistakes at ``error_rate``."""
    rng = random.Random(seed)
    values = [
        v
        for v in store.values.values()
        if isinstance(v.value, int | float) and not isinstance(v.value, bool)
    ]
    by_unit: dict[str | None, list[Value]] = {}
    for v in values:
        by_unit.setdefault(v.unit, []).append(v)
    groups = [g for g in by_unit.values() if len(g) >= 2]

    def sentence_pair() -> tuple[Value, Value]:
        a, b = rng.sample(rng.choice(groups), 2)
        return (a, b) if float(a.value) < float(b.value) else (b, a)

    def respond(system: str, user: str) -> str:
        placeholders = "HARD RULE" in system
        lines = ["## Results", ""]
        for v in rng.sample(values, min(4, len(values))):
            if placeholders:
                if rng.random() < error_rate:
                    lines.append(
                        f"The {v.description.lower()} was {format_with_unit(v)}."
                    )  # forbidden digits
                else:
                    lines.append(f"The {v.description.lower()} was {{{{val:{v.id}}}}}.")
            else:
                if rng.random() < error_rate:
                    wrong = perturbed(v, 1.3, 1.0)
                    lines.append(f"The {v.description.lower()} was {format_with_unit(wrong)}.")
                else:
                    lines.append(f"The {v.description.lower()} was {format_with_unit(v)}.")
        lo, hi = sentence_pair()
        flip = rng.random() < error_rate
        a, b = (hi, lo) if flip else (lo, hi)
        if placeholders:
            lines.append(
                f"The {a.description.lower()} ({{{{val:{a.id}}}}}) was lower than the "
                f"{b.description.lower()} ({{{{val:{b.id}}}}})."
            )
        else:
            lines.append(
                f"The {a.description.lower()} ({format_with_unit(a)}) was lower than the "
                f"{b.description.lower()} ({format_with_unit(b)})."
            )
        return "\n".join(lines) + "\n"

    return FakeLLM(responder=respond, model="fake-drafter")
