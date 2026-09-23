"""Checker recall: inject known errors into clean drafts and measure what gets caught."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from tracememo.check.numbers import check_text, current_source_hashes
from tracememo.evaluation.common import (
    build_project,
    load_store,
    md_table,
    prepare_example,
    synth_drone,
    synth_rag,
)
from tracememo.evaluation.drafting import perturbed
from tracememo.report.render import format_with_unit
from tracememo.store.models import Manifest, Value
from tracememo.store.store import ValueStore

ERROR_TYPES = [
    "swap_number",
    "invent_statistic",
    "flip_comparison",
    "unknown_reference",
    "stale_reference",
]
EXPECTED_CHECK = {
    "swap_number": "raw_number",
    "invent_statistic": "raw_number",
    "flip_comparison": "comparison",
    "unknown_reference": "unknown_reference",
    "stale_reference": "stale",
}
FLIP = {
    "lower than": "higher than",
    "higher than": "lower than",
    "below": "above",
    "above": "below",
    "under": "over",
    "exceeds": "is under",
    "less than": "more than",
    "more than": "less than",
}
PLACEHOLDER = re.compile(r"\{\{val:([a-z0-9_.]+)\}\}")


def _numeric_values(store: ValueStore) -> list[Value]:
    """Numeric values whose descriptions carry no digits (so clean docs contain none)."""
    return [
        v
        for v in store.values.values()
        if isinstance(v.value, int | float)
        and not isinstance(v.value, bool)
        and not re.search(r"\d", v.description)
    ]


def make_clean_doc(store: ValueStore, rng: random.Random, n_sentences: int = 6) -> str:
    """A correct draft: value statements, at least one true comparison, a figure/table ref."""
    values = _numeric_values(store)
    by_unit: dict[str | None, list[Value]] = {}
    for v in values:
        by_unit.setdefault(v.unit, []).append(v)
    groups = [g for g in by_unit.values() if len(g) >= 2]

    def comparison() -> str:
        while True:
            a, b = rng.sample(rng.choice(groups), 2)
            if float(a.value) != float(b.value):
                break
        lo, hi = (a, b) if float(a.value) < float(b.value) else (b, a)
        if rng.random() < 0.5:
            return (
                f"The {lo.description.lower()} ({{{{val:{lo.id}}}}}) was lower than the "
                f"{hi.description.lower()} ({{{{val:{hi.id}}}}})."
            )
        return (
            f"The {hi.description.lower()} of {{{{val:{hi.id}}}}} exceeds the "
            f"{lo.description.lower()} of {{{{val:{lo.id}}}}}."
        )

    def statement() -> str:
        v = rng.choice(values)
        return f"The {v.description.lower()} was {{{{val:{v.id}}}}}."

    lines = ["## Results", "", comparison()]
    for _ in range(n_sentences - 1):
        lines.append(comparison() if rng.random() < 0.33 else statement())
    if store.figures:
        lines.append(f"See {{{{fig:{rng.choice(sorted(store.figures))}}}}} for the time history.")
    if store.tables:
        lines.append(f"Events are listed in {{{{tab:{rng.choice(sorted(store.tables))}}}}}.")
    return "\n".join(lines) + "\n"


def inject(
    doc: str, error_type: str, store: ValueStore, rng: random.Random
) -> tuple[str, str | None]:
    """Return ``(mutated_doc, stale_analysis_id)``; the id is set only for stale_reference."""
    lines = doc.split("\n")
    ph_lines = [i for i, ln in enumerate(lines) if PLACEHOLDER.search(ln)]
    if error_type == "swap_number":
        i = rng.choice(ph_lines)
        m = PLACEHOLDER.search(lines[i])
        assert m
        v = store.get_value(m.group(1))
        factor = 1.0 + rng.choice([-1, 1]) * rng.uniform(0.1, 0.5)
        fake = perturbed(v, factor)
        lines[i] = lines[i][: m.start()] + format_with_unit(fake) + lines[i][m.end() :]
        return "\n".join(lines), None
    if error_type == "invent_statistic":
        stat = rng.choice(["standard deviation", "interquartile range", "peak-to-peak variation"])
        lines.append(f"The {stat} of the error was {rng.uniform(1, 40):.1f} cm.")
        return "\n".join(lines), None
    if error_type == "flip_comparison":
        cmp_lines = [i for i in ph_lines if len(PLACEHOLDER.findall(lines[i])) == 2]
        i = rng.choice(cmp_lines)
        first, second = list(PLACEHOLDER.finditer(lines[i]))
        between = lines[i][first.end() : second.start()]
        for phrase in sorted(FLIP, key=len, reverse=True):
            if phrase in between:
                between = between.replace(phrase, FLIP[phrase], 1)
                break
        lines[i] = lines[i][: first.end()] + between + lines[i][second.start() :]
        return "\n".join(lines), None
    if error_type == "unknown_reference":
        i = rng.choice(ph_lines)
        m = PLACEHOLDER.search(lines[i])
        assert m
        bogus = m.group(1).rsplit(".", 1)[0] + ".nonexistent_stat"
        lines[i] = lines[i][: m.start()] + "{{val:" + bogus + "}}" + lines[i][m.end() :]
        return "\n".join(lines), None
    if error_type == "stale_reference":
        i = rng.choice(ph_lines)
        m = PLACEHOLDER.search(lines[i])
        assert m
        prov = store.get_value(m.group(1)).provenance
        assert prov is not None
        return doc, prov.analysis_id
    raise ValueError(error_type)


def evaluate(store: ValueStore, n_docs: int, seed: int) -> dict[str, Any]:
    """Recall per error type and false-positive rate on clean docs for one project."""
    rng = random.Random(seed)
    manifest: Manifest = store.to_manifest()
    hashes = current_source_hashes()
    clean_docs = [make_clean_doc(store, rng) for _ in range(n_docs)]
    fp_docs = 0
    fp_findings = 0
    for doc in clean_docs:
        findings = [
            f
            for f in check_text(doc, "clean.md", manifest, "markdown", None, hashes)
            if f.severity == "error"
        ]
        fp_findings += len(findings)
        fp_docs += int(bool(findings))
    per_type = []
    for et in ERROR_TYPES:
        caught = 0
        for doc in clean_docs:
            mutated, stale_id = inject(doc, et, store, rng)
            h = dict(hashes)
            if stale_id:
                h[stale_id] = "0" * 64
            findings = check_text(mutated, "mut.md", manifest, "markdown", None, h)
            if any(f.check == EXPECTED_CHECK[et] and f.severity == "error" for f in findings):
                caught += 1
        per_type.append(
            {
                "error_type": et,
                "expected_check": EXPECTED_CHECK[et],
                "injected": n_docs,
                "caught": caught,
                "recall": caught / n_docs,
            }
        )
    return {
        "per_type": per_type,
        "clean_docs": n_docs,
        "false_positive_docs": fp_docs,
        "false_positive_findings": fp_findings,
        "false_positive_rate": fp_docs / n_docs,
    }


def run(work: Path, n_docs: int = 50, seed: int = 0) -> dict[str, Any]:
    """Run the recall experiment on the drone and RAG example projects."""
    work.mkdir(parents=True, exist_ok=True)
    drone_cfg = build_project(
        prepare_example("drone_memo", work, synth_drone(work / "data" / "drone", seed))
    )
    rag_cfg = build_project(
        prepare_example("rag_memo", work, synth_rag(work / "data" / "rag", seed))
    )
    results = {
        "drone_memo": evaluate(load_store(drone_cfg), n_docs, seed),
        "rag_memo": evaluate(load_store(rag_cfg), n_docs, seed + 1),
    }
    combined = []
    for et in ERROR_TYPES:
        inj = sum(r["per_type"][ERROR_TYPES.index(et)]["injected"] for r in results.values())
        caught = sum(r["per_type"][ERROR_TYPES.index(et)]["caught"] for r in results.values())
        combined.append(
            {
                "error_type": et,
                "expected_check": EXPECTED_CHECK[et],
                "injected": inj,
                "caught": caught,
                "recall": caught / inj,
            }
        )
    fp_docs = sum(r["false_positive_docs"] for r in results.values())
    n_clean = sum(r["clean_docs"] for r in results.values())
    return {
        "seed": seed,
        "n_docs_per_project": n_docs,
        "projects": results,
        "combined": combined,
        "clean_docs": n_clean,
        "false_positive_docs": fp_docs,
        "false_positive_rate": fp_docs / n_clean,
    }


def to_markdown(result: dict[str, Any]) -> str:
    """Recall table and the false-positive line."""
    text = md_table(
        result["combined"], ["error_type", "expected_check", "injected", "caught", "recall"]
    )
    text += (
        f"\n\nFalse positives on clean drafts: {result['false_positive_docs']} of "
        f"{result['clean_docs']} documents ({100 * result['false_positive_rate']:.1f}%)."
    )
    return text
