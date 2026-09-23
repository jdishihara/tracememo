"""Deterministic grounding checks.

1. raw_number: digits in prose that are not inside a value reference (with an allowlist).
2. unknown_reference: referenced ids missing from the manifest.
3. stale: referenced values whose raw input files or analysis source changed since the run.
4. comparison: simple two-value comparative sentences whose direction contradicts the values.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from tracememo.check.extract import Ref, Syntax, detect_syntax, line_of, prose_mask
from tracememo.check.models import CheckReport, Finding
from tracememo.hashing import sha256_file
from tracememo.store.models import Manifest, Value

# --- check 1: raw numbers ---------------------------------------------------------------

NUMBER = re.compile(r"(?<![A-Za-z0-9_.@])\d+(?:[.,]\d+)*(?![A-Za-z0-9_@])")

# Built-in allowlist, applied to the text surrounding a number (the whole line).
DEFAULT_ALLOW = [
    r"\b(?:19|20)\d\d\b",  # years
    r"\b(?:Section|Sec\.|Chapter|Appendix|Figure|Fig\.|Table|Tab\.|Equation|Eq\.|Step|Part)"
    r"\s*~?\s*\d+(?:\.\d+)*",  # numbered cross references
    r"\[\d+(?:\s*[,–-]\s*\d+)*\]",  # numeric citations [3], [1, 2]
    r"\b\d+(?:st|nd|rd|th)\b",  # ordinals: 95th percentile
]


def _allowed_spans(line: str, allow: list[re.Pattern[str]]) -> list[tuple[int, int]]:
    spans = []
    for pat in allow:
        spans += [(m.start(), m.end()) for m in pat.finditer(line)]
    return spans


def check_raw_numbers(
    masked: str, file: str, allow_patterns: list[str] | None = None
) -> list[Finding]:
    """Flag digit sequences in prose that are not references and not allowlisted."""
    allow = [re.compile(p) for p in DEFAULT_ALLOW + list(allow_patterns or [])]
    findings = []
    for ln, line in enumerate(masked.split("\n"), start=1):
        allowed = _allowed_spans(line, allow)
        for m in NUMBER.finditer(line):
            if any(a <= m.start() and m.end() <= b for a, b in allowed):
                continue
            findings.append(
                Finding(
                    check="raw_number",
                    severity="error",
                    file=file,
                    line=ln,
                    text=line.strip(),
                    message=f"raw number {m.group()!r} in prose; use a value reference",
                )
            )
    return findings


# --- check 2: unknown references --------------------------------------------------------


def check_unknown_references(refs: list[Ref], manifest: Manifest, file: str) -> list[Finding]:
    """Flag references whose id (of the right kind) is not in the manifest."""
    known = {"val": manifest.values, "fig": manifest.figures, "tab": manifest.tables}
    findings = []
    for r in refs:
        if r.id not in known[r.kind]:
            findings.append(
                Finding(
                    check="unknown_reference",
                    severity="error",
                    file=file,
                    line=r.line,
                    text=r.id,
                    ids=[r.id],
                    message=f"unknown {r.kind} id {r.id!r} (not in manifest)",
                )
            )
    return findings


# --- check 3: staleness -----------------------------------------------------------------


def current_source_hashes() -> dict[str, str]:
    """Source hashes of the currently registered analyses."""
    from tracememo.analyses.registry import all_analyses, load_builtin

    load_builtin()
    return {aid: spec.source_sha256 for aid, spec in all_analyses().items()}


def check_stale(
    refs: list[Ref],
    manifest: Manifest,
    file: str,
    source_hashes: dict[str, str] | None = None,
) -> list[Finding]:
    """Flag referenced items whose raw inputs or analysis source changed since the run.

    Each (item, reason) is reported once per file, at the line of the first reference.
    """
    if source_hashes is None:
        source_hashes = current_source_hashes()
    file_cache: dict[str, str | None] = {}

    def file_hash(path: str) -> str | None:
        if path not in file_cache:
            p = Path(path)
            file_cache[path] = sha256_file(p) if p.exists() else None
        return file_cache[path]

    groups: dict[tuple[str, str], tuple[int, list[str]]] = {}
    seen: set[str] = set()
    for r in refs:
        if r.id in seen:
            continue
        seen.add(r.id)
        item = {"val": manifest.values, "fig": manifest.figures, "tab": manifest.tables}[
            r.kind
        ].get(r.id)
        if item is None or item.provenance is None:
            continue
        prov = item.provenance
        reasons = []
        for raw in prov.raw_inputs:
            h = file_hash(raw.path)
            if h is None:
                reasons.append(f"input file {raw.path} is missing")
            elif h != raw.sha256:
                reasons.append(
                    f"input file {Path(raw.path).name} changed ({raw.sha256[:8]} -> {h[:8]})"
                )
        current = source_hashes.get(prov.analysis_id)
        if current is None:
            reasons.append(f"analysis {prov.analysis_id} is no longer registered")
        elif current != prov.analysis_source_sha256:
            reasons.append(
                f"analysis {prov.analysis_id} source changed "
                f"({prov.analysis_source_sha256[:8]} -> {current[:8]})"
            )
        for reason in reasons:
            key = (prov.analysis_id, reason)
            line, ids = groups.get(key, (r.line, []))
            groups[key] = (line, ids + [r.id])
    findings = []
    for (analysis_id, reason), (line, ids) in groups.items():
        findings.append(
            Finding(
                check="stale",
                severity="error",
                file=file,
                line=line,
                text=", ".join(ids),
                ids=ids,
                message=(
                    f"{len(ids)} referenced item(s) from {analysis_id} are stale: {reason}; "
                    "rerun `tracememo build`"
                ),
            )
        )
    return findings


# --- check 4: comparisons ---------------------------------------------------------------

REF_TOKEN = re.compile(r"@R(\d+)@")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n\s*\n")

# phrase -> expected sign of (first - second): -1 means first < second.
LT_PHRASES = ["lower than", "less than", "smaller than", "shorter than", "below", "under"]
GT_PHRASES = [
    "higher than",
    "greater than",
    "larger than",
    "longer than",
    "more than",
    "above",
    "exceeds",
    "exceeded",
    "over",
]
DECREASE_VERBS = ["reduced", "decreased", "dropped", "fell", "declined", "shrank", "lowered"]
INCREASE_VERBS = ["increased", "rose", "grew", "climbed", "raised"]

_BETWEEN = re.compile(
    r"@R(?P<a>\d+)@(?P<mid>[^@]{0,80}?)\b(?P<phrase>"
    + "|".join(re.escape(p) for p in LT_PHRASES + GT_PHRASES)
    + r")\b[^@]{0,80}?@R(?P<b>\d+)@",
    re.I,
)
_FROM_TO = re.compile(
    r"\b(?P<verb>" + "|".join(DECREASE_VERBS + INCREASE_VERBS) + r")\b[^@]{0,60}?"
    r"\bfrom\s+[^@]{0,20}?@R(?P<a>\d+)@[^@]{0,30}?\bto\s+[^@]{0,20}?@R(?P<b>\d+)@",
    re.I,
)
NEGATION = re.compile(r"\b(?:not|no|never|neither)\b", re.I)


def _numeric(v: Value) -> float | None:
    return float(v.value) if isinstance(v.value, int | float) else None


def check_comparisons(masked: str, refs: list[Ref], manifest: Manifest, file: str) -> list[Finding]:
    """Verify the direction of simple comparative sentences that reference two values."""
    findings = []
    pos = 0
    for raw_sentence in SENTENCE_END.split(masked):
        start = masked.find(raw_sentence, pos)
        pos = start + len(raw_sentence)
        sentence = " ".join(raw_sentence.split())  # collapse token padding and newlines
        tokens = REF_TOKEN.findall(sentence)
        if len(tokens) != 2:
            continue
        m = _BETWEEN.search(sentence)
        if m:
            phrase = m.group("phrase").lower()
            expected = -1 if phrase in LT_PHRASES else 1
            a, b = int(m.group("a")), int(m.group("b"))
            context = m.group(0)
        else:
            m = _FROM_TO.search(sentence)
            if not m:
                continue
            verb = m.group("verb").lower()
            expected = 1 if verb in DECREASE_VERBS else -1
            a, b = int(m.group("a")), int(m.group("b"))
            context = m.group(0)
        if NEGATION.search(context):
            continue  # negated claims are left to the LLM claim checker
        ra, rb = refs[a], refs[b]
        if ra.kind != "val" or rb.kind != "val":
            continue
        va, vb = manifest.values.get(ra.id), manifest.values.get(rb.id)
        if va is None or vb is None:
            continue  # reported by unknown_reference
        xa, xb = _numeric(va), _numeric(vb)
        line = line_of(masked, start + len(raw_sentence) - len(raw_sentence.lstrip()))
        text = sentence
        if xa is None or xb is None:
            continue
        if va.unit != vb.unit:
            findings.append(
                Finding(
                    check="comparison",
                    severity="warning",
                    file=file,
                    line=line,
                    text=text,
                    ids=[ra.id, rb.id],
                    message=(
                        f"compares {ra.id} ({va.unit}) with {rb.id} ({vb.unit}): different units"
                    ),
                )
            )
            continue
        actual = (xa > xb) - (xa < xb)
        if actual != expected:
            word = "lower" if expected < 0 else "higher"
            findings.append(
                Finding(
                    check="comparison",
                    severity="error",
                    file=file,
                    line=line,
                    text=text,
                    ids=[ra.id, rb.id],
                    message=(
                        f"claims {ra.id} is {word} than {rb.id}, but values are "
                        f"{va.formatted()} vs {vb.formatted()}"
                    ),
                )
            )
    return findings


# --- driver -----------------------------------------------------------------------------


def check_text(
    text: str,
    file: str,
    manifest: Manifest,
    syntax: Syntax | None = None,
    allow_patterns: list[str] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> list[Finding]:
    """Run all deterministic checks on one template's text."""
    syntax = syntax or detect_syntax(file)
    masked, refs = prose_mask(text, syntax)
    findings = check_raw_numbers(masked, file, allow_patterns)
    findings += check_unknown_references(refs, manifest, file)
    findings += check_stale(refs, manifest, file, source_hashes)
    findings += check_comparisons(masked, refs, manifest, file)
    return findings


def run_checks(
    files: list[Path],
    manifest: Manifest,
    allow_patterns: list[str] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> CheckReport:
    """Run all deterministic checks over several template files."""
    if source_hashes is None:
        source_hashes = current_source_hashes()
    findings: list[Finding] = []
    for f in files:
        text = Path(f).read_text(encoding="utf-8")
        findings += check_text(text, str(f), manifest, None, allow_patterns, source_hashes)
    findings.sort(key=lambda x: (x.file, x.line, x.check))
    return CheckReport(created=datetime.now(UTC), files=[str(f) for f in files], findings=findings)
