"""LLM-assisted claim checks: verdicts on quantitative sentences the deterministic checks
could not verify. These are warnings only and never fail a build."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tracememo.check.extract import Ref, Syntax, detect_syntax, line_of, prose_mask
from tracememo.check.models import Finding
from tracememo.check.numbers import _BETWEEN, _FROM_TO, NEGATION, REF_TOKEN, SENTENCE_END
from tracememo.llm import LLMClient
from tracememo.report.render import format_with_unit
from tracememo.store.models import Manifest

PROMPTS_DIR = Path(__file__).parent.parent / "draft" / "prompts"

CLAIM_WORDS = re.compile(
    r"\b(lower|higher|less|more|fewer|greater|smaller|larger|faster|slower|better|worse|best|"
    r"worst|increase\w*|decrease\w*|reduc\w*|improv\w*|degrad\w*|exceed\w*|below|above|under|"
    r"over|within|half|halved|double\w*|twice|triple\w*|majority|most|negligible|significant\w*|"
    r"roughly|approximately|about|nearly|almost|only|than|percent|dominant|dominat\w*|"
    r"similar|comparable|unchanged|consistent)\b",
    re.I,
)

RESULTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "unsupported", "cant_tell"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["index", "verdict", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Claim:
    """A sentence to verify, with its cited value refs."""

    line: int
    sentence: str
    refs: tuple[Ref, ...]


def deterministically_covered(sentence: str) -> bool:
    """True if check 4 already verified this sentence (two refs, known phrase, no negation)."""
    if len(REF_TOKEN.findall(sentence)) != 2:
        return False
    m = _BETWEEN.search(sentence) or _FROM_TO.search(sentence)
    return bool(m) and not NEGATION.search(m.group(0))


def extract_claims(masked: str, refs: list[Ref]) -> list[Claim]:
    """Sentences with at least one value reference and a claim word, not already verified."""
    claims = []
    pos = 0
    for raw in SENTENCE_END.split(masked):
        start = masked.find(raw, pos)
        pos = start + len(raw)
        sentence = " ".join(raw.split())
        tokens = [int(t) for t in REF_TOKEN.findall(sentence)]
        if not tokens or not CLAIM_WORDS.search(sentence):
            continue
        if deterministically_covered(sentence):
            continue
        cited = tuple(refs[t] for t in tokens if refs[t].kind == "val")
        if not cited:
            continue
        line = line_of(masked, start + len(raw) - len(raw.lstrip()))
        claims.append(Claim(line=line, sentence=sentence, refs=cited))
    return claims


def render_claim(claim: Claim, refs: list[Ref], manifest: Manifest) -> str:
    """Expand ``@R<n>@`` tokens to ``[id = value unit]`` and list the cited values."""

    def expand(m: re.Match[str]) -> str:
        r = refs[int(m.group(1))]
        v = manifest.values.get(r.id)
        if r.kind != "val" or v is None:
            return f"[{r.kind} {r.id}]"
        return f"[{r.id} = {format_with_unit(v)}]"

    text = REF_TOKEN.sub(expand, claim.sentence)
    cited = []
    for r in claim.refs:
        v = manifest.values.get(r.id)
        if v is not None:
            cited.append(
                f"  {r.id} = {v.formatted()}{(' ' + v.unit) if v.unit else ''}: {v.description}"
            )
    return text + "\n" + "\n".join(cited)


def check_claims_text(
    text: str, file: str, manifest: Manifest, client: LLMClient, syntax: Syntax | None = None
) -> list[Finding]:
    """Ask the model about each unverified quantitative sentence in one template."""
    syntax = syntax or detect_syntax(file)
    masked, refs = prose_mask(text, syntax)
    claims = extract_claims(masked, refs)
    if not claims:
        return []
    system = (PROMPTS_DIR / "claims_system.txt").read_text(encoding="utf-8")
    listing = "\n\n".join(
        f"Claim {i}:\n{render_claim(c, refs, manifest)}" for i, c in enumerate(claims)
    )
    user = (PROMPTS_DIR / "claims_user.txt").read_text(encoding="utf-8").format(claims=listing)
    data = client.complete_json(system, user, RESULTS_SCHEMA)
    findings = []
    for item in data.get("results", []):
        try:
            claim = claims[int(item["index"])]
        except (KeyError, ValueError, IndexError):
            continue
        verdict = str(item.get("verdict", "cant_tell"))
        if verdict == "supported":
            continue
        findings.append(
            Finding(
                check="claim",
                severity="warning",
                file=file,
                line=claim.line,
                text=REF_TOKEN.sub(lambda m: refs[int(m.group(1))].id, claim.sentence),
                ids=[r.id for r in claim.refs],
                message=f"{verdict}: {item.get('reason', '')}".strip(),
            )
        )
    return findings


def check_claims(files: list[Path], manifest: Manifest, client: LLMClient) -> list[Finding]:
    """Run the LLM claim check over several files."""
    findings: list[Finding] = []
    for f in files:
        findings += check_claims_text(Path(f).read_text(encoding="utf-8"), str(f), manifest, client)
    return findings
