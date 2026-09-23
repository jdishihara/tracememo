"""Find value/figure/table references in templates and mask everything that is not prose.

Supported syntaxes:

- Markdown/Jinja templates: ``{{ val("id") }}``, ``{{ val("id", with_unit=True) }}``,
  ``{{ val_raw("id") }}``, ``{{ unit("id") }}``, ``{{ fig("id") }}``, ``{{ tab("id") }}``
- Draft fragments: ``{{val:id}}``, ``{{fig:id}}``, ``{{tab:id}}``
- LaTeX: ``\\val{id}``, ``\\valu{id}``, ``\\fig{id}``, ``\\tab{id}``

Masking is length-preserving, so character offsets in the masked text map to the same
lines as in the original file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RefKind = Literal["val", "fig", "tab"]
Syntax = Literal["markdown", "latex"]

ID = r"[a-z0-9_]+(?:\.[a-z0-9_]+)+"

# Jinja calls: {{ val("id") }}, {{ val('id', with_unit=True) }}, {{ fig("id") }} ...
_JINJA_CALL = re.compile(
    r"\{\{-?\s*(?P<fn>val_raw|val|unit|fig|tab)\(\s*(?P<q>['\"])(?P<id>" + ID + r")(?P=q)"
    r"[^}]*?\}\}"
)
# Draft placeholders: {{val:id}} {{fig:id}} {{tab:id}}
_PLACEHOLDER = re.compile(r"\{\{\s*(?P<fn>val|fig|tab):(?P<id>" + ID + r")\s*\}\}")
# LaTeX: \val{id} \valu{id} \fig{id} \tab{id}
_LATEX_REF = re.compile(r"\\(?P<fn>valu|val|fig|tab)\{(?P<id>" + ID + r")\}")

_FN_TO_KIND: dict[str, RefKind] = {
    "val": "val",
    "val_raw": "val",
    "unit": "val",
    "valu": "val",
    "fig": "fig",
    "tab": "tab",
}


@dataclass(frozen=True)
class Ref:
    """A reference to a value, figure or table found in a template."""

    kind: RefKind
    id: str
    start: int
    end: int
    line: int


def detect_syntax(filename: str) -> Syntax:
    """Guess the template syntax from the file name."""
    return "latex" if filename.endswith(".tex") else "markdown"


def line_of(text: str, offset: int) -> int:
    """1-based line number of a character offset."""
    return text.count("\n", 0, offset) + 1


def find_refs(text: str, syntax: Syntax) -> list[Ref]:
    """All references in ``text`` in document order."""
    patterns = [_JINJA_CALL, _PLACEHOLDER] if syntax == "markdown" else [_LATEX_REF]
    refs: list[Ref] = []
    for pat in patterns:
        for m in pat.finditer(text):
            refs.append(
                Ref(
                    _FN_TO_KIND[m.group("fn")],
                    m.group("id"),
                    m.start(),
                    m.end(),
                    line_of(text, m.start()),
                )
            )
    return sorted(refs, key=lambda r: r.start)


def _blank(text: str, start: int, end: int, keep_newlines: bool = True) -> str:
    seg = text[start:end]
    repl = "".join("\n" if (c == "\n" and keep_newlines) else " " for c in seg)
    return text[:start] + repl + text[end:]


def _blank_all(text: str, pattern: re.Pattern[str]) -> str:
    out = text
    for m in reversed(list(pattern.finditer(text))):
        out = _blank(out, m.start(), m.end())
    return out


# --- Markdown / Jinja ------------------------------------------------------------------

_MD_NON_PROSE = [
    re.compile(r"\{#.*?#\}", re.S),  # Jinja comments
    re.compile(r"\{%.*?%\}", re.S),  # Jinja statements
    re.compile(r"\{\{.*?\}\}", re.S),  # any remaining Jinja expression
    re.compile(r"```.*?```", re.S),  # fenced code
    re.compile(r"`[^`\n]*`"),  # inline code
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),  # images
    re.compile(r"\]\([^)]*\)"),  # link targets
    re.compile(r"<[^>\n]+>"),  # inline HTML tags
    re.compile(r"^#+ .*$", re.M),  # headings (section numbering lives here)
    re.compile(r"^\s*\d+\.\s", re.M),  # ordered-list markers
]

# --- LaTeX -----------------------------------------------------------------------------

_TEX_COMMENT = re.compile(r"(?<!\\)%.*$", re.M)
_TEX_NON_PROSE = [
    re.compile(
        r"\\(?:ref|eqref|pageref|label|cite[a-z]*|url|href|input|include|includegraphics)"
        r"\*?(?:\[[^\]]*\])?\{[^}]*\}"
    ),
    re.compile(r"\\begin\{[^}]*\}(?:\[[^\]]*\])?"),
    re.compile(r"\\end\{[^}]*\}"),
    re.compile(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?"),  # any command name with optional args
    re.compile(r"\$[^$]*\$"),  # inline math
    re.compile(r"\\\[.*?\\\]", re.S),  # display math
]


def prose_mask(text: str, syntax: Syntax) -> tuple[str, list[Ref]]:
    """Return ``(masked_text, refs)``.

    In the masked text every reference is replaced by a token ``@R<n>@`` (index into
    ``refs``) padded to the reference's original length, and all non-prose constructs
    (markup, code, commands, comments, the LaTeX preamble) are blanked with spaces. Line
    structure is preserved.
    """
    refs = find_refs(text, syntax)
    masked = text
    for n, r in reversed(list(enumerate(refs))):
        token = f"@R{n}@"
        width = r.end - r.start
        token = token[:width].ljust(width)
        masked = masked[: r.start] + token + masked[r.end :]

    if syntax == "latex":
        begin = masked.find(r"\begin{document}")
        if begin >= 0:
            masked = _blank(masked, 0, begin)
        end = masked.find(r"\end{document}")
        if end >= 0:
            masked = _blank(masked, end, len(masked))
        masked = _blank_all(masked, _TEX_COMMENT)
        for pat in _TEX_NON_PROSE:
            masked = _blank_all(masked, pat)
    else:
        for pat in _MD_NON_PROSE:
            masked = _blank_all(masked, pat)
    return masked, refs
