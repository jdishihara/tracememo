"""Render dataframes as Markdown and LaTeX tables."""

from __future__ import annotations

import math

import pandas as pd

LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    return "".join(LATEX_SPECIALS.get(ch, ch) for ch in text)


def format_cell(v: object, float_fmt: str) -> str:
    """Format one table cell as text."""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return format(v, float_fmt)
    if v is None:
        return ""
    return str(v)


def df_to_markdown(df: pd.DataFrame, float_fmt: str = ".4g") -> str:
    """Render a dataframe as a GitHub-flavoured Markdown table."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        cells = [format_cell(v, float_fmt).replace("|", "\\|") for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def df_to_latex(df: pd.DataFrame, float_fmt: str = ".4g") -> str:
    """Render a dataframe as a booktabs ``tabular`` (no float/caption; ``\\tab`` adds those)."""
    cols = [str(c) for c in df.columns]
    numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]
    spec = "".join("r" if n else "l" for n in numeric)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    lines.append(" & ".join(latex_escape(c) for c in cols) + " \\\\")
    lines.append("\\midrule")
    for row in df.itertuples(index=False):
        lines.append(" & ".join(latex_escape(format_cell(v, float_fmt)) for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)
