"""Render report templates with live values from the value store."""

from __future__ import annotations

import os
from pathlib import Path

import jinja2
import pandas as pd

from tracememo.report.tables import df_to_markdown
from tracememo.store.models import Provenance, Value
from tracememo.store.store import ValueStore

TEMPLATES_DIR = Path(__file__).parent / "templates"


def anchor_for(id_: str) -> str:
    """Markdown/HTML anchor name for a value id."""
    return "prov-" + id_.replace(".", "-")


def format_with_unit(v: Value) -> str:
    """Formatted value followed by its unit (no space before a percent sign)."""
    text = v.formatted()
    if not v.unit:
        return text
    return f"{text}%" if v.unit == "%" else f"{text} {v.unit}"


def provenance_summary(p: Provenance) -> str:
    """One-line provenance summary used as a link tooltip."""
    commit = (p.git_commit or "no-git")[:8] + ("*" if p.git_dirty else "")
    return f"{p.analysis_id} @ {commit}, {p.timestamp:%Y-%m-%d %H:%M} UTC"


class MarkdownRenderer:
    """Jinja2 renderer exposing ``val``, ``fig``, ``tab`` and friends to templates."""

    def __init__(
        self,
        store: ValueStore,
        out_dir: Path,
        provenance_links: bool = False,
        extra_search_paths: list[Path] | None = None,
    ) -> None:
        self.store = store
        self.out_dir = Path(out_dir)
        self.provenance_links = provenance_links
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(
                [str(p) for p in (extra_search_paths or [])] + [str(TEMPLATES_DIR)]
            ),
            undefined=jinja2.StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.env.globals.update(
            val=self.val,
            val_raw=self.val_raw,
            unit=self.unit,
            fig=self.fig,
            tab=self.tab,
            provenance_appendix=self.provenance_appendix,
        )

    # -- template functions ---------------------------------------------------

    def val(self, id_: str, with_unit: bool = False) -> str:
        """Formatted value, optionally followed by its unit."""
        v = self.store.get_value(id_)
        text = format_with_unit(v) if with_unit else v.formatted()
        if self.provenance_links and v.provenance is not None:
            text = f'[{text}](#{anchor_for(id_)} "{provenance_summary(v.provenance)}")'
        return text

    def val_raw(self, id_: str) -> float | int | str:
        """The unformatted Python value (for arithmetic or conditionals in templates)."""
        return self.store.get_value(id_).value

    def unit(self, id_: str) -> str:
        """The value's unit, or an empty string."""
        return self.store.get_value(id_).unit or ""

    def fig(self, id_: str) -> str:
        """Markdown image plus caption for a figure id."""
        f = self.store.get_figure(id_)
        if f.png_path is None:
            raise ValueError(f"figure {id_} has no PNG file")
        rel = os.path.relpath(f.png_path, self.out_dir)
        return f"![{f.caption}]({rel})\n\n*{f.caption}*"

    def tab(self, id_: str) -> str:
        """Markdown table plus caption for a table id."""
        t = self.store.get_table(id_)
        if t.markdown_path and Path(t.markdown_path).exists():
            body = Path(t.markdown_path).read_text(encoding="utf-8").rstrip("\n")
        elif t.parquet_path:
            body = df_to_markdown(pd.read_parquet(t.parquet_path), t.float_fmt)
        else:
            raise ValueError(f"table {id_} has no rendered file")
        return f"*{t.caption}*\n\n{body}"

    def provenance_appendix(self) -> str:
        """A Markdown section listing the provenance of every value, with anchors."""
        lines = ["## Provenance", ""]
        for id_ in sorted(self.store.values):
            v = self.store.values[id_]
            if v.provenance is None:
                continue
            p = v.provenance
            files = ", ".join(f"`{Path(r.path).name}` ({r.sha256[:8]})" for r in p.raw_inputs)
            lines.append(
                f'- <a id="{anchor_for(id_)}"></a>`{id_}` = {format_with_unit(v)}: '
                f"{p.analysis_id} "
                f"(source {p.analysis_source_sha256[:8]}, commit {(p.git_commit or 'none')[:8]}"
                f"{', dirty' if p.git_dirty else ''}), inputs {files}"
            )
        return "\n".join(lines) + "\n"

    # -- rendering ------------------------------------------------------------

    def render(self, template_path: Path, out_path: Path) -> Path:
        """Render ``template_path`` to ``out_path`` and return it."""
        template_path = Path(template_path)
        loader = jinja2.ChoiceLoader(
            [jinja2.FileSystemLoader(str(template_path.parent)), self.env.loader]  # type: ignore[list-item]
        )
        env = self.env.overlay(loader=loader)
        template = env.get_template(template_path.name)
        text = template.render()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        return out_path
