"""Standalone HTML viewer: the rendered report where clicking a number shows its provenance."""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

import pandas as pd

from tracememo.report.render import MarkdownRenderer, format_with_unit
from tracememo.store.store import ValueStore

_INLINE = [
    (re.compile(r"`([^`]+)`"), lambda m: f"<code>{html.escape(m.group(1))}</code>"),
    (re.compile(r"\*\*(.+?)\*\*"), lambda m: f"<strong>{m.group(1)}</strong>"),
    (re.compile(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])"), lambda m: f"<em>{m.group(1)}</em>"),
    (
        re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)"),
        lambda m: f'<img alt="{m.group(1)}" src="{m.group(2)}">',
    ),
    (
        re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"([^\"]*)\")?\)"),
        lambda m: f'<a href="{m.group(2)}" title="{m.group(3) or ""}">{m.group(1)}</a>',
    ),
]


def _inline(text: str) -> str:
    for pat, fn in _INLINE:
        text = pat.sub(fn, text)
    return text


def markdown_to_html(md: str) -> str:
    """A small Markdown-to-HTML converter covering what the report templates produce.

    Supports headings, paragraphs, bullet/numbered lists, pipe tables, fenced code, images,
    links, inline code, bold/italic, and passes raw HTML lines through.
    """
    out: list[str] = []
    lines = md.split("\n")
    i = 0
    para: list[str] = []

    def flush() -> None:
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            j = i + 1
            code = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code.append(lines[j])
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush()
            out.append(f"<h{len(m.group(1))}>{_inline(m.group(2))}</h{len(m.group(1))}>")
            i += 1
            continue
        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip())
        ):
            flush()
            header = [c.strip() for c in stripped.strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            out.append(
                "<table><thead><tr>"
                + "".join(f"<th>{_inline(h)}</th>" for h in header)
                + "</tr></thead><tbody>"
                + "".join(
                    "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows
                )
                + "</tbody></table>"
            )
            i = j
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            flush()
            ordered = m.group(2)[0].isdigit()
            tag = "ol" if ordered else "ul"
            items = []
            while i < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if not mm:
                    break
                items.append(f"<li>{_inline(mm.group(3))}</li>")
                i += 1
            out.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
            continue
        if stripped.startswith("<") and stripped.endswith(">"):
            flush()
            out.append(stripped)
            i += 1
            continue
        if not stripped:
            flush()
            i += 1
            continue
        para.append(stripped)
        i += 1
    flush()
    return "\n".join(out)


class HtmlRenderer(MarkdownRenderer):
    """Markdown renderer whose values, figures and tables carry ids for the viewer."""

    def val(self, id_: str, with_unit: bool = False) -> str:
        v = self.store.get_value(id_)
        text = format_with_unit(v) if with_unit else v.formatted()
        return f'<span class="tm-val" data-id="{id_}" tabindex="0">{html.escape(text)}</span>'

    def fig(self, id_: str) -> str:
        f = self.store.get_figure(id_)
        if f.png_path is None:
            raise ValueError(f"figure {id_} has no PNG file")
        data = base64.b64encode(Path(f.png_path).read_bytes()).decode("ascii")
        return (
            f'<figure class="tm-fig" data-id="{id_}"><img src="data:image/png;base64,{data}" '
            f'alt="{html.escape(f.caption)}"><figcaption>{html.escape(f.caption)}</figcaption></figure>'
        )

    def tab(self, id_: str) -> str:
        t = self.store.get_table(id_)
        if t.parquet_path is None:
            raise ValueError(f"table {id_} has no Parquet file")
        df = pd.read_parquet(t.parquet_path)
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(_fmt(v, t.float_fmt))}</td>" for v in row) + "</tr>"
            for row in df.itertuples(index=False)
        )
        return (
            f'<figure class="tm-tab" data-id="{id_}"><figcaption>{html.escape(t.caption)}</figcaption>'
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></figure>"
        )

    def provenance_appendix(self) -> str:
        return ""  # the viewer shows provenance interactively


def _fmt(v: object, float_fmt: str) -> str:
    return format(v, float_fmt) if isinstance(v, float) else str(v)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --accent:#2a78d6; --bg:#fcfcfb; --panel:#f4f3ef; }}
@media (prefers-color-scheme: dark) {{ :root {{ --ink:#f2f2f0; --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a; --accent:#3987e5; --bg:#1a1a19; --panel:#232322; }} }}
body {{ margin:0; font-family: system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); line-height:1.5; }}
main {{ max-width:760px; margin:0 auto; padding:24px 16px 80px; }}
aside {{ position:fixed; right:0; top:0; bottom:0; width:min(380px, 90vw); background:var(--panel); border-left:1px solid var(--line); padding:16px; overflow:auto; transform:translateX(100%); transition:transform .15s; font-size:14px; }}
aside.open {{ transform:none; }}
aside h2 {{ font-size:15px; margin:0 0 8px; }}
aside dl {{ display:grid; grid-template-columns: 110px 1fr; gap:4px 8px; margin:0; }}
aside dt {{ color:var(--muted); }} aside dd {{ margin:0; word-break:break-all; }}
aside button {{ float:right; border:0; background:none; color:var(--ink2); font-size:18px; cursor:pointer; }}
.tm-val {{ color:var(--accent); border-bottom:1px dotted var(--accent); cursor:pointer; }}
.tm-val.active {{ background:color-mix(in srgb, var(--accent) 18%, transparent); }}
figure {{ margin:24px 0; }} figure img {{ max-width:100%; height:auto; }} figcaption {{ color:var(--ink2); font-size:14px; margin-top:6px; }}
table {{ border-collapse:collapse; font-size:14px; }} th,td {{ border-bottom:1px solid var(--line); padding:4px 10px; text-align:left; }} th {{ color:var(--ink2); font-weight:600; }}
code {{ font-size:13px; }} .hint {{ color:var(--muted); font-size:13px; }}
</style></head><body>
<main>
<p class="hint">Click any highlighted number to see where it came from. Generated by tracememo from <code>manifest.json</code>.</p>
{body}
</main>
<aside id="panel"><button id="close" aria-label="close">&times;</button><div id="panel-body"></div></aside>
<script id="manifest" type="application/json">{manifest}</script>
<script>
const manifest = JSON.parse(document.getElementById('manifest').textContent);
const panel = document.getElementById('panel'), body = document.getElementById('panel-body');
function esc(s) {{ return String(s).replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c])); }}
function show(id) {{
  const v = manifest.values[id] || manifest.figures[id] || manifest.tables[id];
  if (!v) return;
  const p = v.provenance || {{}};
  const rows = [
    ['id', id], ['value', v.value !== undefined ? esc(v.value) + (v.unit ? ' ' + esc(v.unit) : '') : (v.caption || '')],
    ['description', v.description || ''], ['analysis', p.analysis_id || ''],
    ['source hash', (p.analysis_source_sha256 || '').slice(0, 12)],
    ['git commit', (p.git_commit || 'none').slice(0, 12) + (p.git_dirty ? ' (dirty)' : '')],
    ['params', esc(JSON.stringify(p.params || {{}}))],
    ['input tables', (p.input_tables || []).map(t => t.id + ' (' + t.sha256.slice(0, 8) + ')').join(', ')],
    ['raw inputs', (p.raw_inputs || []).map(r => r.path.split('/').pop() + ' (' + r.sha256.slice(0, 8) + ')').join(', ')],
    ['computed', p.timestamp || ''],
  ];
  body.innerHTML = '<h2>Provenance</h2><dl>' + rows.map(([k, val]) => '<dt>' + esc(k) + '</dt><dd>' + val + '</dd>').join('') + '</dl>';
  panel.classList.add('open');
  document.querySelectorAll('.tm-val.active').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.tm-val[data-id="' + id + '"]').forEach(e => e.classList.add('active'));
}}
document.querySelectorAll('[data-id]').forEach(el => {{
  el.addEventListener('click', () => show(el.dataset.id));
  el.addEventListener('keydown', e => {{ if (e.key === 'Enter') show(el.dataset.id); }});
}});
document.getElementById('close').addEventListener('click', () => panel.classList.remove('open'));
</script></body></html>
"""


def render_viewer(store: ValueStore, template_path: Path, build_dir: Path, out_path: Path) -> Path:
    """Render the Markdown template to a standalone HTML page with a provenance panel."""
    build_dir = Path(build_dir)
    renderer = HtmlRenderer(store, build_dir)
    scratch = build_dir / ".viewer.md"
    renderer.render(template_path, scratch)
    body = markdown_to_html(scratch.read_text(encoding="utf-8"))
    scratch.unlink()
    manifest = store.to_manifest().model_dump(mode="json")
    page = PAGE.format(
        title=html.escape(store.project),
        body=body,
        manifest=json.dumps(manifest).replace("</", "<\\/"),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path
