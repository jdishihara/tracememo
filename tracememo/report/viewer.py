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


_BLOCK_TAG = re.compile(
    r"^<(?:a\s+id=|figure|div|table|img|pre|hr|section|p\b|h[1-6]|ul|ol|blockquote)"
)


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
        if _BLOCK_TAG.match(stripped) and stripped.endswith(">"):
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


PAGE = r"""<!doctype html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:#070a12; --bg2:#0b1020; --panel:rgba(14,20,38,.72); --line:rgba(120,150,255,.16); --line2:rgba(120,150,255,.32);
  --ink:#e8ecf8; --ink2:#a9b3cf; --muted:#6c7695; --accent:#5ee7ff; --accent2:#8b7bff; --ok:#3ddc97; --warn:#ffb84d;
  --glow:0 0 18px rgba(94,231,255,.35); --radius:14px; --sans:"Space Grotesk",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;
}}
:root[data-theme="light"] {{
  --bg:#f5f7fc; --bg2:#ffffff; --panel:rgba(255,255,255,.8); --line:rgba(40,60,120,.14); --line2:rgba(40,60,120,.3);
  --ink:#0d1330; --ink2:#3c4566; --muted:#7c85a3; --accent:#0a7fd8; --accent2:#6a4cff; --ok:#0f9d6b; --warn:#c77700;
  --glow:0 0 14px rgba(10,127,216,.25);
}}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; font-family:var(--sans); color:var(--ink); background:var(--bg); line-height:1.6; min-height:100vh;
  background-image:
    radial-gradient(1200px 600px at 10% -10%, rgba(139,123,255,.18), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(94,231,255,.14), transparent 55%),
    linear-gradient(var(--line) 1px, transparent 1px), linear-gradient(90deg, var(--line) 1px, transparent 1px);
  background-size:auto, auto, 48px 48px, 48px 48px; background-attachment:fixed; }}
header {{ position:sticky; top:0; z-index:10; backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); background:var(--panel);
  border-bottom:1px solid var(--line); }}
.bar {{ max-width:1080px; margin:0 auto; padding:10px 20px; display:flex; align-items:center; gap:10px 14px; flex-wrap:wrap; }}
.brand {{ font-family:var(--mono); font-size:13px; letter-spacing:.18em; text-transform:uppercase; color:var(--accent); text-shadow:var(--glow); }}
.brand b {{ color:var(--ink); font-weight:500; letter-spacing:0; text-transform:none; text-shadow:none; margin-left:10px; font-family:var(--sans); font-size:15px; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; flex-basis:100%; }}
.chip {{ font-family:var(--mono); font-size:11.5px; padding:4px 10px; border:1px solid var(--line2); border-radius:999px; color:var(--ink2); background:transparent; }}
.chip i {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:7px; vertical-align:middle; background:var(--ok); box-shadow:0 0 8px var(--ok); }}
.chip.cached i {{ background:var(--accent2); box-shadow:0 0 8px var(--accent2); }}
.chip.warn i {{ background:var(--warn); box-shadow:0 0 8px var(--warn); }}
button.toggle {{ margin-left:auto; font:inherit; font-size:12px; color:var(--ink2); background:transparent; border:1px solid var(--line2); border-radius:999px; padding:4px 12px; cursor:pointer; }}
button.toggle:hover {{ border-color:var(--accent); color:var(--accent); }}
main {{ max-width:1080px; margin:0 auto; padding:36px 20px 120px; display:grid; grid-template-columns:minmax(0,1fr); }}
article {{ max-width:780px; }}
.hint {{ font-family:var(--mono); font-size:12px; color:var(--muted); margin:0 0 28px; }}
.hint kbd {{ font:inherit; color:var(--accent); }}
h1 {{ font-size:34px; line-height:1.15; font-weight:600; letter-spacing:-.01em; margin:0 0 20px;
  background:linear-gradient(90deg, var(--ink), var(--accent)); -webkit-background-clip:text; background-clip:text; color:transparent; }}
h2 {{ font-size:20px; font-weight:600; margin:44px 0 12px; padding-top:14px; border-top:1px solid var(--line); position:relative; }}
h2::before {{ content:""; position:absolute; top:-1px; left:0; width:56px; height:2px; background:linear-gradient(90deg,var(--accent),var(--accent2)); box-shadow:var(--glow); }}
h3 {{ font-size:16px; font-weight:600; margin:28px 0 8px; color:var(--ink2); }}
p {{ margin:0 0 14px; }}
.tm-val {{ font-family:var(--mono); font-size:.93em; color:var(--accent); padding:0 5px; border-radius:6px; cursor:pointer;
  background:rgba(94,231,255,.07); border:1px solid transparent; transition:all .15s; white-space:nowrap; }}
.tm-val:hover, .tm-val:focus {{ border-color:var(--accent); box-shadow:var(--glow); outline:none; }}
.tm-val.active {{ background:var(--accent); color:var(--bg); box-shadow:var(--glow); }}
figure {{ margin:28px 0; padding:16px; border:1px solid var(--line); border-radius:var(--radius); background:var(--panel);
  backdrop-filter:blur(10px); box-shadow:0 10px 40px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.04); }}
figure img {{ display:block; max-width:100%; height:auto; border-radius:8px; background:#fff; }}
figcaption {{ color:var(--ink2); font-size:13.5px; margin-top:10px; padding-left:12px; border-left:2px solid var(--accent2); }}
figure.tm-tab figcaption {{ margin:0 0 10px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; font-variant-numeric:tabular-nums; }}
th, td {{ padding:8px 12px; text-align:left; border-bottom:1px solid var(--line); }}
th {{ font-family:var(--mono); font-size:11.5px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:500; }}
td {{ font-family:var(--mono); font-size:13px; }}
tbody tr:hover td {{ background:rgba(94,231,255,.06); }}
ul, ol {{ padding-left:22px; }}
code {{ font-family:var(--mono); font-size:.9em; color:var(--accent2); }}
pre {{ background:var(--bg2); border:1px solid var(--line); border-radius:10px; padding:12px 14px; overflow:auto; }}
a {{ color:var(--accent); }}
aside {{ position:fixed; right:0; top:0; bottom:0; width:min(440px, 94vw); z-index:20; padding:22px 22px 40px; overflow:auto;
  background:var(--panel); backdrop-filter:blur(22px); -webkit-backdrop-filter:blur(22px); border-left:1px solid var(--line2);
  box-shadow:-30px 0 80px rgba(0,0,0,.45); transform:translateX(105%); transition:transform .22s cubic-bezier(.2,.8,.2,1); }}
aside.open {{ transform:none; }}
aside .head {{ display:flex; align-items:center; gap:10px; margin-bottom:14px; }}
aside .head .k {{ font-family:var(--mono); font-size:11px; letter-spacing:.2em; text-transform:uppercase; color:var(--accent); }}
aside .head button {{ margin-left:auto; border:1px solid var(--line2); background:transparent; color:var(--ink2); border-radius:8px; width:30px; height:30px; cursor:pointer; font-size:16px; }}
.big {{ font-family:var(--mono); font-size:30px; color:var(--ink); line-height:1.1; margin:2px 0 4px; text-shadow:var(--glow); }}
.big small {{ font-size:14px; color:var(--ink2); margin-left:6px; }}
.desc {{ color:var(--ink2); font-size:13.5px; margin-bottom:18px; word-break:break-word; }}
.raw {{ font-family:var(--mono); font-size:11.5px; color:var(--muted); margin:-2px 0 6px; }}
.idline {{ font-family:var(--mono); font-size:12px; color:var(--muted); word-break:break-all; margin-bottom:12px; }}
.lineage {{ position:relative; margin:8px 0 18px; padding-left:22px; }}
.lineage::before {{ content:""; position:absolute; left:6px; top:8px; bottom:8px; width:2px; background:linear-gradient(var(--accent2), var(--accent)); opacity:.7; }}
.node {{ position:relative; margin:0 0 12px; font-size:13px; }}
.node::before {{ content:""; position:absolute; left:-20px; top:7px; width:8px; height:8px; border-radius:50%; background:var(--accent); box-shadow:var(--glow); }}
.node .t {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); display:block; }}
.node .v {{ color:var(--ink); word-break:break-word; }}
.hash {{ font-family:var(--mono); font-size:11.5px; color:var(--ink2); background:rgba(139,123,255,.12); border:1px solid var(--line); border-radius:6px; padding:1px 6px; cursor:copy; }}
.hash:hover {{ border-color:var(--accent2); }}
.hash.copied {{ color:var(--ok); border-color:var(--ok); }}
dl {{ display:grid; grid-template-columns:104px 1fr; gap:6px 10px; margin:0; font-size:13px; }}
dt {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); padding-top:3px; }}
dd {{ margin:0; word-break:break-word; color:var(--ink); }}
.empty {{ color:var(--muted); font-size:13px; }}
@media (max-width:720px) {{ h1 {{ font-size:26px; }} }}
</style></head><body>
<header><div class="bar">
  <span class="brand">tracememo <b>{title}</b></span>
  <button class="toggle" id="theme" type="button">light mode</button>
  <div class="chips" id="chips"></div>
</div></header>
<main><article>
<p class="hint">Highlighted numbers are live values. <kbd>Click</kbd> one to trace it back to the code and data that produced it. Generated from <code>manifest.json</code>.</p>
{body}
</article></main>
<aside id="panel" aria-label="provenance">
  <div class="head"><span class="k">provenance</span><button id="close" aria-label="close" type="button">&times;</button></div>
  <div id="panel-body"><p class="empty">Select a value.</p></div>
</aside>
<script id="manifest" type="application/json">{manifest}</script>
<script>
const manifest = JSON.parse(document.getElementById('manifest').textContent);
const panel = document.getElementById('panel'), body = document.getElementById('panel-body');
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
const short = h => (h || '').slice(0, 10);
const hash = (h, title) => '<span class="hash" data-full="' + esc(h || '') + '" title="' + esc(title || 'click to copy') + '">' + esc(short(h) || '-') + '</span>';
const base = p => (p || '').split('/').pop();

// Header chips: counts, commit, per-analysis status.
(function () {{
  const chips = document.getElementById('chips');
  const n = o => Object.keys(o || {{}}).length;
  const a = Object.values(manifest.analyses || {{}});
  const commit = a.length && manifest.values && Object.values(manifest.values)[0]?.provenance?.git_commit;
  const items = [
    '<span class="chip">' + n(manifest.values) + ' values · ' + n(manifest.figures) + ' figures · ' + n(manifest.tables) + ' tables</span>',
    '<span class="chip" title="git commit">' + (commit ? '⌥ ' + esc(commit.slice(0, 8)) : 'no commit') + '</span>',
  ].concat(a.map(r => '<span class="chip ' + (r.cached ? 'cached' : '') + '" title="source ' + esc(short(r.source_sha256)) + '"><i></i>' + esc(r.id) + ' · ' + (r.cached ? 'cached' : 'ran') + '</span>'));
  chips.innerHTML = items.join('');
}})();

function show(id) {{
  const v = manifest.values[id] || manifest.figures[id] || manifest.tables[id];
  if (!v) return;
  const p = v.provenance || {{}};
  const shown = document.querySelector('.tm-val[data-id="' + id + '"]');
  const pretty = shown ? shown.textContent.trim() : String(v.value ?? '');
  const raw = v.value !== undefined && String(v.value) !== pretty.replace(/\s*[a-zA-Z%\/]+$/, '') ? '<div class="raw">raw ' + esc(v.value) + '</div>' : '';
  const value = v.value !== undefined
    ? '<div class="big">' + esc(pretty.replace(/\s*([a-zA-Z%\/]+)$/, '')) + (v.unit ? '<small>' + esc(v.unit) + '</small>' : '') + '</div>' + raw
    : '<div class="big" style="font-size:18px">' + esc(v.caption || '') + '</div>';
  const lineage = [
    ['raw inputs', (p.raw_inputs || []).map(r => esc(base(r.path)) + ' ' + hash(r.sha256, r.path)).join('<br>') || '-'],
    ['input tables', (p.input_tables || []).map(t => esc(t.id) + ' ' + hash(t.sha256)).join('<br>') || '-'],
    ['analysis', esc(p.analysis_id || '-') + ' ' + hash(p.analysis_source_sha256, 'source hash') + (p.params && Object.keys(p.params).length ? '<div class="desc" style="margin:4px 0 0">params ' + esc(JSON.stringify(p.params)) + '</div>' : '')],
    ['this value', '<code>' + esc(id) + '</code>'],
  ];
  body.innerHTML = value + '<div class="desc">' + esc(v.description || '') + '</div>'
    + '<div class="lineage">' + lineage.map(([t, val]) => '<div class="node"><span class="t">' + t + '</span><span class="v">' + val + '</span></div>').join('') + '</div>'
    + '<dl><dt>commit</dt><dd>' + (p.git_commit ? hash(p.git_commit, 'git commit') + (p.git_dirty ? ' <span class="chip warn"><i></i>dirty tree</span>' : '') : 'none') + '</dd>'
    + '<dt>computed</dt><dd>' + esc((p.timestamp || '').replace('T', ' ').slice(0, 19)) + ' UTC</dd>'
    + (v.fmt ? '<dt>format</dt><dd><code>' + esc(v.fmt) + '</code></dd>' : '') + '</dl>';
  panel.classList.add('open');
  document.querySelectorAll('.tm-val.active').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.tm-val[data-id="' + id + '"]').forEach(e => e.classList.add('active'));
  body.querySelectorAll('.hash').forEach(h => h.addEventListener('click', () => {{
    navigator.clipboard?.writeText(h.dataset.full).then(() => {{ h.classList.add('copied'); setTimeout(() => h.classList.remove('copied'), 900); }});
  }}));
}}
document.querySelectorAll('[data-id]').forEach(el => {{
  el.addEventListener('click', () => show(el.dataset.id));
  el.addEventListener('keydown', e => {{ if (e.key === 'Enter') show(el.dataset.id); }});
}});
document.getElementById('close').addEventListener('click', () => panel.classList.remove('open'));
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') panel.classList.remove('open'); }});
const toggle = document.getElementById('theme');
function setTheme(t) {{ document.documentElement.dataset.theme = t; toggle.textContent = t === 'dark' ? 'light mode' : 'dark mode'; try {{ localStorage.setItem('tm-theme', t); }} catch (e) {{}} }}
try {{ setTheme(localStorage.getItem('tm-theme') || 'dark'); }} catch (e) {{ setTheme('dark'); }}
toggle.addEventListener('click', () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
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
