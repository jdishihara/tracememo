"""HTML viewer: Markdown conversion and the provenance-bearing page."""

from pathlib import Path

from typer.testing import CliRunner

from tracememo.cli import app
from tracememo.report.viewer import markdown_to_html

runner = CliRunner()


def test_markdown_to_html_basics() -> None:
    md = (
        '# Title\n\nA *para* with **bold** and `code` and [link](http://x "t").\n\n'
        "- one\n- two\n\n1. a\n2. b\n\n| h1 | h2 |\n| --- | --- |\n| 1 | 2 |\n\n"
        '```\nraw < code\n```\n\n<a id="x"></a>\n![alt](img.png)\n'
    )
    out = markdown_to_html(md)
    assert "<h1>Title</h1>" in out
    assert "<em>para</em>" in out and "<strong>bold</strong>" in out and "<code>code</code>" in out
    assert '<a href="http://x" title="t">link</a>' in out
    assert "<ul><li>one</li><li>two</li></ul>" in out and "<ol><li>a</li><li>b</li></ol>" in out
    assert (
        "<table><thead><tr><th>h1</th><th>h2</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr>"
        in out
    )
    assert "<pre><code>raw &lt; code</code></pre>" in out
    assert '<a id="x"></a>' in out and '<img alt="alt" src="img.png">' in out
    # An inline span on its own line stays inside the paragraph.
    para = markdown_to_html('was\n<span class="tm-val" data-id="a.b">7.3</span>\n(median).')
    assert para.count("<p>") == 1
    assert 'was <span class="tm-val" data-id="a.b">7.3</span> (median).' in para


def test_viewer_page(project_dir: Path) -> None:
    cfg = project_dir / "project.yaml"
    assert runner.invoke(app, ["build", "--config", str(cfg)]).exit_code == 0
    res = runner.invoke(app, ["viewer", "--config", str(cfg)])
    assert res.exit_code == 0, res.output
    page = (project_dir / "build" / "report.html").read_text(encoding="utf-8")
    assert 'class="tm-val" data-id="drone.loc_err.mean_cm"' in page
    assert 'data-id="drone.loc_err.error_vs_time"' in page and "data:image/png;base64," in page
    assert 'data-id="drone.failures.events"' in page and "<td>dropout</td>" in page
    assert '<script id="manifest" type="application/json">' in page
    assert '"analysis_source_sha256"' in page
    assert "## Provenance" not in page  # appendix replaced by the interactive panel
    assert not (project_dir / "build" / ".viewer.md").exists()
