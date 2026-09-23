"""Run every evaluation and rewrite eval/RESULTS.md."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from _common import RESULTS_DIR, work_dir

from tracememo.evaluation import checker_recall, drafting, reproduction, staleness
from tracememo.evaluation.common import (
    build_project,
    load_store,
    prepare_example,
    synth_drone,
    write_results,
)
from tracememo.llm import make_client

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-docs", type=int, default=50)
    ap.add_argument("--n-drafts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-drafting", action="store_true")
    args = ap.parse_args()

    sections = []
    rep = reproduction.run(work_dir("reproduction"), seed=args.seed)
    write_results("reproduction", rep, RESULTS_DIR)
    sections.append(("Reproduction test", reproduction.to_markdown(rep)))

    rec = checker_recall.run(work_dir("recall"), n_docs=args.n_docs, seed=args.seed)
    write_results("checker_recall", rec, RESULTS_DIR)
    sections.append(("Checker recall", checker_recall.to_markdown(rec)))

    st = staleness.run(work_dir("staleness"), seed=args.seed)
    write_results("staleness", st, RESULTS_DIR)
    sections.append(("Staleness demo", staleness.to_markdown(st)))

    have_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("TRACEMEMO_FAKE_LLM"))
    if have_key and not args.skip_drafting:
        work = work_dir("drafting")
        cfg = build_project(
            prepare_example("drone_memo", work, synth_drone(work / "data" / "drone", args.seed))
        )
        store = load_store(cfg)
        dr = drafting.run(make_client(cfg.llm), store, n=args.n_drafts)
        dr["fake"] = False
        write_results("drafting", dr, RESULTS_DIR)
        sections.append(("Drafting grounding rate", drafting.to_markdown(dr)))
    else:
        prev = RESULTS_DIR / "drafting.json"
        if prev.exists():
            import json

            dr = json.loads(prev.read_text())
            sections.append(("Drafting grounding rate (previous run)", drafting.to_markdown(dr)))
        else:
            sections.append(
                (
                    "Drafting grounding rate",
                    "Not run: set `ANTHROPIC_API_KEY` and run "
                    "`python eval/drafting.py --n 20` (or `--fake` for a dry run).",
                )
            )

    commit = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE.parent
        ).stdout.strip()
        or "unknown"
    )
    out = [
        f"# Evaluation results\n\nGenerated {datetime.now(UTC):%Y-%m-%d} at commit `{commit}` by "
        f"`python eval/run_all.py` (seed {args.seed}). Raw numbers are in `eval/results/*.json`.\n"
    ]
    for title, body in sections:
        out.append(f"## {title}\n\n{body}\n")
    (HERE / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    return 0 if rep["all_pass"] and st["as_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
