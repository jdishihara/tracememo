"""Drafting grounding rate: placeholders required vs. free-form numbers."""

from __future__ import annotations

import argparse
import os
import sys

from _common import RESULTS_DIR, work_dir

from tracememo.evaluation import drafting
from tracememo.evaluation.common import (
    build_project,
    load_store,
    prepare_example,
    synth_drone,
    write_results,
)
from tracememo.llm import make_client


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="drafts per condition")
    ap.add_argument("--fix-rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="use the scripted stand-in (no API calls)")
    args = ap.parse_args()
    work = work_dir("drafting")
    cfg = build_project(
        prepare_example("drone_memo", work, synth_drone(work / "data" / "drone", args.seed))
    )
    store = load_store(cfg)
    if args.fake:
        client = drafting.fake_drafter(store, seed=args.seed)
    elif not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("TRACEMEMO_FAKE_LLM"):
        print("ANTHROPIC_API_KEY is not set; pass --fake for a dry run", file=sys.stderr)
        return 2
    else:
        client = make_client(cfg.llm)
    result = drafting.run(client, store, n=args.n, fix_rounds=args.fix_rounds)
    result["fake"] = bool(args.fake)
    name = "drafting_fake" if args.fake else "drafting"
    write_results(name, result, RESULTS_DIR)
    print(drafting.to_markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
