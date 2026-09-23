"""Checker recall on injected errors and false positives on clean drafts."""

from __future__ import annotations

import argparse
import sys

from _common import RESULTS_DIR, work_dir

from tracememo.evaluation import checker_recall
from tracememo.evaluation.common import write_results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-docs", type=int, default=50, help="clean docs per project")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    result = checker_recall.run(work_dir("recall"), n_docs=args.n_docs, seed=args.seed)
    write_results("checker_recall", result, RESULTS_DIR)
    print(checker_recall.to_markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
