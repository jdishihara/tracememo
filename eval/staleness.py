"""Staleness demo: perturb one input table and show what the rebuild recomputes."""

from __future__ import annotations

import argparse
import sys

from _common import RESULTS_DIR, work_dir

from tracememo.evaluation import staleness
from tracememo.evaluation.common import write_results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    result = staleness.run(work_dir("staleness"), seed=args.seed)
    write_results("staleness", result, RESULTS_DIR)
    print(staleness.to_markdown(result))
    return 0 if result["as_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
