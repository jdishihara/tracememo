"""Reproduction test: computed values vs. synthetic ground truth."""

from __future__ import annotations

import argparse
import sys

from _common import RESULTS_DIR, work_dir

from tracememo.evaluation import reproduction
from tracememo.evaluation.common import write_results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    result = reproduction.run(work_dir("reproduction"), seed=args.seed)
    write_results("reproduction", result, RESULTS_DIR)
    print(reproduction.to_markdown(result))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
