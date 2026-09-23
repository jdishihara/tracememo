# Evaluation scripts

Each script writes `eval/results/<name>.json` and prints a Markdown summary; `run_all.py`
runs them all and rewrites `eval/RESULTS.md`.

| Script | What it measures | Needs API key |
|---|---|---|
| `reproduction.py` | Every computed value vs. synthetic ground truth (normalized tables, raw adapters, RAG) | no |
| `checker_recall.py` | Fraction of injected errors the deterministic checker catches, per error type, and the false-positive rate on clean drafts | no |
| `staleness.py` | Change one flight-log table, rebuild, show what was recomputed vs. cached | no |
| `drafting.py` | Grounding rate of LLM drafts: placeholders required vs. free-form numbers | yes (`--fake` for a dry run) |

```
.venv/bin/python eval/run_all.py                 # deterministic experiments (+ drafting if ANTHROPIC_API_KEY is set)
.venv/bin/python eval/drafting.py --n 20         # drafting experiment only
.venv/bin/python eval/drafting.py --fake --n 5   # scripted stand-in, no API calls
```
