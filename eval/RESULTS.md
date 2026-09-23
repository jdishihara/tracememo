# Evaluation results

Generated 2026-09-23 at commit `d8f6a26` by `python eval/run_all.py` (seed 0). Raw numbers are in `eval/results/*.json`.

## Reproduction test

| pipeline | values | passed | max_rel_err |
| --- | --- | --- | --- |
| drone (normalized tables) | 20 | 20 | 0 |
| drone (ArduPilot .bin + Marvelmind CSV) | 20 | 20 | 0.005 |
| rag (Langfuse export + eval CSV) | 57 | 57 | 4.222e-06 |

## Checker recall

| error_type | expected_check | injected | caught | recall |
| --- | --- | --- | --- | --- |
| swap_number | raw_number | 100 | 100 | 1 |
| invent_statistic | raw_number | 100 | 100 | 1 |
| flip_comparison | comparison | 100 | 100 | 1 |
| unknown_reference | unknown_reference | 100 | 100 | 1 |
| stale_reference | stale | 100 | 100 | 1 |

False positives on clean drafts: 0 of 100 documents (0.0%).

## Staleness demo

Perturbation: `beacon.parquet x_m += 0.02 m`

| analysis | status | expected |
| --- | --- | --- |
| drone.loc_err | reran | reran |
| drone.trajectory | cached | cached |
| drone.failures | reran | reran |

Values changed: 12; values unchanged: 13 (all `drone.trajectory.*` values unchanged and served from cache).

Changed values: `drone.loc_err.max_3d_cm`, `drone.loc_err.max_cm`, `drone.loc_err.mean_3d_cm`, `drone.loc_err.mean_cm`, `drone.loc_err.median_3d_cm`, `drone.loc_err.median_cm`, `drone.loc_err.p95_3d_cm`, `drone.loc_err.p95_cm`, `drone.loc_err.pct_under_10cm`, `drone.loc_err.pct_under_10cm_3d`, `drone.loc_err.rmse_3d_cm`, `drone.loc_err.rmse_cm`

## Drafting grounding rate

Not run: set `ANTHROPIC_API_KEY` and run `python eval/drafting.py --n 20` (or `--fake` for a dry run).
