"""Evaluation experiments run end to end at small scale."""

from pathlib import Path

import pytest

from tracememo.evaluation import checker_recall, drafting, reproduction, staleness
from tracememo.evaluation.common import build_project, load_store, prepare_example, synth_drone
from tracememo.evaluation.drafting import match_value, score_freeform, score_placeholder
from tracememo.llm import FakeLLM


@pytest.fixture(scope="module")
def drone_store(tmp_path_factory: pytest.TempPathFactory):
    work = tmp_path_factory.mktemp("eval")
    cfg = build_project(prepare_example("drone_memo", work, synth_drone(work / "d", 0, 60.0)))
    return load_store(cfg)


def test_reproduction_all_pass(tmp_path: Path) -> None:
    result = reproduction.run(tmp_path, seed=3)
    assert result["all_pass"], [r for r in result["rows"] if not r["pass"]]
    assert {s["pipeline"] for s in result["summary"]} == {
        "drone (normalized tables)",
        "drone (ArduPilot .bin + Marvelmind CSV)",
        "rag (Langfuse export + eval CSV)",
    }
    assert "| pipeline |" in reproduction.to_markdown(result)


def test_checker_recall_catches_everything(tmp_path: Path) -> None:
    result = checker_recall.run(tmp_path, n_docs=8, seed=1)
    for row in result["combined"]:
        assert row["recall"] == 1.0, row
    assert result["false_positive_docs"] == 0
    assert "False positives" in checker_recall.to_markdown(result)


def test_staleness_reruns_only_dependents(tmp_path: Path) -> None:
    result = staleness.run(tmp_path, seed=2)
    assert result["as_expected"], result["analyses"]
    assert all(not v.startswith("drone.trajectory") for v in result["changed_values"])
    assert any(v.startswith("drone.loc_err") for v in result["changed_values"])
    assert "drone.trajectory" in staleness.to_markdown(result)


def test_match_value(drone_store) -> None:
    manifest = drone_store.to_manifest()
    mean = drone_store.get_value("drone.loc_err.mean_cm")
    assert match_value(mean.formatted(), manifest) is not None
    assert match_value("123456.7", manifest) is None
    assert match_value(f"{float(mean.value) * 1.5:.1f}", manifest) != "drone.loc_err.mean_cm"


def test_scoring(drone_store) -> None:
    manifest = drone_store.to_manifest()
    mean = drone_store.get_value("drone.loc_err.mean_cm")
    p95 = drone_store.get_value("drone.loc_err.p95_cm")
    ph = (
        "The mean was {{val:drone.loc_err.mean_cm}} and 42 samples failed. "
        "The p95 {{val:drone.loc_err.p95_cm}} was lower than the mean "
        "{{val:drone.loc_err.mean_cm}}.\n"
    )
    s = score_placeholder(ph, manifest)
    assert s.n_numbers == 4 and s.n_unsupported == 1 and s.n_wrong_comparisons == 1
    ff = (
        f"The mean was {mean.formatted()} cm and 999.9 cm was invented. "
        f"The p95 of {p95.formatted()} cm was lower than the mean of {mean.formatted()} cm.\n"
    )
    s = score_freeform(ff, manifest)
    assert s.n_numbers == 4 and s.n_unsupported == 1
    assert s.n_comparisons == 1 and s.n_wrong_comparisons == 1


def test_drafting_run_with_fake(drone_store) -> None:
    client = drafting.fake_drafter(drone_store, seed=0, error_rate=0.5)
    result = drafting.run(client, drone_store, n=4, fix_rounds=1)
    assert [s["condition"] for s in result["summary"]][2] == "free-form numbers"
    assert result["summary"][0]["drafts"] == 4 and result["summary"][2]["numbers"] > 0
    assert result["summary"][2]["unsupported_numbers"] > 0  # the fake plants wrong numbers
    assert "Model: `fake-drafter`" in drafting.to_markdown(result)
    assert isinstance(client, FakeLLM) and len(client.calls) >= 8
