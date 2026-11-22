"""Calibration pipeline (on synthetic benchmark data with known ground truth) and CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from specsim.calibrate import calibration_report, fit_latency, measured_speedups, validate
from specsim.cli import main

from .conftest import TRUE_ALPHAS, TRUE_LATENCY


def test_fit_latency_recovers_profile(bench_results: dict[str, Any]) -> None:
    lat, ctx = fit_latency(bench_results)
    assert ctx == 256
    assert lat.target_fixed_s == pytest.approx(TRUE_LATENCY.target_fixed_s, rel=1e-6)
    assert lat.target_per_token_s == pytest.approx(TRUE_LATENCY.target_per_token_s, rel=1e-6)


def test_tabulated_latency_interpolates_between_contexts() -> None:
    from specsim.calibrate import tabulated_latency

    res = {
        "latency_ms": {
            "256": {"target": {"1": 10.0, "4": 12.0}, "draft": {"1": 2.0}},
            "1024": {"target": {"1": 14.0, "4": 16.0}, "draft": {"1": 3.0}},
        },
        "runs": [],
    }
    lat, ctx = tabulated_latency(res, context=640)
    assert ctx == 640
    assert lat.target(1) == pytest.approx(0.012)
    assert lat.target(4) == pytest.approx(0.014)
    assert lat.draft(1) == pytest.approx(0.0025)
    lat_hi, _ = tabulated_latency(res, context=5000)
    assert lat_hi.target(1) == pytest.approx(0.014)


def test_calibration_report_recovers_acceptance(bench_results: dict[str, Any]) -> None:
    rep = calibration_report(bench_results)
    by_task = {r["task"]: r for r in rep["acceptance"]}
    for task, alphas in TRUE_ALPHAS.items():
        assert by_task[task]["positional"][0] == pytest.approx(alphas[0], abs=0.03)


def test_greedy_match() -> None:
    from specsim.calibrate import greedy_match

    def run(g: int, toks: list[int], pid: int = 0) -> dict[str, Any]:
        return {"task": "t", "prompt_id": pid, "gamma": g, "temperature": 0.0, "tokens": toks}

    res = {"runs": [run(0, [1, 2, 3, 4]), run(2, [1, 2, 3, 4]), run(4, [1, 2, 9, 9])]}
    m = greedy_match(res)
    assert m["runs"] == 2
    assert m["exact_match_rate"] == 0.5
    assert m["mean_matching_prefix"] == pytest.approx((1.0 + 0.5) / 2)


def test_measured_speedups_cover_all_spec_runs(bench_results: dict[str, Any]) -> None:
    m = measured_speedups(bench_results)
    assert set(m) == {(t, 0.0, g) for t in TRUE_ALPHAS for g in (1, 2, 4, 6)}
    assert m[("code", 0.0, 4)] > m[("chat", 0.0, 4)]


def test_validate_predicts_out_of_sample(bench_results: dict[str, Any]) -> None:
    rep = validate(bench_results, trials=1000)
    assert rep["summary"]["n"] == 8
    assert rep["summary"]["mae_pct"] < 4.0, rep["rows"]
    assert len(rep["gamma_choice"]) == 2
    assert 0.0 <= rep["summary"]["gamma_regret_mean_pct"] < 5.0
    iid = validate(bench_results, acceptance="iid", trials=1000)
    # Position-dependent acceptance is the truth here, so it should predict better.
    assert rep["summary"]["mae_pct"] <= iid["summary"]["mae_pct"]


def test_cycle_overhead_is_learned_from_other_gammas() -> None:
    from .conftest import synthetic_results

    res = synthetic_results(seed=3, cycle_overhead_s=0.006)
    rep = validate(res, trials=300)
    s = rep["summary"]
    # Residual error is acceptance sampling noise (6 prompts x 160 tokens).
    assert s["mae_pct"] < 6.0, rep["rows"]
    assert s["mae_raw_pct"] > s["mae_pct"] + 5.0
    assert all(r["cycle_overhead_ms"] == pytest.approx(6.0, rel=0.25) for r in rep["rows"])


def test_load_rejects_wrong_json(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text("{}")
    with pytest.raises(ValueError):
        validate(p)


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    code = main(list(argv))
    return code, capsys.readouterr().out


def test_cli_simulate_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(
        capsys,
        "simulate",
        "--alpha",
        "0.8",
        "--gamma",
        "4",
        "--c",
        "0.05",
        "--tokens",
        "2000",
        "--trials",
        "200",
        "--seed",
        "0",
        "--json",
    )
    assert code == 0
    data = json.loads(out)
    assert data["speedup"] == pytest.approx(data["analytic_speedup"], rel=0.02)
    assert data["analytic_speedup"] == pytest.approx(2.8013, rel=1e-3)


def test_cli_simulate_text_and_ms_latency(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(capsys, "simulate", "--target-ms", "25", "--draft-ms", "2", "--seed", "1")
    assert code == 0 and "speedup" in out


def test_cli_roofline_and_batch(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(
        capsys,
        "optimal-gamma",
        "--alpha",
        "0.8",
        "--roofline",
        "llama-3.1-70b",
        "llama-3.2-1b",
        "--gpu",
        "h100-sxm",
        "--batch",
        "1",
        "--json",
    )
    assert code == 0
    assert json.loads(out)[0]["best_gamma"] >= 4


def test_cli_sweep_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_csv = tmp_path / "sweep.csv"
    code, _ = _run(
        capsys,
        "sweep",
        "--alpha",
        "0.6:0.8:0.1",
        "--gamma",
        "1:3",
        "--trials",
        "50",
        "--csv",
        str(out_csv),
    )
    assert code == 0
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == 9
    assert {"alpha", "gamma", "analytic", "simulated", "ci_low", "ci_high"} <= set(rows[0])


def test_cli_calibrate_and_validate(bench_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(capsys, "calibrate", str(bench_file))
    assert code == 0 and "code" in out
    code, out = _run(capsys, "validate", str(bench_file), "--json")
    assert code == 0 and json.loads(out)["summary"]["n"] == 8


def test_cli_lossless(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(capsys, "lossless-check", "--vocab", "12", "--samples", "20000")
    assert code == 0 and json.loads(out)["passed"]


def test_cli_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["simulate", "--target-ms", "20"])
    with pytest.raises(SystemExit):
        main(["simulate", "--c", "0.1", "--overhead-ms", "1"])
    code = main(["simulate", "--alpha", "1.5"])
    assert code == 2
    assert "alpha" in capsys.readouterr().err
