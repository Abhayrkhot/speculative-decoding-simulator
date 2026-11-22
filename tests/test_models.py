"""Latency and acceptance models."""

from __future__ import annotations

import numpy as np
import pytest

from specsim import (
    HARDWARE,
    MODELS,
    ConstantLatency,
    IIDAcceptance,
    LinearLatency,
    PositionalAcceptance,
    RooflineLatency,
    TabulatedLatency,
    expected_speedup,
    optimal_gamma,
)


def test_constant_latency() -> None:
    lat = ConstantLatency.from_ratio(0.1, target_s=0.02)
    assert lat.c == pytest.approx(0.1)
    assert lat.target(9, batch=4) == 0.02
    with pytest.raises(ValueError):
        ConstantLatency(0.0, 0.1)


def test_linear_fit_recovers_parameters() -> None:
    true = LinearLatency(0.021, 0.0003, 0.004, 0.0001)
    t_prof = {k: true.target(k) for k in (1, 2, 4, 8, 16)}
    d_prof = {k: true.draft(k) for k in (1, 2, 4)}
    fit = LinearLatency.fit(t_prof, d_prof)
    assert fit.target_fixed_s == pytest.approx(0.021)
    assert fit.target_per_token_s == pytest.approx(0.0003)
    assert fit.draft_fixed_s == pytest.approx(0.004)


def test_linear_fit_clamps_negative_slope() -> None:
    fit = LinearLatency.fit({1: 0.02, 2: 0.0199, 4: 0.0198}, {1: 0.003})
    assert fit.target_per_token_s == 0.0
    assert fit.draft_per_token_s == 0.0


def test_tabulated_latency_interpolates_and_extrapolates() -> None:
    lat = TabulatedLatency.from_profiles({1: 0.020, 4: 0.020, 8: 0.024}, {1: 0.004})
    assert lat.target(1) == pytest.approx(0.020)
    assert lat.target(6) == pytest.approx(0.022)
    assert lat.target(12) == pytest.approx(0.028)  # slope of last segment
    assert lat.draft(3) == pytest.approx(0.004)
    assert lat.with_overhead(0.001).target(1) == pytest.approx(0.021)
    with pytest.raises(ValueError):
        TabulatedLatency.from_profiles({}, {1: 0.004})


def test_roofline_memory_bound_at_batch_one() -> None:
    lat = RooflineLatency(
        MODELS["llama-3.1-8b"], MODELS["llama-3.2-1b"], HARDWARE["h100-sxm"], context=0
    )
    expected = MODELS["llama-3.1-8b"].weight_bytes / (3.35e12 * 0.8)
    assert lat.target(1) == pytest.approx(expected)
    # Verifying a handful of tokens is ~free while memory-bound.
    assert lat.target(5) == pytest.approx(lat.target(1))
    assert 100 < lat.critical_tokens() < 1000


def test_roofline_speculation_stops_paying_at_large_batch() -> None:
    lat = RooflineLatency(MODELS["llama-3.1-70b"], MODELS["llama-3.2-1b"], HARDWARE["h100-sxm"])
    small = optimal_gamma(0.8, lat, batch=1)
    large = expected_speedup(0.8, 4, lat, batch=512)
    assert small[1] > 2.0
    assert large < 1.0


def test_roofline_validation() -> None:
    with pytest.raises(ValueError):
        RooflineLatency(
            MODELS["qwen2.5-7b"], MODELS["qwen2.5-0.5b"], HARDWARE["a100-80gb"], bw_efficiency=0
        )


def test_iid_acceptance() -> None:
    assert IIDAcceptance(0.7).alphas(3).tolist() == [0.7, 0.7, 0.7]
    with pytest.raises(ValueError):
        IIDAcceptance(-0.1)


def test_positional_extends_last_value() -> None:
    m = PositionalAcceptance((0.9, 0.8))
    assert m.alphas(4).tolist() == [0.9, 0.8, 0.8, 0.8]
    assert m.alphas(1).tolist() == [0.9]


def test_from_traces_recovers_positional_rates() -> None:
    rng = np.random.default_rng(0)
    true = np.array([0.9, 0.75, 0.6, 0.5])
    cycles = []
    for _ in range(40_000):
        k = int(rng.integers(1, 5))
        hits = rng.random(k) < true[:k]
        cycles.append((k, int(np.cumprod(hits).sum())))
    est = PositionalAcceptance.from_traces(cycles)
    assert np.allclose(est.values, true, atol=0.015)
    assert est.counts[0] == 40_000


def test_from_traces_pooled_fallback_and_errors() -> None:
    est = PositionalAcceptance.from_traces([(4, 2), (4, 4)], min_count=100)
    # Tried 3 + 4 = 7 positions, accepted 6.
    assert est.values == (pytest.approx(6 / 7),)
    with pytest.raises(ValueError):
        PositionalAcceptance.from_traces([(2, 3)])
    with pytest.raises(ValueError):
        PositionalAcceptance.from_traces([])
