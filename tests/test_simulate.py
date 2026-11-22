from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from specsim import ConstantLatency, LinearLatency, expected_speedup, simulate


@pytest.mark.parametrize(
    ("alpha", "gamma", "c"),
    [(0.6, 4, 0.05), (0.8, 4, 0.05), (0.9, 8, 0.05), (0.8, 4, 0.5), (0.95, 3, 0.2)],
)
def test_converges_to_closed_form(alpha: float, gamma: int, c: float) -> None:
    lat = ConstantLatency.from_ratio(c)
    r = simulate(alpha, gamma, lat, tokens=20_000, trials=40, seed=1)
    assert r.speedup == pytest.approx(expected_speedup(alpha, gamma, lat), rel=0.01)


def test_ci_covers_long_horizon_truth() -> None:
    lat = LinearLatency(0.02, 0.0005, 0.003)
    truth = simulate(0.75, 5, lat, tokens=256, trials=40_000, seed=123).speedup
    r = simulate(0.75, 5, lat, tokens=256, trials=2000, seed=7)
    lo, hi = r.speedup_ci95
    assert lo <= truth <= hi
    assert r.speedup_p05 <= r.speedup <= r.speedup_p95


def test_seed_reproducible() -> None:
    lat = ConstantLatency.from_ratio(0.1)
    assert simulate(0.7, 4, lat, seed=3) == simulate(0.7, 4, lat, seed=3)
    assert simulate(0.7, 4, lat, seed=3) != simulate(0.7, 4, lat, seed=4)


def test_zero_acceptance() -> None:
    r = simulate(0.0, 4, ConstantLatency.from_ratio(0.1), tokens=100, trials=10, seed=0)
    assert r.tokens_per_cycle == 1.0
    assert r.acceptance_rate == 0.0
    assert r.draft_efficiency == 0.0
    assert r.target_calls_saved == 0.0
    assert r.speedup < 1.0


def test_acceptance_rate_estimates_alpha() -> None:
    r = simulate(0.8, 4, ConstantLatency.from_ratio(0.05), tokens=5000, trials=20, seed=0)
    assert r.acceptance_rate == pytest.approx(0.8, abs=0.01)
    # Only (1 - a^g)/(1 - a) * a / g of drafted tokens survive on average.
    assert r.draft_efficiency == pytest.approx(0.8 * (1 - 0.8**4) / 0.2 / 4, abs=0.01)


def test_full_acceptance() -> None:
    r = simulate(1.0, 4, ConstantLatency.from_ratio(0.05), tokens=1000, trials=5, seed=0)
    assert r.acceptance_rate == 1.0
    assert r.tokens_per_cycle == pytest.approx(5.0, rel=0.01)
    assert r.target_calls_saved == pytest.approx(0.8, abs=0.01)


def test_gamma_zero_is_exactly_baseline() -> None:
    r = simulate(0.9, 0, ConstantLatency(0.02, 0.002), tokens=64, trials=3, seed=0)
    assert r.speedup == pytest.approx(1.0)
    assert r.baseline_time == pytest.approx(64 * 0.02)


def test_never_overshoots_length() -> None:
    # tokens < gamma: the last cycles shrink so exactly `tokens` are produced.
    r = simulate(1.0, 8, ConstantLatency.from_ratio(0.05), tokens=3, trials=4, seed=0)
    assert r.tokens_per_cycle == pytest.approx(3.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"tokens": 0}, {"trials": 0}, {"gamma": -1}, {"batch": 0}],
)
def test_validation(kwargs: dict[str, int]) -> None:
    args = {"acceptance": 0.5, "gamma": 4, "latency": ConstantLatency(1, 0.1), **kwargs}
    with pytest.raises(ValueError):
        simulate(**args)  # type: ignore[arg-type]


@settings(max_examples=40, deadline=None)
@given(
    alpha=st.floats(0, 1),
    gamma=st.integers(0, 10),
    c=st.floats(0.01, 1.0),
    tokens=st.integers(1, 300),
    seed=st.integers(0, 2**32 - 1),
)
def test_speedup_within_theoretical_bounds(
    alpha: float, gamma: int, c: float, tokens: int, seed: int
) -> None:
    r = simulate(alpha, gamma, ConstantLatency.from_ratio(c), tokens=tokens, trials=20, seed=seed)
    assert 1 / (gamma * c + 1) - 1e-9 <= r.speedup <= (gamma + 1) / (gamma * c + 1) + 1e-9
    assert 1.0 <= r.tokens_per_cycle <= gamma + 1
