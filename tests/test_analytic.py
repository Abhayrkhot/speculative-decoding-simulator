from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from specsim import (
    ConstantLatency,
    LinearLatency,
    expected_speedup,
    expected_tokens_per_cycle,
    leviathan_speedup,
    optimal_gamma,
)


def test_tokens_per_cycle_closed_form() -> None:
    assert expected_tokens_per_cycle(0.8, 4) == pytest.approx((1 - 0.8**5) / 0.2)
    assert expected_tokens_per_cycle(0.0, 7) == 1.0
    assert expected_tokens_per_cycle(1.0, 7) == 8.0
    assert expected_tokens_per_cycle(0.5, 0) == 1.0


def test_positional_reduces_to_iid() -> None:
    assert expected_tokens_per_cycle([0.7] * 5, 5) == pytest.approx(
        expected_tokens_per_cycle(0.7, 5)
    )
    # 1 + a1 + a1*a2
    assert expected_tokens_per_cycle([0.9, 0.5], 2) == pytest.approx(1 + 0.9 + 0.45)


def test_leviathan_reference_values() -> None:
    # Leviathan et al. (2023), Thm 3.8.
    assert leviathan_speedup(0.8, 4, 0.05) == pytest.approx(3.3616 / 1.2)
    assert leviathan_speedup(0.6, 4, 0.5) == pytest.approx((1 - 0.6**5) / 0.4 / 3.0)


def test_one_target_pass_per_cycle() -> None:
    """Regression test for the original bug (two target passes charged per cycle).

    With perfect acceptance and a free draft, a cycle of gamma drafts costs one
    target pass and yields gamma + 1 tokens.
    """
    lat = ConstantLatency(1.0, 1e-9)
    assert expected_speedup(1.0, 4, lat) == pytest.approx(5.0, rel=1e-6)


def test_gamma_zero_is_baseline() -> None:
    assert expected_speedup(0.9, 0, ConstantLatency.from_ratio(0.1)) == 1.0


def test_optimal_gamma_matches_brute_force() -> None:
    lat = ConstantLatency.from_ratio(0.05)
    g, s = optimal_gamma(0.8, lat, max_gamma=20)
    brute = max(range(1, 21), key=lambda k: expected_speedup(0.8, k, lat))
    assert g == brute
    assert s == pytest.approx(expected_speedup(0.8, brute, lat))


def test_optimal_gamma_declines_to_speculate_when_hopeless() -> None:
    assert optimal_gamma(0.0, ConstantLatency.from_ratio(0.1)) == (0, 1.0)
    assert optimal_gamma(0.3, ConstantLatency.from_ratio(0.9))[0] == 0


def test_verification_cost_growth_lowers_best_gamma() -> None:
    flat = LinearLatency(0.02, 0.0, 0.002)
    steep = LinearLatency(0.02, 0.004, 0.002)
    assert optimal_gamma(0.85, steep)[0] < optimal_gamma(0.85, flat)[0]


@pytest.mark.parametrize("gamma", [-1, 1.5])
def test_rejects_bad_gamma(gamma: float) -> None:
    with pytest.raises(ValueError):
        expected_tokens_per_cycle(0.5, gamma)  # type: ignore[arg-type]


def test_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        expected_tokens_per_cycle(1.2, 3)


@given(
    a1=st.floats(0, 1),
    a2=st.floats(0, 1),
    gamma=st.integers(1, 16),
    c=st.floats(0.001, 1.0),
)
def test_speedup_monotone_in_alpha_and_bounded(a1: float, a2: float, gamma: int, c: float) -> None:
    lo, hi = sorted((a1, a2))
    lat = ConstantLatency.from_ratio(c)
    s_lo, s_hi = expected_speedup(lo, gamma, lat), expected_speedup(hi, gamma, lat)
    assert s_lo <= s_hi + 1e-12
    assert 1 / (gamma * c + 1) - 1e-12 <= s_lo
    assert s_hi <= (gamma + 1) / (gamma * c + 1) + 1e-12
    assert math.isfinite(s_hi)
