from __future__ import annotations

import numpy as np
import pytest

from specsim.sampling import (
    MarkovPair,
    acceptance_probability,
    chi_square_sf,
    lossless_check,
    sample_target,
    speculative_sample,
)


def test_acceptance_probability_is_one_minus_tv() -> None:
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    tv = 0.5 * np.abs(p - q).sum()
    assert acceptance_probability(p, q) == pytest.approx(1 - tv)


def test_speculative_sampling_is_lossless() -> None:
    rep = lossless_check(vocab=16, gamma=4, divergence=1.0, samples=60_000, seed=1)
    assert rep["passed"], rep


def test_check_catches_missing_residual_correction() -> None:
    """Negative control: resampling from P instead of max(0, P - Q) biases the output."""
    rep = lossless_check(vocab=16, gamma=4, divergence=1.5, samples=60_000, seed=1, residual=False)
    assert not rep["passed"], rep


def test_acceptance_matches_one_minus_tv_per_context() -> None:
    rep = lossless_check(vocab=24, gamma=3, divergence=0.8, samples=60_000, seed=2)
    assert rep["acceptance_rate"] == pytest.approx(rep["expected_acceptance"], abs=0.005)


@pytest.mark.parametrize(
    ("divergence", "lo", "hi"), [(0.0, 0.999, 1.0), (0.5, 0.6, 0.95), (3.0, 0.0, 0.5)]
)
def test_acceptance_falls_with_divergence(divergence: float, lo: float, hi: float) -> None:
    pair = MarkovPair.random(32, divergence, seed=0)
    _, stats = speculative_sample(pair, 5000, 8, 4, np.random.default_rng(0))
    assert lo <= stats["acceptance_rate"] <= hi


def test_identical_models_accept_everything() -> None:
    pair = MarkovPair.random(8, 0.0, seed=0)
    # 60 tokens = exactly 10 full cycles of gamma + 1 = 6 tokens.
    _, stats = speculative_sample(pair, 2000, 60, 5, np.random.default_rng(0))
    assert stats["acceptance_rate"] == 1.0
    assert stats["tokens_per_cycle"] == pytest.approx(6.0)


def test_target_sampler_matches_marginals() -> None:
    pair = MarkovPair.random(10, 1.0, seed=3)
    seqs = sample_target(pair, 50_000, 3, np.random.default_rng(3))
    emp = np.bincount(seqs[:, 2], minlength=10) / 50_000
    assert np.allclose(emp, pair.marginals(3)[2], atol=0.01)


def test_chi_square_sf_matches_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    for dof in (5, 15, 60, 300):
        for x in (0.5 * dof, dof, 1.5 * dof, 2 * dof):
            assert chi_square_sf(x, dof) == pytest.approx(stats.chi2.sf(x, dof), abs=0.01)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        MarkovPair.random(1, 1.0)
    with pytest.raises(ValueError):
        speculative_sample(MarkovPair.random(4, 1.0), 10, 4, 0, np.random.default_rng())
